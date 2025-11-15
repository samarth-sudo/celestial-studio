"""
PyBullet Robot Interface

Clean abstraction layer for PyBullet physics simulation, inspired by Isaac Lab's architecture.
Separates physics engine specifics from robot control logic.

Usage:
    from backend.assets.robot_configs import FRANKA_PANDA_CONFIG
    from backend.simulation.pybullet_interface import PyBulletRobotInterface

    robot = PyBulletRobotInterface(FRANKA_PANDA_CONFIG)
    state = robot.get_state()
    robot.set_joint_targets([0.1, 0.2, ...])
    robot.step_simulation()
"""

import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from backend.assets.robot_configs import RobotConfig
from backend.utils.math_utils import normalize_quat


@dataclass
class RobotState:
    """
    Complete robot state at a point in time

    Attributes:
        joint_pos: Joint positions [num_joints]
        joint_vel: Joint velocities [num_joints]
        joint_names: Names of joints
        ee_pos: End-effector position [x, y, z]
        ee_quat: End-effector orientation [w, x, y, z]
        ee_lin_vel: End-effector linear velocity [vx, vy, vz]
        ee_ang_vel: End-effector angular velocity [wx, wy, wz]
        base_pos: Base position [x, y, z]
        base_quat: Base orientation [w, x, y, z]
    """
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_names: List[str]
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    ee_lin_vel: np.ndarray
    ee_ang_vel: np.ndarray
    base_pos: np.ndarray
    base_quat: np.ndarray


class PyBulletRobotInterface:
    """
    Interface between robot control logic and PyBullet physics engine

    Follows Isaac Lab's pattern: clean read/write operations,
    separating simulation state from control commands.
    """

    def __init__(
        self,
        config: RobotConfig,
        physics_client_id: Optional[int] = None,
        use_gui: bool = False
    ):
        """
        Initialize PyBullet robot interface

        Args:
            config: RobotConfig with URDF path and settings
            physics_client_id: Existing PyBullet client ID (or create new)
            use_gui: Whether to show PyBullet GUI
        """
        self.config = config

        # Connect to PyBullet
        if physics_client_id is None:
            if use_gui:
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        else:
            self.physics_client = physics_client_id

        # Physics settings
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1/240.0, physicsClientId=self.physics_client)  # 240 Hz

        # Load robot
        self.robot_id = p.loadURDF(
            config.urdf_path,
            basePosition=config.base_position,
            baseOrientation=p.getQuaternionFromEuler(config.base_orientation),
            useFixedBase=config.fixed_base,
            physicsClientId=self.physics_client
        )

        # Get joint information
        self._setup_joints()

        # Find end-effector link
        self._setup_end_effector()

        # Reset to initial pose
        self.reset_to_initial_pose()

        print(f"✅ Loaded robot: {config.urdf_path}")
        print(f"   Total joints: {self.num_joints}")
        print(f"   Controllable joints: {len(self.joint_indices)}")
        print(f"   End-effector: {config.ee_link_name}")

    def _setup_joints(self):
        """Discover and categorize joints from URDF"""
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)

        # Get all joint info
        self.joint_indices = []
        self.joint_names = []
        self.joint_types = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        self.joint_max_forces = []
        self.joint_max_velocities = []

        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]

            # Only track revolute and prismatic joints (controllable)
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)
                self.joint_types.append(joint_type)
                self.joint_limits_lower.append(joint_info[8])
                self.joint_limits_upper.append(joint_info[9])
                self.joint_max_forces.append(joint_info[10])
                self.joint_max_velocities.append(joint_info[11])

        # Convert to numpy arrays for efficiency
        self.joint_limits_lower = np.array(self.joint_limits_lower)
        self.joint_limits_upper = np.array(self.joint_limits_upper)
        self.joint_max_forces = np.array(self.joint_max_forces)
        self.joint_max_velocities = np.array(self.joint_max_velocities)

        # Get actuator configs for each joint
        self.joint_actuators = []
        for joint_name in self.joint_names:
            try:
                actuator = self.config.get_actuator_for_joint(joint_name)
                self.joint_actuators.append(actuator)
            except ValueError:
                # Use default actuator if no match
                from backend.assets.robot_configs import ActuatorGroupConfig
                self.joint_actuators.append(
                    ActuatorGroupConfig(joint_pattern=".*", max_force=100.0, kp=100.0, kd=10.0)
                )

    def _setup_end_effector(self):
        """Find end-effector link index"""
        self.ee_link_index = None

        for i in range(self.num_joints):
            link_name = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)[12].decode('utf-8')
            if link_name == self.config.ee_link_name:
                self.ee_link_index = i
                break

        if self.ee_link_index is None:
            raise ValueError(f"End-effector link '{self.config.ee_link_name}' not found in URDF")

    def reset_to_initial_pose(self):
        """Reset robot to initial joint configuration"""
        initial_positions = self.config.expand_initial_positions(self.joint_names)

        for joint_idx, joint_name in zip(self.joint_indices, self.joint_names):
            initial_pos = initial_positions.get(joint_name, 0.0)
            p.resetJointState(
                self.robot_id,
                joint_idx,
                initial_pos,
                physicsClientId=self.physics_client
            )

    # ========================================================================
    # State Reading (following Isaac Lab's read pattern)
    # ========================================================================

    def get_state(self) -> RobotState:
        """
        Read complete robot state from simulation

        Returns:
            RobotState with all current state information
        """
        # Get joint states
        joint_states = p.getJointStates(
            self.robot_id,
            self.joint_indices,
            physicsClientId=self.physics_client
        )

        joint_pos = np.array([state[0] for state in joint_states])
        joint_vel = np.array([state[1] for state in joint_states])

        # Get end-effector state
        ee_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeLinkVelocity=1,
            physicsClientId=self.physics_client
        )

        ee_pos = np.array(ee_state[4])  # World position
        ee_quat_xyzw = np.array(ee_state[5])  # World orientation (x,y,z,w)
        ee_quat = np.array([ee_quat_xyzw[3], ee_quat_xyzw[0], ee_quat_xyzw[1], ee_quat_xyzw[2]])  # Convert to (w,x,y,z)
        ee_lin_vel = np.array(ee_state[6])
        ee_ang_vel = np.array(ee_state[7])

        # Apply end-effector offset (TCP)
        from backend.utils.math_utils import quat_apply
        ee_offset = np.array(self.config.ee_offset)
        ee_pos = ee_pos + quat_apply(ee_quat, ee_offset)

        # Get base state
        base_pos, base_quat_xyzw = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client
        )
        base_pos = np.array(base_pos)
        base_quat = np.array([base_quat_xyzw[3], base_quat_xyzw[0], base_quat_xyzw[1], base_quat_xyzw[2]])  # (w,x,y,z)

        return RobotState(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_names=self.joint_names,
            ee_pos=ee_pos,
            ee_quat=normalize_quat(ee_quat),
            ee_lin_vel=ee_lin_vel,
            ee_ang_vel=ee_ang_vel,
            base_pos=base_pos,
            base_quat=normalize_quat(base_quat)
        )

    def get_jacobian(self, local_position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get Jacobian matrix for end-effector

        Args:
            local_position: Position in EE link frame (default: [0,0,0])

        Returns:
            Jacobian matrix [6, num_joints] mapping joint velocities to EE twist
            First 3 rows: linear velocity, Last 3 rows: angular velocity
        """
        if local_position is None:
            local_position = [0, 0, 0]

        state = self.get_state()

        # Calculate Jacobian
        jac_linear, jac_angular = p.calculateJacobian(
            self.robot_id,
            self.ee_link_index,
            localPosition=local_position,
            objPositions=state.joint_pos.tolist(),
            objVelocities=state.joint_vel.tolist(),
            objAccelerations=[0.0] * len(self.joint_indices),
            physicsClientId=self.physics_client
        )

        # Stack into 6xN matrix
        jacobian = np.vstack([
            np.array(jac_linear),
            np.array(jac_angular)
        ])

        return jacobian

    # ========================================================================
    # Action Writing (following Isaac Lab's write pattern)
    # ========================================================================

    def set_joint_targets(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None
    ):
        """
        Set target joint positions (with PD control)

        Args:
            positions: Target joint positions [num_joints]
            velocities: Target joint velocities (optional)
            forces: Maximum forces per joint (optional, uses config defaults)
        """
        if velocities is None:
            velocities = np.zeros(len(self.joint_indices))

        if forces is None:
            forces = np.array([actuator.max_force for actuator in self.joint_actuators])

        # Clip to joint limits
        positions = np.clip(positions, self.joint_limits_lower, self.joint_limits_upper)

        for i, joint_idx in enumerate(self.joint_indices):
            actuator = self.joint_actuators[i]

            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=positions[i],
                targetVelocity=velocities[i],
                force=forces[i],
                positionGain=actuator.kp,
                velocityGain=actuator.kd,
                physicsClientId=self.physics_client
            )

    def set_joint_velocities(self, velocities: np.ndarray, forces: Optional[np.ndarray] = None):
        """
        Set target joint velocities

        Args:
            velocities: Target joint velocities [num_joints]
            forces: Maximum forces per joint (optional)
        """
        if forces is None:
            forces = np.array([actuator.max_force for actuator in self.joint_actuators])

        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=velocities[i],
                force=forces[i],
                physicsClientId=self.physics_client
            )

    def set_joint_torques(self, torques: np.ndarray):
        """
        Set joint torques directly (torque control mode)

        Args:
            torques: Joint torques [num_joints]
        """
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                controlMode=p.TORQUE_CONTROL,
                force=torques[i],
                physicsClientId=self.physics_client
            )

    # ========================================================================
    # Simulation Control
    # ========================================================================

    def step_simulation(self, num_steps: int = 1):
        """
        Advance physics simulation

        Args:
            num_steps: Number of simulation steps (default: 1)
        """
        for _ in range(num_steps):
            p.stepSimulation(physicsClientId=self.physics_client)

    def reset_simulation(self):
        """Reset simulation and robot to initial state"""
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)

        # Reload robot
        self.robot_id = p.loadURDF(
            self.config.urdf_path,
            basePosition=self.config.base_position,
            baseOrientation=p.getQuaternionFromEuler(self.config.base_orientation),
            useFixedBase=self.config.fixed_base,
            physicsClientId=self.physics_client
        )

        self.reset_to_initial_pose()

    def disconnect(self):
        """Disconnect from PyBullet"""
        p.disconnect(physicsClientId=self.physics_client)

    # ========================================================================
    # Utility Functions
    # ========================================================================

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joint position limits

        Returns:
            (lower_limits, upper_limits) as numpy arrays
        """
        return self.joint_limits_lower.copy(), self.joint_limits_upper.copy()

    def print_state(self):
        """Print current robot state (for debugging)"""
        state = self.get_state()
        print("Robot State:")
        print(f"  End-Effector Position: {state.ee_pos}")
        print(f"  End-Effector Orientation: {state.ee_quat}")
        print(f"  Joint Positions:")
        for name, pos in zip(state.joint_names, state.joint_pos):
            print(f"    {name}: {pos:.3f} rad ({np.degrees(pos):.1f}°)")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from backend.assets.robot_configs import FRANKA_PANDA_CONFIG

    # Create interface with GUI
    robot = PyBulletRobotInterface(FRANKA_PANDA_CONFIG, use_gui=True)

    # Get current state
    state = robot.get_state()
    print(f"\nInitial EE Position: {state.ee_pos}")
    print(f"Initial EE Orientation: {state.ee_quat}")

    # Get Jacobian
    jacobian = robot.get_jacobian()
    print(f"\nJacobian shape: {jacobian.shape}")

    # Run simulation for 5 seconds
    for _ in range(int(5 * 240)):  # 5 seconds at 240 Hz
        robot.step_simulation()

    # Print final state
    robot.print_state()

    input("Press Enter to exit...")
    robot.disconnect()
