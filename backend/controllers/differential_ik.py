"""
Differential Inverse Kinematics Controller

Clean implementation of differential IK using damped least squares, inspired by Isaac Lab.
Converts end-effector pose commands into joint velocities/positions.

Algorithm: Damped Least Squares (DLS) / Levenberg-Marquardt
    Δq = J^T(JJ^T + λ²I)^(-1) * Δx

Where:
    - J: Jacobian matrix [6, num_joints]
    - Δx: Pose error [dx, dy, dz, ax, ay, az]
    - λ: Damping factor (prevents singularities)
    - Δq: Joint velocity command

Usage:
    from backend.controllers.differential_ik import DifferentialIKController

    controller = DifferentialIKController()
    target_pose = {"pos": [0.5, 0.0, 0.3], "quat": [1, 0, 0, 0]}
    joint_velocities = controller.compute(robot, target_pose)
    robot.set_joint_velocities(joint_velocities)

References:
    - Isaac Lab: isaaclab/controllers/differential_ik.py
    - Buss & Kim (2004): "Selectively Damped Least Squares for IK"
"""

import numpy as np
from typing import Dict, Optional, Tuple, Literal
from dataclasses import dataclass

from backend.simulation.pybullet_interface import PyBulletRobotInterface, RobotState
from backend.utils.math_utils import compute_pose_error, normalize_quat


@dataclass
class DifferentialIKConfig:
    """
    Configuration for differential IK controller

    Attributes:
        command_type: "position" (3-DOF) or "pose" (6-DOF) control
        control_mode: "absolute" (world frame) or "relative" (incremental)
        ik_method: "dls" (damped least squares) or "svd" (singular value decomposition)
        position_gain: Proportional gain for position error (m/s per m error)
        rotation_gain: Proportional gain for rotation error (rad/s per rad error)
        damping_lambda: Damping factor for DLS (larger = more stable, less accurate)
        min_singular_value: Minimum singular value threshold for SVD
        max_iterations: Maximum IK iterations per solve
        tolerance: Position/rotation error tolerance for convergence (meters/radians)
    """
    command_type: Literal["position", "pose"] = "pose"
    control_mode: Literal["absolute", "relative"] = "absolute"
    ik_method: Literal["dls", "svd"] = "dls"

    # Gains (convert pose error to velocity command)
    position_gain: float = 1.0      # 1 m/s per 1m error
    rotation_gain: float = 1.0      # 1 rad/s per 1 rad error

    # Numerical stability
    damping_lambda: float = 0.01    # DLS damping factor
    min_singular_value: float = 0.01  # SVD threshold

    # Convergence settings
    max_iterations: int = 1          # Typically 1 for real-time control
    tolerance: float = 0.001         # 1mm position, ~0.06° rotation tolerance


class DifferentialIKController:
    """
    Differential IK controller using damped least squares

    Converts end-effector pose commands (position + orientation) into
    joint velocities or position targets using the robot's Jacobian matrix.

    Features:
        - Position-only (3-DOF) or full pose (6-DOF) control
        - Absolute or relative control modes
        - DLS or SVD methods
        - Singularity-robust with damping
        - Real-time capable (single iteration)
    """

    def __init__(self, config: Optional[DifferentialIKConfig] = None):
        """
        Initialize differential IK controller

        Args:
            config: IK configuration (uses defaults if None)
        """
        self.config = config or DifferentialIKConfig()

        # Internal state for relative control
        self._target_pos: Optional[np.ndarray] = None
        self._target_quat: Optional[np.ndarray] = None

    # ========================================================================
    # Main Interface
    # ========================================================================

    def compute(
        self,
        robot: PyBulletRobotInterface,
        target: Dict[str, np.ndarray],
        dt: float = 1/240.0
    ) -> np.ndarray:
        """
        Compute joint velocities to reach target end-effector pose

        Args:
            robot: PyBulletRobotInterface with get_state() and get_jacobian()
            target: Target pose dictionary with keys:
                - "pos": [x, y, z] position (required)
                - "quat": [w, x, y, z] orientation (required for pose mode)
                - "pos_delta": [dx, dy, dz] relative position (relative mode)
                - "rot_delta": [droll, dpitch, dyaw] relative rotation (relative mode)
            dt: Timestep for velocity scaling (default: 1/240 = PyBullet default)

        Returns:
            Joint velocities [num_joints] or joint position deltas

        Example:
            >>> controller = DifferentialIKController()
            >>> target = {"pos": [0.5, 0.0, 0.3], "quat": [1, 0, 0, 0]}
            >>> joint_vel = controller.compute(robot, target)
            >>> robot.set_joint_velocities(joint_vel)
        """
        # Get current robot state
        state = robot.get_state()

        # Get target pose (handle absolute vs relative)
        target_pos, target_quat = self._get_target_pose(state, target)

        # Compute pose error
        pos_error, rot_error = compute_pose_error(
            state.ee_pos, state.ee_quat,
            target_pos, target_quat,
            rot_error_type="axis_angle"
        )

        # Build 6D error vector
        if self.config.command_type == "position":
            # Position-only mode: only use position error
            pose_error = pos_error * self.config.position_gain
            dof = 3
        else:
            # Full pose mode: position + rotation
            pose_error = np.concatenate([
                pos_error * self.config.position_gain,
                rot_error * self.config.rotation_gain
            ])
            dof = 6

        # Get Jacobian
        jacobian = robot.get_jacobian()

        # Select rows based on command type
        if self.config.command_type == "position":
            jacobian = jacobian[:3, :]  # Only linear velocity rows

        # Solve IK: Δq = f(J, Δx)
        if self.config.ik_method == "dls":
            delta_q = self._solve_dls(jacobian, pose_error)
        else:  # svd
            delta_q = self._solve_svd(jacobian, pose_error)

        # Clip to reasonable joint velocities (prevent large jumps)
        max_joint_vel = 1.0  # rad/s
        delta_q = np.clip(delta_q, -max_joint_vel, max_joint_vel)

        return delta_q

    def compute_joint_targets(
        self,
        robot: PyBulletRobotInterface,
        target: Dict[str, np.ndarray],
        dt: float = 1/240.0
    ) -> np.ndarray:
        """
        Compute joint position targets (instead of velocities)

        Convenience method that integrates velocities into position commands.

        Args:
            robot: PyBulletRobotInterface
            target: Target pose dictionary
            dt: Timestep for integration

        Returns:
            Joint position targets [num_joints]

        Example:
            >>> joint_targets = controller.compute_joint_targets(robot, target)
            >>> robot.set_joint_targets(joint_targets)
        """
        # Get joint velocities from IK
        joint_vel = self.compute(robot, target, dt)

        # Integrate: q_new = q_current + Δq * dt
        state = robot.get_state()
        joint_targets = state.joint_pos + joint_vel * dt

        # Clip to joint limits
        lower, upper = robot.get_joint_limits()
        joint_targets = np.clip(joint_targets, lower, upper)

        return joint_targets

    # ========================================================================
    # IK Solvers
    # ========================================================================

    def _solve_dls(self, jacobian: np.ndarray, pose_error: np.ndarray) -> np.ndarray:
        """
        Damped Least Squares (DLS) IK solver

        Solves: Δq = J^T(JJ^T + λ²I)^(-1) * Δx

        Advantages:
            - Numerically stable near singularities
            - No matrix decomposition needed
            - Fast for real-time control

        Args:
            jacobian: Jacobian matrix [m, n] where m=3 or 6, n=num_joints
            pose_error: Pose error vector [m] (position or pose)

        Returns:
            Joint velocity command [n]
        """
        m = jacobian.shape[0]  # 3 or 6

        # Compute J * J^T (m x m matrix)
        JJT = jacobian @ jacobian.T

        # Add damping: λ²I
        damping_matrix = (self.config.damping_lambda ** 2) * np.eye(m)

        # Invert: (JJ^T + λ²I)^(-1)
        try:
            JJT_damped_inv = np.linalg.inv(JJT + damping_matrix)
        except np.linalg.LinAlgError:
            # Fallback if singular (shouldn't happen with damping)
            JJT_damped_inv = np.linalg.pinv(JJT + damping_matrix)

        # Solve: Δq = J^T * (JJ^T + λ²I)^(-1) * Δx
        delta_q = jacobian.T @ JJT_damped_inv @ pose_error

        return delta_q

    def _solve_svd(self, jacobian: np.ndarray, pose_error: np.ndarray) -> np.ndarray:
        """
        SVD-based IK solver with singular value thresholding

        Decomposes J = UΣV^T, then solves using pseudoinverse with threshold.

        Advantages:
            - Direct singularity detection
            - Can analyze manipulability

        Disadvantages:
            - Slower than DLS
            - Requires SVD computation

        Args:
            jacobian: Jacobian matrix [m, n]
            pose_error: Pose error vector [m]

        Returns:
            Joint velocity command [n]
        """
        # Compute SVD: J = UΣV^T
        U, S, Vt = np.linalg.svd(jacobian, full_matrices=False)

        # Threshold small singular values
        S_inv = np.zeros_like(S)
        for i, s in enumerate(S):
            if s > self.config.min_singular_value:
                S_inv[i] = 1.0 / s
            else:
                S_inv[i] = 0.0  # Ignore near-singular directions

        # Pseudoinverse: J^+ = V * Σ^(-1) * U^T
        J_pinv = Vt.T @ np.diag(S_inv) @ U.T

        # Solve: Δq = J^+ * Δx
        delta_q = J_pinv @ pose_error

        return delta_q

    # ========================================================================
    # Target Pose Handling
    # ========================================================================

    def _get_target_pose(
        self,
        state: RobotState,
        target: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get target pose from command (handle absolute vs relative)

        Args:
            state: Current robot state
            target: Target command dictionary

        Returns:
            (target_pos, target_quat) in world frame
        """
        if self.config.control_mode == "absolute":
            # Absolute mode: use target directly
            target_pos = np.array(target["pos"])

            if "quat" in target:
                target_quat = normalize_quat(np.array(target["quat"]))
            else:
                # No orientation specified, keep current
                target_quat = state.ee_quat

        else:  # relative
            # Relative mode: apply delta to current pose
            if "pos_delta" in target:
                target_pos = state.ee_pos + np.array(target["pos_delta"])
            else:
                target_pos = state.ee_pos

            if "rot_delta" in target:
                from backend.utils.math_utils import euler_to_quat, quat_mul
                delta_euler = np.array(target["rot_delta"])
                delta_quat = euler_to_quat(delta_euler[0], delta_euler[1], delta_euler[2])
                target_quat = quat_mul(state.ee_quat, delta_quat)
            else:
                target_quat = state.ee_quat

        # Store for next iteration (useful for debugging)
        self._target_pos = target_pos
        self._target_quat = target_quat

        return target_pos, target_quat

    def reset(self):
        """Reset internal state (for relative control)"""
        self._target_pos = None
        self._target_quat = None

    # ========================================================================
    # Utilities
    # ========================================================================

    def compute_error(
        self,
        robot: PyBulletRobotInterface,
        target: Dict[str, np.ndarray]
    ) -> Tuple[float, float]:
        """
        Compute current pose error magnitude

        Useful for checking convergence.

        Args:
            robot: PyBulletRobotInterface
            target: Target pose

        Returns:
            (position_error_norm, rotation_error_norm) in meters and radians

        Example:
            >>> pos_err, rot_err = controller.compute_error(robot, target)
            >>> if pos_err < 0.001 and rot_err < 0.01:
            >>>     print("Target reached!")
        """
        state = robot.get_state()
        target_pos, target_quat = self._get_target_pose(state, target)

        pos_error, rot_error = compute_pose_error(
            state.ee_pos, state.ee_quat,
            target_pos, target_quat,
            rot_error_type="axis_angle"
        )

        return np.linalg.norm(pos_error), np.linalg.norm(rot_error)

    def is_converged(
        self,
        robot: PyBulletRobotInterface,
        target: Dict[str, np.ndarray]
    ) -> bool:
        """
        Check if robot has reached target pose within tolerance

        Args:
            robot: PyBulletRobotInterface
            target: Target pose

        Returns:
            True if within tolerance
        """
        pos_err, rot_err = self.compute_error(robot, target)
        return (pos_err < self.config.tolerance and
                rot_err < self.config.tolerance)


# ============================================================================
# Convenience Functions
# ============================================================================

def solve_ik_position(
    robot: PyBulletRobotInterface,
    target_pos: np.ndarray,
    gain: float = 1.0,
    damping: float = 0.01
) -> np.ndarray:
    """
    Quick position-only IK solve (convenience function)

    Args:
        robot: PyBulletRobotInterface
        target_pos: Target position [x, y, z]
        gain: Position gain
        damping: DLS damping factor

    Returns:
        Joint velocity command

    Example:
        >>> joint_vel = solve_ik_position(robot, [0.5, 0.0, 0.3])
        >>> robot.set_joint_velocities(joint_vel)
    """
    config = DifferentialIKConfig(
        command_type="position",
        position_gain=gain,
        damping_lambda=damping
    )
    controller = DifferentialIKController(config)

    return controller.compute(robot, {"pos": target_pos})


def solve_ik_pose(
    robot: PyBulletRobotInterface,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    position_gain: float = 1.0,
    rotation_gain: float = 1.0,
    damping: float = 0.01
) -> np.ndarray:
    """
    Quick full pose IK solve (convenience function)

    Args:
        robot: PyBulletRobotInterface
        target_pos: Target position [x, y, z]
        target_quat: Target orientation [w, x, y, z]
        position_gain: Position gain
        rotation_gain: Rotation gain
        damping: DLS damping factor

    Returns:
        Joint velocity command

    Example:
        >>> joint_vel = solve_ik_pose(
        ...     robot,
        ...     [0.5, 0.0, 0.3],
        ...     [1, 0, 0, 0]
        ... )
        >>> robot.set_joint_velocities(joint_vel)
    """
    config = DifferentialIKConfig(
        command_type="pose",
        position_gain=position_gain,
        rotation_gain=rotation_gain,
        damping_lambda=damping
    )
    controller = DifferentialIKController(config)

    return controller.compute(robot, {
        "pos": target_pos,
        "quat": target_quat
    })


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from backend.assets.robot_configs import FRANKA_PANDA_CONFIG
    from backend.simulation.pybullet_interface import PyBulletRobotInterface
    import time

    print("Differential IK Controller Test")
    print("=" * 50)

    # Create robot with GUI
    robot = PyBulletRobotInterface(FRANKA_PANDA_CONFIG, use_gui=True)

    # Create IK controller
    config = DifferentialIKConfig(
        command_type="pose",
        position_gain=1.0,
        rotation_gain=1.0,
        damping_lambda=0.01
    )
    controller = DifferentialIKController(config)

    # Get initial state
    initial_state = robot.get_state()
    print(f"\nInitial EE Position: {initial_state.ee_pos}")
    print(f"Initial EE Orientation: {initial_state.ee_quat}")

    # Define target pose (10cm forward in X)
    target = {
        "pos": initial_state.ee_pos + np.array([0.1, 0.0, 0.0]),
        "quat": initial_state.ee_quat
    }

    print(f"\nTarget Position: {target['pos']}")
    print("Running IK control for 5 seconds...")

    # Control loop
    dt = 1/240.0
    for i in range(int(5 / dt)):
        # Compute joint velocities
        joint_vel = controller.compute(robot, target, dt)

        # Apply velocities
        robot.set_joint_velocities(joint_vel)

        # Step simulation
        robot.step_simulation()

        # Print progress every second
        if i % 240 == 0:
            pos_err, rot_err = controller.compute_error(robot, target)
            print(f"  t={i*dt:.1f}s | Pos Error: {pos_err*1000:.2f}mm | Rot Error: {np.degrees(rot_err):.2f}°")

            if controller.is_converged(robot, target):
                print("  ✅ Target reached!")
                break

    # Final state
    final_state = robot.get_state()
    print(f"\nFinal EE Position: {final_state.ee_pos}")
    print(f"Final position error: {np.linalg.norm(final_state.ee_pos - target['pos'])*1000:.2f}mm")

    input("\nPress Enter to exit...")
    robot.disconnect()
