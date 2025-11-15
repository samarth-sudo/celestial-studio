"""
Lift Task

Pick-and-place task: Grasp an object and lift it to a target height.
Demonstrates multi-stage manipulation with grasping.

Task Stages:
    1. Reach: Move end-effector above object
    2. Descend: Lower to grasp height
    3. Grasp: Close gripper
    4. Lift: Raise object to target height

Success Criteria:
    - Object grasped (gripper closed around object)
    - Object lifted above threshold height
    - Object stable (low velocity)

Observations:
    - Robot state (joint positions, EE pose)
    - Object pose and velocity
    - Gripper state
    - Task stage

Rewards:
    - Reaching object
    - Grasping object
    - Lifting object
    - Maintaining stability

Usage:
    from backend.tasks.lift_task import LiftTask

    task = LiftTask(robot, object_id, target_height=0.3)
    obs = task.reset()

    while not done:
        # Multi-stage control logic
        if task.current_stage == "reach":
            target = task.get_reach_target()
        elif task.current_stage == "grasp":
            task.close_gripper()
        # ...
        obs, reward, done, info = task.step(action)

References:
    - Isaac Lab: isaaclab_tasks/isaaclab_tasks/direct/*/lift_env.py
"""

import numpy as np
import pybullet as p
from typing import Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass

from backend.simulation.pybullet_interface import PyBulletRobotInterface
from backend.tasks.base_task import BaseTask, TaskConfig
from backend.tasks.rewards import (
    compute_reaching_reward,
    compute_grasp_reward,
    compute_lift_reward,
    compute_velocity_penalty
)


@dataclass
class LiftTaskConfig(TaskConfig):
    """
    Configuration for lift task

    Attributes:
        object_initial_pos: Initial object position [x, y, z]
        target_lift_height: Target height to lift object (meters)
        grasp_threshold: Distance for successful grasp (meters)
        lift_threshold: Height threshold for success (meters)
        randomize_object_pos: Randomize object position on reset
        object_pos_range_x: X position range for randomization
        object_pos_range_y: Y position range for randomization
        approach_height_offset: Height above object for approach (meters)
        stability_threshold: Velocity threshold for stability (m/s)
    """
    object_initial_pos: Tuple[float, float, float] = (0.5, 0.0, 0.05)
    target_lift_height: float = 0.3
    grasp_threshold: float = 0.02  # 2cm
    lift_threshold: float = 0.01   # 1cm
    randomize_object_pos: bool = True
    object_pos_range_x: Tuple[float, float] = (0.3, 0.6)
    object_pos_range_y: Tuple[float, float] = (-0.2, 0.2)
    approach_height_offset: float = 0.1  # 10cm above object
    stability_threshold: float = 0.01    # 1cm/s


TaskStage = Literal["reach", "descend", "grasp", "lift", "hold"]


class LiftTask(BaseTask):
    """
    Object lifting task with multi-stage control

    The robot must:
        1. Reach above the object
        2. Descend to grasp height
        3. Close gripper around object
        4. Lift object to target height
        5. Hold stably

    Features:
        - Automatic stage progression
        - Dense reward shaping per stage
        - Object randomization
        - Gripper control
        - Stability checks
    """

    def __init__(
        self,
        robot: PyBulletRobotInterface,
        object_id: Optional[int] = None,
        config: Optional[LiftTaskConfig] = None
    ):
        """
        Initialize lift task

        Args:
            robot: PyBulletRobotInterface
            object_id: PyBullet object ID (creates cube if None)
            config: LiftTaskConfig
        """
        super().__init__(robot, config or LiftTaskConfig())
        self.config: LiftTaskConfig  # Type hint

        # Object tracking
        self.object_id = object_id
        self.object_initial_pos = np.array(self.config.object_initial_pos)
        self.object_size = 0.04  # 4cm cube

        # Create object if not provided
        if self.object_id is None:
            self._create_object()

        # Task state
        self.current_stage: TaskStage = "reach"
        self.gripper_closed = False
        self.object_grasped = False
        self.object_initial_height = self.object_initial_pos[2]

        # Stage waypoints
        self.approach_target: Optional[np.ndarray] = None
        self.grasp_target: Optional[np.ndarray] = None

    # ========================================================================
    # Object Management
    # ========================================================================

    def _create_object(self):
        """Create a simple cube object for grasping"""
        # Create collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.object_size/2] * 3,
            physicsClientId=self.robot.physics_client
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.object_size/2] * 3,
            rgbaColor=[0.8, 0.2, 0.2, 1.0],  # Red cube
            physicsClientId=self.robot.physics_client
        )

        # Create multibody
        self.object_id = p.createMultiBody(
            baseMass=0.1,  # 100g
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.object_initial_pos,
            physicsClientId=self.robot.physics_client
        )

    def get_object_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get object pose and velocity

        Returns:
            (pos, quat, lin_vel, ang_vel)
        """
        pos, quat_xyzw = p.getBasePositionAndOrientation(
            self.object_id,
            physicsClientId=self.robot.physics_client
        )
        lin_vel, ang_vel = p.getBaseVelocity(
            self.object_id,
            physicsClientId=self.robot.physics_client
        )

        # Convert quaternion to w,x,y,z
        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return np.array(pos), quat, np.array(lin_vel), np.array(ang_vel)

    def reset_object_pose(self, pos: Optional[np.ndarray] = None):
        """Reset object to initial position"""
        if pos is None:
            pos = self.object_initial_pos

        p.resetBasePositionAndOrientation(
            self.object_id,
            pos,
            [0, 0, 0, 1],  # Identity quaternion (x,y,z,w format)
            physicsClientId=self.robot.physics_client
        )

        # Reset velocity
        p.resetBaseVelocity(
            self.object_id,
            [0, 0, 0],  # Linear velocity
            [0, 0, 0],  # Angular velocity
            physicsClientId=self.robot.physics_client
        )

    # ========================================================================
    # Task Interface Implementation
    # ========================================================================

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get task observations

        Returns:
            Dictionary with robot state, object state, task stage
        """
        robot_state = self.robot.get_state()
        object_pos, object_quat, object_lin_vel, object_ang_vel = self.get_object_state()

        obs = {
            # Robot state
            "ee_pos": robot_state.ee_pos,
            "ee_quat": robot_state.ee_quat,
            "joint_pos": robot_state.joint_pos,
            "joint_vel": robot_state.joint_vel,

            # Object state
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_vel": object_lin_vel,

            # Relative quantities
            "ee_to_object": object_pos - robot_state.ee_pos,
            "object_height": object_pos[2] - self.object_initial_height,

            # Task state
            "gripper_closed": float(self.gripper_closed),
            "object_grasped": float(self.object_grasped),
            "stage": self._stage_to_int(self.current_stage)
        }

        return obs

    def compute_rewards(self) -> float:
        """
        Compute stage-specific rewards

        Reward structure:
            - Reach: Distance to approach target
            - Descend: Distance to grasp target
            - Grasp: Gripper closing + contact
            - Lift: Object height + stability
            - Hold: Maintain position
        """
        robot_state = self.robot.get_state()
        object_pos, _, object_vel, _ = self.get_object_state()

        reward = 0.0

        if self.current_stage == "reach":
            # Reward approaching object from above
            if self.approach_target is not None:
                reward = compute_reaching_reward(
                    robot_state.ee_pos,
                    self.approach_target,
                    threshold=0.03,
                    scale=2.0
                )

        elif self.current_stage == "descend":
            # Reward descending to grasp height
            if self.grasp_target is not None:
                reward = compute_reaching_reward(
                    robot_state.ee_pos,
                    self.grasp_target,
                    threshold=self.config.grasp_threshold,
                    scale=3.0
                )

        elif self.current_stage == "grasp":
            # Reward grasping object
            reward = compute_grasp_reward(
                robot_state.ee_pos,
                object_pos,
                gripper_state=0.0 if self.gripper_closed else 1.0,
                grasp_threshold=self.config.grasp_threshold,
                object_lifted=False
            )

        elif self.current_stage in ["lift", "hold"]:
            # Reward lifting object
            reward = compute_lift_reward(
                object_pos,
                self.object_initial_height,
                self.config.target_lift_height,
                threshold=self.config.lift_threshold
            )

            # Penalty for excessive velocity (stability)
            reward += compute_velocity_penalty(object_vel, scale=0.1)

        return reward

    def is_done(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check termination conditions

        Success: Object lifted to target height and stable
        Failure: Timeout or object dropped
        """
        object_pos, _, object_vel, _ = self.get_object_state()

        # Check success (lifted and stable)
        height_error = abs(object_pos[2] - self.config.target_lift_height)
        velocity_mag = np.linalg.norm(object_vel)

        if (self.current_stage == "hold" and
            height_error < self.config.lift_threshold and
            velocity_mag < self.config.stability_threshold):
            self.is_success = True
            if not hasattr(self, '_success_counted'):
                self.success_count += 1
                self._success_counted = True
            return True, {
                "success": True,
                "timeout": False,
                "stage": self.current_stage,
                "height_error": height_error
            }

        # Check failure (object dropped)
        if self.object_grasped and object_pos[2] < self.object_initial_height + 0.02:
            return True, {
                "success": False,
                "timeout": False,
                "dropped": True,
                "stage": self.current_stage
            }

        # Check timeout
        if self.check_timeout():
            self.is_timeout = True
            return True, {
                "success": False,
                "timeout": True,
                "stage": self.current_stage,
                "height_error": height_error
            }

        return False, {
            "success": False,
            "timeout": False,
            "stage": self.current_stage,
            "height_error": height_error
        }

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset task to initial state

        - Reset robot and object
        - Randomize object position (if enabled)
        - Reset task stage to "reach"
        """
        # Reset robot
        self.robot.reset_to_initial_pose()

        # Randomize or reset object
        if self.config.randomize_object_pos:
            x = np.random.uniform(*self.config.object_pos_range_x)
            y = np.random.uniform(*self.config.object_pos_range_y)
            z = self.object_size / 2 + 0.001  # Just above ground
            object_pos = np.array([x, y, z])
            self.object_initial_height = z
        else:
            object_pos = self.object_initial_pos

        self.reset_object_pose(object_pos)

        # Compute waypoints
        self._compute_waypoints(object_pos)

        # Reset task state
        self.current_stage = "reach"
        self.gripper_closed = False
        self.object_grasped = False

        # Reset counters
        self.step_count = 0
        self.episode_count += 1
        self.is_success = False
        self.is_timeout = False
        self._success_counted = False

        return self.get_observations()

    # ========================================================================
    # Stage Management
    # ========================================================================

    def _compute_waypoints(self, object_pos: np.ndarray):
        """Compute approach and grasp waypoints"""
        # Approach target: above object
        self.approach_target = object_pos + np.array([0, 0, self.config.approach_height_offset])

        # Grasp target: at object height
        self.grasp_target = object_pos.copy()
        self.grasp_target[2] += self.object_size / 2  # Center of object

    def update_stage(self):
        """
        Update task stage based on current state

        Stage transitions:
            reach -> descend (when above object)
            descend -> grasp (when at grasp height)
            grasp -> lift (when gripper closed)
            lift -> hold (when at target height)
        """
        robot_state = self.robot.get_state()
        object_pos, _, _, _ = self.get_object_state()

        if self.current_stage == "reach":
            # Transition when above object
            if self.approach_target is not None:
                dist = np.linalg.norm(robot_state.ee_pos - self.approach_target)
                if dist < 0.03:  # 3cm threshold
                    self.current_stage = "descend"

        elif self.current_stage == "descend":
            # Transition when at grasp height
            if self.grasp_target is not None:
                dist = np.linalg.norm(robot_state.ee_pos - self.grasp_target)
                if dist < self.config.grasp_threshold:
                    self.current_stage = "grasp"
                    self.close_gripper()

        elif self.current_stage == "grasp":
            # Transition when gripper closed (after a few steps)
            if self.gripper_closed and self.step_count % 50 == 0:
                # Check if object is grasped (simple heuristic: close to EE)
                dist = np.linalg.norm(robot_state.ee_pos - object_pos)
                if dist < self.config.grasp_threshold * 1.5:
                    self.object_grasped = True
                    self.current_stage = "lift"

        elif self.current_stage == "lift":
            # Transition when at target height
            height_error = abs(object_pos[2] - self.config.target_lift_height)
            if height_error < self.config.lift_threshold * 2:
                self.current_stage = "hold"

    def _stage_to_int(self, stage: TaskStage) -> int:
        """Convert stage name to integer for observation"""
        stages = {"reach": 0, "descend": 1, "grasp": 2, "lift": 3, "hold": 4}
        return stages.get(stage, 0)

    # ========================================================================
    # Gripper Control
    # ========================================================================

    def close_gripper(self):
        """Close gripper (simplified - sets flag)"""
        self.gripper_closed = True
        # In real implementation, would command gripper joints
        # For now, just set flag for reward computation

    def open_gripper(self):
        """Open gripper"""
        self.gripper_closed = False

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def get_current_target(self) -> Optional[np.ndarray]:
        """
        Get current target position based on stage

        Returns:
            Target position for current stage
        """
        if self.current_stage == "reach":
            return self.approach_target
        elif self.current_stage == "descend":
            return self.grasp_target
        elif self.current_stage in ["lift", "hold"]:
            # Lift target: above current object position
            object_pos, _, _, _ = self.get_object_state()
            target = object_pos.copy()
            target[2] = self.config.target_lift_height
            return target
        return None

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Extended step with automatic stage updates

        Args:
            action: Optional joint command

        Returns:
            (obs, reward, done, info)
        """
        # Update stage before step
        self.update_stage()

        # Execute base step
        obs, reward, done, info = super().step(action)

        # Add stage info
        info["stage"] = self.current_stage
        info["object_grasped"] = self.object_grasped

        return obs, reward, done, info


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from backend.assets.robot_configs import FRANKA_PANDA_CONFIG
    from backend.simulation.pybullet_interface import PyBulletRobotInterface
    from backend.controllers.differential_ik import DifferentialIKController, DifferentialIKConfig
    import time

    print("Lift Task Example")
    print("=" * 50)

    # Create robot
    robot = PyBulletRobotInterface(FRANKA_PANDA_CONFIG, use_gui=True)

    # Create task
    config = LiftTaskConfig(
        randomize_object_pos=True,
        target_lift_height=0.3,
        timeout_steps=1000
    )
    task = LiftTask(robot, config=config)

    # Create IK controller
    ik_config = DifferentialIKConfig(
        command_type="position",
        position_gain=2.0,
        damping_lambda=0.05
    )
    controller = DifferentialIKController(ik_config)

    # Run episode
    obs = task.reset()
    print(f"Initial Object Position: {obs['object_pos']}")
    print(f"Target Lift Height: {config.target_lift_height}m")

    done = False
    total_reward = 0.0

    while not done:
        # Get target based on current stage
        target_pos = task.get_current_target()

        if target_pos is not None:
            # Compute IK
            joint_targets = controller.compute_joint_targets(
                robot,
                {"pos": target_pos}
            )

            # Step task
            obs, reward, done, info = task.step(joint_targets)
        else:
            # No target, just step
            obs, reward, done, info = task.step()

        total_reward += reward

        # Print progress every 100 steps
        if task.step_count % 100 == 0:
            print(f"Step {task.step_count:3d} | "
                  f"Stage: {info['stage']:8s} | "
                  f"Obj Height: {obs['object_height']*100:5.1f}cm | "
                  f"Reward: {reward:+.2f}")

    # Episode results
    print(f"\nEpisode Results:")
    print(f"  Success: {info['success']}")
    print(f"  Final Stage: {info['stage']}")
    print(f"  Steps: {task.step_count}")
    print(f"  Total Reward: {total_reward:.2f}")

    input("\nPress Enter to exit...")
    robot.disconnect()
