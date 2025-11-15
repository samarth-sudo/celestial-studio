"""
Reach Task

Simple end-effector reaching task: move the end-effector to a target pose.
Great for testing IK controllers and as a foundation for more complex tasks.

Task:
    - Move end-effector to target position (and optionally orientation)
    - Success: Within threshold distance
    - Failure: Timeout

Observations:
    - End-effector position/orientation
    - Target position/orientation
    - Distance to target

Rewards:
    - Dense: Negative distance to target
    - Sparse: Success bonus

Usage:
    from backend.tasks.reach_task import ReachTask

    task = ReachTask(robot, target_pos=[0.5, 0.0, 0.3])
    obs = task.reset()

    while not done:
        action = controller.compute(robot, task.target_pose)
        obs, reward, done, info = task.step(action)

References:
    - Isaac Lab: isaaclab_tasks/isaaclab_tasks/direct/*/reach_env.py
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from backend.simulation.pybullet_interface import PyBulletRobotInterface
from backend.tasks.base_task import BaseTask, TaskConfig
from backend.tasks.rewards import (
    compute_distance_reward,
    compute_orientation_reward,
    compute_velocity_penalty,
    compute_reach_reward
)
from backend.utils.math_utils import euler_to_quat


@dataclass
class ReachTaskConfig(TaskConfig):
    """
    Configuration for reach task

    Attributes:
        control_position_only: If True, only control position (ignore orientation)
        randomize_target: Randomize target on each reset
        target_distance_range: Range for random target distance (meters)
        target_height_range: Range for random target Z position (meters)
        reward_type: "dense" (distance-based) or "sparse" (success only)
        orientation_weight: Weight for orientation reward (0 = position-only)
    """
    control_position_only: bool = False
    randomize_target: bool = True
    target_distance_range: Tuple[float, float] = (0.3, 0.6)  # 30-60cm from base
    target_height_range: Tuple[float, float] = (0.2, 0.5)    # 20-50cm height
    reward_type: str = "dense"  # "dense" or "sparse"
    orientation_weight: float = 0.5


class ReachTask(BaseTask):
    """
    End-effector reaching task

    The robot must move its end-effector to a target pose in 3D space.
    Can be configured for position-only or full pose control.

    Features:
        - Target randomization on reset
        - Dense or sparse rewards
        - Position-only or full pose control
        - Timeout detection
    """

    def __init__(
        self,
        robot: PyBulletRobotInterface,
        target_pos: Optional[np.ndarray] = None,
        target_quat: Optional[np.ndarray] = None,
        config: Optional[ReachTaskConfig] = None
    ):
        """
        Initialize reach task

        Args:
            robot: PyBulletRobotInterface
            target_pos: Fixed target position [x, y, z] (None for random)
            target_quat: Fixed target orientation [w, x, y, z] (None for random)
            config: ReachTaskConfig
        """
        super().__init__(robot, config or ReachTaskConfig())
        self.config: ReachTaskConfig  # Type hint

        # Target pose
        self._fixed_target_pos = target_pos
        self._fixed_target_quat = target_quat
        self.target_pos: np.ndarray = None
        self.target_quat: np.ndarray = None

        # Initialize target
        if target_pos is not None:
            self.target_pos = np.array(target_pos)
        if target_quat is not None:
            self.target_quat = np.array(target_quat)

        # Task tracking
        self.initial_distance = 0.0

    # ========================================================================
    # Task Interface Implementation
    # ========================================================================

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get task observations

        Returns:
            Dictionary with:
                - "ee_pos": End-effector position [3]
                - "ee_quat": End-effector orientation [4]
                - "ee_vel": End-effector linear velocity [3]
                - "target_pos": Target position [3]
                - "target_quat": Target orientation [4]
                - "pos_error": Position error vector [3]
                - "distance": Scalar distance to target
                - "joint_pos": Joint positions [num_joints]
                - "joint_vel": Joint velocities [num_joints]
        """
        state = self.robot.get_state()

        obs = {
            "ee_pos": state.ee_pos,
            "ee_quat": state.ee_quat,
            "ee_vel": state.ee_lin_vel,
            "target_pos": self.target_pos,
            "target_quat": self.target_quat,
            "pos_error": self.target_pos - state.ee_pos,
            "distance": np.linalg.norm(self.target_pos - state.ee_pos),
            "joint_pos": state.joint_pos,
            "joint_vel": state.joint_vel
        }

        return obs

    def compute_rewards(self) -> float:
        """
        Compute reward for current state

        Supports:
            - Dense rewards: Shaped distance + orientation
            - Sparse rewards: Only success bonus

        Returns:
            Scalar reward
        """
        state = self.robot.get_state()

        if self.config.reward_type == "sparse":
            # Sparse: only reward success
            distance = np.linalg.norm(state.ee_pos - self.target_pos)
            if distance < self.config.success_threshold:
                return self.config.reward_success
            else:
                return 0.0

        else:  # dense
            # Dense: shaped distance + orientation
            if self.config.control_position_only or self.config.orientation_weight == 0:
                # Position-only reward
                reward = compute_distance_reward(
                    state.ee_pos,
                    self.target_pos,
                    threshold=self.config.success_threshold,
                    kernel="exponential",
                    sigma=0.1
                )
            else:
                # Full pose reward
                reward = compute_reach_reward(
                    state.ee_pos,
                    state.ee_quat,
                    self.target_pos,
                    self.target_quat,
                    pos_weight=1.0,
                    rot_weight=self.config.orientation_weight,
                    joint_vel=state.joint_vel,
                    vel_penalty_scale=0.001
                )

            return reward

    def is_done(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check termination conditions

        Terminates on:
            - Success: Reached target within threshold
            - Timeout: Exceeded max steps

        Returns:
            (done, info) with termination reason
        """
        state = self.robot.get_state()

        # Compute errors
        pos_distance = np.linalg.norm(state.ee_pos - self.target_pos)

        if not self.config.control_position_only:
            from backend.utils.math_utils import compute_pose_error
            _, rot_error = compute_pose_error(
                np.zeros(3), state.ee_quat,
                np.zeros(3), self.target_quat,
                rot_error_type="axis_angle"
            )
            rot_distance = np.linalg.norm(rot_error)
        else:
            rot_distance = 0.0

        # Check success
        if pos_distance < self.config.success_threshold:
            if self.config.control_position_only or rot_distance < 0.1:  # ~6 degrees
                self.is_success = True
                if not hasattr(self, '_success_counted'):
                    self.success_count += 1
                    self._success_counted = True
                return True, {
                    "success": True,
                    "timeout": False,
                    "pos_error": pos_distance,
                    "rot_error": rot_distance
                }

        # Check timeout
        if self.check_timeout():
            self.is_timeout = True
            return True, {
                "success": False,
                "timeout": True,
                "pos_error": pos_distance,
                "rot_error": rot_distance
            }

        return False, {
            "success": False,
            "timeout": False,
            "pos_error": pos_distance,
            "rot_error": rot_distance
        }

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset task to initial state

        - Resets robot to initial pose
        - Randomizes target (if enabled)
        - Resets counters

        Returns:
            Initial observations
        """
        # Reset robot
        self.robot.reset_to_initial_pose()

        # Randomize or set target
        if self.config.randomize_target and self._fixed_target_pos is None:
            self._randomize_target()
        else:
            # Use fixed target
            if self._fixed_target_pos is not None:
                self.target_pos = np.array(self._fixed_target_pos)
            if self._fixed_target_quat is not None:
                self.target_quat = np.array(self._fixed_target_quat)

        # Ensure target_quat is set
        if self.target_quat is None:
            self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity

        # Compute initial distance
        state = self.robot.get_state()
        self.initial_distance = np.linalg.norm(state.ee_pos - self.target_pos)

        # Reset counters
        self.step_count = 0
        self.episode_count += 1
        self.is_success = False
        self.is_timeout = False
        self._success_counted = False

        return self.get_observations()

    # ========================================================================
    # Target Randomization
    # ========================================================================

    def _randomize_target(self):
        """
        Randomize target pose within workspace

        Generates random position in cylindrical coordinates around robot base.
        """
        # Random distance and angle
        distance = np.random.uniform(*self.config.target_distance_range)
        angle = np.random.uniform(-np.pi, np.pi)
        height = np.random.uniform(*self.config.target_height_range)

        # Convert to Cartesian (relative to base)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        z = height

        self.target_pos = np.array([x, y, z])

        # Random orientation (slight variations from identity)
        if not self.config.control_position_only:
            roll = np.random.uniform(-0.2, 0.2)   # ±11°
            pitch = np.random.uniform(-0.2, 0.2)
            yaw = np.random.uniform(-0.5, 0.5)    # ±29°
            self.target_quat = euler_to_quat(roll, pitch, yaw)
        else:
            self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def set_target(self, pos: np.ndarray, quat: Optional[np.ndarray] = None):
        """
        Manually set target pose (useful for testing)

        Args:
            pos: Target position [x, y, z]
            quat: Target orientation [w, x, y, z] (optional)
        """
        self.target_pos = np.array(pos)
        if quat is not None:
            self.target_quat = np.array(quat)
        else:
            self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # ========================================================================
    # Utilities
    # ========================================================================

    @property
    def target_pose(self) -> Dict[str, np.ndarray]:
        """
        Get target pose as dictionary (for IK controller)

        Returns:
            {"pos": target_pos, "quat": target_quat}
        """
        return {
            "pos": self.target_pos,
            "quat": self.target_quat
        }

    def get_progress(self) -> float:
        """
        Get task progress (0 to 1)

        Returns:
            Progress ratio (1 = at target, 0 = at initial distance)
        """
        state = self.robot.get_state()
        current_distance = np.linalg.norm(state.ee_pos - self.target_pos)

        if self.initial_distance == 0:
            return 1.0

        progress = 1.0 - (current_distance / self.initial_distance)
        return np.clip(progress, 0.0, 1.0)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from backend.assets.robot_configs import FRANKA_PANDA_CONFIG
    from backend.simulation.pybullet_interface import PyBulletRobotInterface
    from backend.controllers.differential_ik import DifferentialIKController, DifferentialIKConfig
    import time

    print("Reach Task Example")
    print("=" * 50)

    # Create robot
    robot = PyBulletRobotInterface(FRANKA_PANDA_CONFIG, use_gui=True)

    # Create task
    config = ReachTaskConfig(
        control_position_only=False,
        randomize_target=True,
        reward_type="dense",
        success_threshold=0.01,  # 1cm
        timeout_steps=500
    )
    task = ReachTask(robot, config=config)

    # Create IK controller
    ik_config = DifferentialIKConfig(
        command_type="pose",
        position_gain=2.0,
        rotation_gain=1.0,
        damping_lambda=0.05
    )
    controller = DifferentialIKController(ik_config)

    # Run multiple episodes
    num_episodes = 3

    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")

        # Reset task
        obs = task.reset()
        print(f"Initial EE Position: {obs['ee_pos']}")
        print(f"Target Position: {obs['target_pos']}")
        print(f"Initial Distance: {obs['distance']*1000:.1f}mm")

        done = False
        total_reward = 0.0

        while not done:
            # Compute IK to reach target
            joint_targets = controller.compute_joint_targets(robot, task.target_pose)

            # Step task
            obs, reward, done, info = task.step(joint_targets)
            total_reward += reward

            # Print progress every 50 steps
            if task.step_count % 50 == 0:
                print(f"  Step {task.step_count:3d}: "
                      f"Distance = {obs['distance']*1000:5.1f}mm, "
                      f"Progress = {task.get_progress()*100:5.1f}%, "
                      f"Reward = {reward:+.3f}")

        # Episode results
        print(f"\nEpisode Results:")
        print(f"  Success: {info['success']}")
        print(f"  Steps: {task.step_count}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Final Distance: {info['pos_error']*1000:.2f}mm")
        if not config.control_position_only:
            print(f"  Final Rotation Error: {np.degrees(info['rot_error']):.2f}°")

        time.sleep(1)  # Pause between episodes

    # Final statistics
    print(f"\n{'='*50}")
    print(f"Overall Success Rate: {task.get_success_rate()*100:.1f}%")
    print(f"Total Episodes: {task.episode_count}")

    input("\nPress Enter to exit...")
    robot.disconnect()
