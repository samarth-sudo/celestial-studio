"""
Walking Task

Locomotion task for legged robots (quadrupeds and humanoids).
Robot must walk forward maintaining balance and gait rhythm.

Task:
    - Walk forward at target velocity
    - Maintain upright posture
    - Use appropriate gait pattern
    - Track distance traveled

Observations:
    - Body orientation and position
    - Joint positions and velocities
    - Foot contact states
    - Target velocity

Rewards:
    - Forward progress
    - Balance (upright penalty)
    - Gait regularity
    - Energy efficiency

Usage:
    from backend.tasks.walking_task import WalkingTask
    from backend.controllers import GaitType

    task = WalkingTask(robot, gait_type=GaitType.TROT, target_velocity=0.5)
    obs = task.reset()

    while not done:
        # Use quadruped controller
        obs, reward, done, info = task.step(action)

References:
    - Quadruped gait patterns
    - Balance metrics for legged locomotion
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from backend.simulation.pybullet_interface import PyBulletRobotInterface
from backend.tasks.base_task import BaseTask, TaskConfig


@dataclass
class WalkingTaskConfig(TaskConfig):
    """
    Configuration for walking task

    Attributes:
        target_distance: Target distance to walk (meters)
        target_velocity: Target forward velocity (m/s)
        max_tilt_angle: Maximum body tilt before failure (radians)
        min_height: Minimum body height before failure (meters)
        max_height: Maximum body height (meters)
        energy_penalty_scale: Scaling for energy consumption penalty
        balance_reward_scale: Scaling for balance reward
    """
    target_distance: float = 10.0  # 10 meters
    target_velocity: float = 0.5   # 0.5 m/s
    max_tilt_angle: float = 0.5    # ~29 degrees
    min_height: float = 0.15       # 15cm (fallen)
    max_height: float = 0.8        # 80cm
    energy_penalty_scale: float = 0.001
    balance_reward_scale: float = 1.0


class WalkingTask(BaseTask):
    """
    Forward walking task for legged robots

    The robot must walk forward maintaining balance and using
    appropriate gait patterns (trot, walk, etc.).

    Features:
        - Forward progress tracking
        - Balance monitoring
        - Gait rhythm tracking
        - Energy efficiency rewards
    """

    def __init__(
        self,
        robot: PyBulletRobotInterface,
        config: Optional[WalkingTaskConfig] = None
    ):
        """
        Initialize walking task

        Args:
            robot: PyBulletRobotInterface
            config: WalkingTaskConfig
        """
        super().__init__(robot, config or WalkingTaskConfig())
        self.config: WalkingTaskConfig  # Type hint

        # Progress tracking
        self.initial_position: Optional[np.ndarray] = None
        self.distance_traveled = 0.0
        self.max_distance = 0.0

    # ========================================================================
    # Task Interface Implementation
    # ========================================================================

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get task observations

        Returns:
            Dictionary with:
                - "base_position": Body position [x, y, z]
                - "base_orientation": Body orientation [w, x, y, z]
                - "base_linear_vel": Body linear velocity [vx, vy, vz]
                - "base_angular_vel": Body angular velocity [wx, wy, wz]
                - "joint_positions": Joint angles [num_joints]
                - "joint_velocities": Joint velocities [num_joints]
                - "roll": Body roll angle (radians)
                - "pitch": Body pitch angle (radians)
                - "yaw": Body yaw angle (radians)
                - "height": Body height above ground (meters)
                - "forward_velocity": Forward velocity (m/s)
                - "distance_traveled": Total distance traveled (meters)
        """
        state = self.robot.get_state()

        # Extract body orientation
        from backend.utils.math_utils import quat_to_euler
        roll, pitch, yaw = quat_to_euler(state.base_quat)

        # Estimate forward velocity (project velocity onto forward direction)
        forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        forward_vel = np.dot(state.base_pos[:3], forward_dir) if hasattr(state, 'base_vel') else 0.0

        obs = {
            "base_position": state.base_pos,
            "base_orientation": state.base_quat,
            "base_linear_vel": np.zeros(3),  # Simplified - would need to track
            "base_angular_vel": np.zeros(3),  # Simplified
            "joint_positions": state.joint_pos,
            "joint_velocities": state.joint_vel,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "height": state.base_pos[2],
            "forward_velocity": forward_vel,
            "distance_traveled": self.distance_traveled
        }

        return obs

    def compute_rewards(self) -> float:
        """
        Compute reward for current state

        Reward components:
            1. Forward progress (primary)
            2. Balance (upright orientation)
            3. Height maintenance
            4. Energy efficiency (low joint velocities)
            5. Velocity tracking

        Returns:
            Scalar reward
        """
        obs = self.get_observations()

        reward = 0.0

        # 1. Forward progress reward
        if self.initial_position is not None:
            # Distance traveled in forward direction
            forward_dist = obs["base_position"][0] - self.initial_position[0]
            reward += forward_dist * 2.0  # Scale progress reward

        # 2. Balance reward (penalize tilt)
        tilt_magnitude = np.sqrt(obs["roll"]**2 + obs["pitch"]**2)
        balance_reward = np.exp(-tilt_magnitude / 0.2)  # Gaussian around upright
        reward += balance_reward * self.config.balance_reward_scale

        # 3. Height reward (stay at target height)
        target_height = (self.config.min_height + self.config.max_height) / 2
        height_error = abs(obs["height"] - target_height)
        height_reward = np.exp(-height_error / 0.1)
        reward += height_reward * 0.5

        # 4. Velocity tracking reward
        vel_error = abs(obs["forward_velocity"] - self.config.target_velocity)
        vel_reward = np.exp(-vel_error / 0.2)
        reward += vel_reward * 0.5

        # 5. Energy penalty (penalize large joint velocities)
        energy = np.sum(np.abs(obs["joint_velocities"]))
        reward -= energy * self.config.energy_penalty_scale

        return reward

    def is_done(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check termination conditions

        Terminates on:
            - Success: Reached target distance
            - Failure: Body too tilted (fallen)
            - Failure: Body too low (collapsed)
            - Timeout: Exceeded max steps

        Returns:
            (done, info) with termination reason
        """
        obs = self.get_observations()

        # Check success (reached target distance)
        if self.distance_traveled >= self.config.target_distance:
            self.is_success = True
            if not hasattr(self, '_success_counted'):
                self.success_count += 1
                self._success_counted = True
            return True, {
                "success": True,
                "timeout": False,
                "fallen": False,
                "distance_traveled": self.distance_traveled
            }

        # Check failure (fallen)
        tilt_magnitude = np.sqrt(obs["roll"]**2 + obs["pitch"]**2)
        if tilt_magnitude > self.config.max_tilt_angle:
            return True, {
                "success": False,
                "timeout": False,
                "fallen": True,
                "reason": "excessive_tilt",
                "distance_traveled": self.distance_traveled
            }

        # Check failure (collapsed)
        if obs["height"] < self.config.min_height:
            return True, {
                "success": False,
                "timeout": False,
                "fallen": True,
                "reason": "too_low",
                "distance_traveled": self.distance_traveled
            }

        # Check timeout
        if self.check_timeout():
            self.is_timeout = True
            return True, {
                "success": False,
                "timeout": True,
                "fallen": False,
                "distance_traveled": self.distance_traveled
            }

        return False, {
            "success": False,
            "timeout": False,
            "fallen": False,
            "distance_traveled": self.distance_traveled
        }

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset task to initial state

        - Reset robot to initial pose
        - Reset distance counter

        Returns:
            Initial observations
        """
        # Reset robot
        self.robot.reset_to_initial_pose()

        # Reset tracking
        state = self.robot.get_state()
        self.initial_position = state.base_pos.copy()
        self.distance_traveled = 0.0
        self.max_distance = 0.0

        # Reset counters
        self.step_count = 0
        self.episode_count += 1
        self.is_success = False
        self.is_timeout = False
        self._success_counted = False

        return self.get_observations()

    # ========================================================================
    # Progress Tracking
    # ========================================================================

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Extended step with distance tracking

        Args:
            action: Optional action (joint targets)

        Returns:
            (obs, reward, done, info)
        """
        # Track position before step
        if self.initial_position is not None:
            prev_pos = self.robot.get_state().base_pos.copy()

        # Execute base step
        obs, reward, done, info = super().step(action)

        # Update distance traveled
        if self.initial_position is not None:
            current_pos = self.robot.get_state().base_pos
            # Forward distance (x-axis)
            forward_dist = current_pos[0] - self.initial_position[0]
            self.distance_traveled = max(0, forward_dist)
            self.max_distance = max(self.max_distance, self.distance_traveled)

        # Add distance to info
        info["distance_traveled"] = self.distance_traveled
        info["max_distance"] = self.max_distance

        return obs, reward, done, info

    # ========================================================================
    # Utilities
    # ========================================================================

    def get_progress(self) -> float:
        """
        Get task progress (0 to 1)

        Returns:
            Progress ratio
        """
        return min(1.0, self.distance_traveled / self.config.target_distance)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from backend.assets.robot_configs import ANYMAL_C_CONFIG
    from backend.simulation.pybullet_interface import PyBulletRobotInterface
    from backend.controllers import QuadrupedController, GaitType
    import time

    print("Walking Task Example")
    print("=" * 50)

    # Create robot
    robot = PyBulletRobotInterface(ANYMAL_C_CONFIG, use_gui=True)

    # Create task
    config = WalkingTaskConfig(
        target_distance=5.0,  # 5 meters
        target_velocity=0.5,
        timeout_steps=3000
    )
    task = WalkingTask(robot, config=config)

    # Create controller
    controller = QuadrupedController()

    # Reset task
    obs = task.reset()
    print(f"Target distance: {config.target_distance}m")
    print(f"Target velocity: {config.target_velocity}m/s")

    done = False
    total_reward = 0.0
    time_elapsed = 0.0

    print(f"\nStarting walk...")

    while not done:
        # Generate gait
        foot_positions = controller.compute_gait(
            gait_type=GaitType.TROT,
            forward_velocity=config.target_velocity,
            time=time_elapsed
        )

        # Convert to joint angles
        joint_angles = controller.compute_joint_angles(foot_positions)

        # Flatten joint angles (assume leg order: LF, RF, LH, RH)
        all_angles = np.concatenate([
            joint_angles["LF"], joint_angles["RF"],
            joint_angles["LH"], joint_angles["RH"]
        ])

        # Step task
        obs, reward, done, info = task.step(all_angles)
        total_reward += reward
        time_elapsed += 1/240.0

        # Print progress every 200 steps
        if task.step_count % 200 == 0:
            dist = info["distance_traveled"]
            height = obs["height"]
            tilt = np.degrees(np.sqrt(obs["roll"]**2 + obs["pitch"]**2))
            print(f"  Step {task.step_count:4d} | "
                  f"Dist: {dist:5.2f}m | "
                  f"Height: {height:.2f}m | "
                  f"Tilt: {tilt:5.1f}° | "
                  f"Progress: {task.get_progress()*100:5.1f}%")

    # Episode results
    print(f"\nEpisode Results:")
    print(f"  Success: {info['success']}")
    print(f"  Distance traveled: {info['distance_traveled']:.2f}m")
    if 'fallen' in info and info['fallen']:
        print(f"  Fallen: Yes (reason: {info.get('reason', 'unknown')})")
    print(f"  Total steps: {task.step_count}")
    print(f"  Total reward: {total_reward:.2f}")

    input("\nPress Enter to exit...")
    robot.disconnect()
