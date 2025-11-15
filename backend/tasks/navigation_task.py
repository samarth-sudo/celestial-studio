"""
Navigation Task

Waypoint navigation task for mobile robots (wheeled and legged).
Robot must navigate to a series of waypoints while avoiding obstacles.

Task:
    - Follow waypoints in sequence
    - Reach each waypoint within threshold
    - Maintain safe velocity
    - Track progress through waypoint list

Observations:
    - Current position and orientation
    - Target waypoint
    - Distance to waypoint
    - Velocity

Rewards:
    - Dense: Progress towards waypoint
    - Sparse: Waypoint reached bonus

Usage:
    from backend.tasks.navigation_task import NavigationTask

    waypoints = [
        np.array([1, 0]),
        np.array([1, 1]),
        np.array([0, 1]),
    ]
    task = NavigationTask(robot, waypoints=waypoints)
    obs = task.reset()

    while not done:
        # Use mobile controller to follow waypoints
        obs, reward, done, info = task.step(action)

References:
    - Pure pursuit path following
    - Waypoint navigation
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from backend.simulation.pybullet_interface import PyBulletRobotInterface
from backend.tasks.base_task import BaseTask, TaskConfig
from backend.tasks.rewards import compute_distance_reward


@dataclass
class NavigationTaskConfig(TaskConfig):
    """
    Configuration for navigation task

    Attributes:
        waypoint_threshold: Distance to consider waypoint reached (meters)
        randomize_waypoints: Randomize waypoint locations on reset
        waypoint_area_size: Size of area for random waypoints (meters)
        max_waypoints: Maximum number of waypoints
        require_final_orientation: Must reach final heading at last waypoint
        final_orientation_threshold: Heading error threshold (radians)
    """
    waypoint_threshold: float = 0.2  # 20cm
    randomize_waypoints: bool = False
    waypoint_area_size: float = 5.0  # 5m x 5m area
    max_waypoints: int = 5
    require_final_orientation: bool = False
    final_orientation_threshold: float = 0.1  # ~6 degrees


class NavigationTask(BaseTask):
    """
    Waypoint navigation task for mobile/legged robots

    The robot must navigate through a series of waypoints,
    reaching each one in sequence before moving to the next.

    Features:
        - Sequential waypoint following
        - Progress tracking
        - Waypoint randomization
        - Dense reward shaping
    """

    def __init__(
        self,
        robot: PyBulletRobotInterface,
        waypoints: Optional[List[np.ndarray]] = None,
        config: Optional[NavigationTaskConfig] = None
    ):
        """
        Initialize navigation task

        Args:
            robot: PyBulletRobotInterface
            waypoints: List of waypoint positions [[x, y], [x, y], ...]
            config: NavigationTaskConfig
        """
        super().__init__(robot, config or NavigationTaskConfig())
        self.config: NavigationTaskConfig  # Type hint

        # Waypoints
        self._fixed_waypoints = waypoints
        self.waypoints: List[np.ndarray] = []
        self.current_waypoint_idx = 0

        # Initialize waypoints
        if waypoints is not None:
            self.waypoints = [np.array(wp) for wp in waypoints]

        # Progress tracking
        self.total_waypoints = 0
        self.waypoints_reached = 0
        self.cumulative_distance = 0.0

    # ========================================================================
    # Task Interface Implementation
    # ========================================================================

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get task observations

        Returns:
            Dictionary with:
                - "position": Current 2D position [x, y]
                - "yaw": Current heading (radians)
                - "velocity": Current 2D velocity [vx, vy]
                - "target_waypoint": Current target waypoint [x, y]
                - "distance_to_waypoint": Scalar distance
                - "heading_error": Angle to waypoint (radians)
                - "waypoint_index": Current waypoint index
                - "progress": Fraction of waypoints completed (0-1)
        """
        state = self.robot.get_state()

        # Extract 2D position (x, y from base position)
        position = state.base_pos[:2]

        # Extract yaw from quaternion
        from backend.utils.math_utils import quat_to_euler
        _, _, yaw = quat_to_euler(state.base_quat)

        # Get current target waypoint
        if self.current_waypoint_idx < len(self.waypoints):
            target_wp = self.waypoints[self.current_waypoint_idx]
        else:
            target_wp = self.waypoints[-1] if self.waypoints else np.array([0.0, 0.0])

        # Compute error
        error = target_wp - position
        distance = np.linalg.norm(error)

        # Heading to waypoint
        heading_to_wp = np.arctan2(error[1], error[0])
        heading_error = self._normalize_angle(heading_to_wp - yaw)

        # Velocity (approximate from base velocity if available)
        velocity = np.array([0.0, 0.0])  # Simplified - would need to track or estimate

        obs = {
            "position": position,
            "yaw": yaw,
            "velocity": velocity,
            "target_waypoint": target_wp,
            "distance_to_waypoint": distance,
            "heading_error": heading_error,
            "waypoint_index": self.current_waypoint_idx,
            "progress": self.waypoints_reached / max(1, self.total_waypoints)
        }

        return obs

    def compute_rewards(self) -> float:
        """
        Compute reward for current state

        Reward structure:
            - Dense: Negative distance to current waypoint
            - Bonus: Waypoint reached
            - Penalty: Excessive heading error

        Returns:
            Scalar reward
        """
        obs = self.get_observations()

        # Distance reward (negative distance)
        reward = -obs["distance_to_waypoint"]

        # Heading alignment reward
        heading_penalty = -abs(obs["heading_error"]) * 0.1
        reward += heading_penalty

        # Waypoint reached bonus
        if obs["distance_to_waypoint"] < self.config.waypoint_threshold:
            reward += 10.0

        return reward

    def is_done(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check termination conditions

        Terminates on:
            - Success: All waypoints reached
            - Timeout: Exceeded max steps

        Returns:
            (done, info) with termination reason
        """
        obs = self.get_observations()

        # Check if all waypoints reached
        if self.current_waypoint_idx >= len(self.waypoints):
            self.is_success = True
            if not hasattr(self, '_success_counted'):
                self.success_count += 1
                self._success_counted = True
            return True, {
                "success": True,
                "timeout": False,
                "waypoints_reached": self.waypoints_reached,
                "total_waypoints": self.total_waypoints
            }

        # Check timeout
        if self.check_timeout():
            self.is_timeout = True
            return True, {
                "success": False,
                "timeout": True,
                "waypoints_reached": self.waypoints_reached,
                "total_waypoints": self.total_waypoints
            }

        return False, {
            "success": False,
            "timeout": False,
            "waypoints_reached": self.waypoints_reached,
            "total_waypoints": self.total_waypoints
        }

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset task to initial state

        - Reset robot to start position
        - Randomize waypoints (if enabled)
        - Reset waypoint counter

        Returns:
            Initial observations
        """
        # Reset robot
        self.robot.reset_to_initial_pose()

        # Randomize or set waypoints
        if self.config.randomize_waypoints and self._fixed_waypoints is None:
            self._randomize_waypoints()
        elif self._fixed_waypoints is not None:
            self.waypoints = [np.array(wp) for wp in self._fixed_waypoints]

        # Reset counters
        self.current_waypoint_idx = 0
        self.total_waypoints = len(self.waypoints)
        self.waypoints_reached = 0
        self.cumulative_distance = 0.0

        self.step_count = 0
        self.episode_count += 1
        self.is_success = False
        self.is_timeout = False
        self._success_counted = False

        return self.get_observations()

    # ========================================================================
    # Waypoint Management
    # ========================================================================

    def _randomize_waypoints(self):
        """Generate random waypoints in the navigation area"""
        num_waypoints = np.random.randint(2, self.config.max_waypoints + 1)

        self.waypoints = []
        for _ in range(num_waypoints):
            # Random position in area
            x = np.random.uniform(-self.config.waypoint_area_size / 2,
                                 self.config.waypoint_area_size / 2)
            y = np.random.uniform(-self.config.waypoint_area_size / 2,
                                 self.config.waypoint_area_size / 2)
            self.waypoints.append(np.array([x, y]))

    def check_waypoint_reached(self) -> bool:
        """
        Check if current waypoint has been reached

        Returns:
            True if waypoint reached, False otherwise
        """
        obs = self.get_observations()

        if obs["distance_to_waypoint"] < self.config.waypoint_threshold:
            # Check orientation if required
            if self.config.require_final_orientation and \
               self.current_waypoint_idx == len(self.waypoints) - 1:
                # At final waypoint, check heading too
                if abs(obs["heading_error"]) < self.config.final_orientation_threshold:
                    return True
                return False
            return True

        return False

    def advance_waypoint(self):
        """Move to next waypoint"""
        if self.current_waypoint_idx < len(self.waypoints):
            self.waypoints_reached += 1
            self.current_waypoint_idx += 1

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Extended step with automatic waypoint advancement

        Args:
            action: Optional action (wheel velocities or joint commands)

        Returns:
            (obs, reward, done, info)
        """
        # Check waypoint before step
        if self.check_waypoint_reached():
            self.advance_waypoint()

        # Execute base step
        return super().step(action)

    # ========================================================================
    # Utilities
    # ========================================================================

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def get_remaining_distance(self) -> float:
        """
        Estimate remaining distance to complete task

        Returns:
            Approximate distance (meters)
        """
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0.0

        obs = self.get_observations()
        total_distance = obs["distance_to_waypoint"]

        # Add distances between remaining waypoints
        for i in range(self.current_waypoint_idx, len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            total_distance += np.linalg.norm(wp2 - wp1)

        return total_distance

    def visualize_waypoints(self):
        """
        Print waypoint information (for debugging)
        """
        print(f"\nNavigation Task:")
        print(f"  Total waypoints: {len(self.waypoints)}")
        print(f"  Current waypoint: {self.current_waypoint_idx}")
        print(f"  Reached: {self.waypoints_reached}/{self.total_waypoints}")
        print(f"  Waypoints:")
        for i, wp in enumerate(self.waypoints):
            marker = " → " if i == self.current_waypoint_idx else "   "
            status = "✓" if i < self.current_waypoint_idx else " "
            print(f"    {marker}[{status}] {i}: [{wp[0]:6.2f}, {wp[1]:6.2f}]")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from backend.assets.robot_configs import SIMPLE_MOBILE_ROBOT_CONFIG
    from backend.simulation.pybullet_interface import PyBulletRobotInterface
    from backend.controllers import MobileController
    import time

    print("Navigation Task Example")
    print("=" * 50)

    # Create robot
    robot = PyBulletRobotInterface(SIMPLE_MOBILE_ROBOT_CONFIG, use_gui=True)

    # Create task with waypoints
    waypoints = [
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
    ]

    config = NavigationTaskConfig(
        waypoint_threshold=0.2,
        timeout_steps=2000
    )
    task = NavigationTask(robot, waypoints=waypoints, config=config)

    # Create controller
    controller = MobileController()

    # Reset task
    obs = task.reset()
    task.visualize_waypoints()

    done = False
    total_reward = 0.0

    print(f"\nStarting navigation...")

    while not done:
        # Get current state
        position = obs["position"]
        yaw = obs["yaw"]
        target = obs["target_waypoint"]

        # Compute control (simple goal seeking)
        linear, angular = controller.compute_velocity_to_goal(
            position, yaw, target
        )

        # Apply control
        left, right = controller.compute_wheel_velocities(linear, angular)

        # Step task
        obs, reward, done, info = task.step(np.array([left, right]))
        total_reward += reward

        # Print progress every 100 steps
        if task.step_count % 100 == 0:
            dist = obs["distance_to_waypoint"]
            heading = np.degrees(obs["heading_error"])
            print(f"  Step {task.step_count:4d} | "
                  f"WP {obs['waypoint_index']}/{len(waypoints)} | "
                  f"Dist: {dist:5.2f}m | "
                  f"Heading: {heading:+6.1f}°")

    # Episode results
    print(f"\nEpisode Results:")
    print(f"  Success: {info['success']}")
    print(f"  Waypoints reached: {info['waypoints_reached']}/{info['total_waypoints']}")
    print(f"  Total steps: {task.step_count}")
    print(f"  Total reward: {total_reward:.2f}")

    input("\nPress Enter to exit...")
    robot.disconnect()
