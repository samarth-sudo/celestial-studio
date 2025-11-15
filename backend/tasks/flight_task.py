"""
Flight Task

Aerial navigation task for quadcopter drones.
Robot must fly through waypoints while maintaining stable flight.

Task:
    - Fly through 3D waypoints in sequence
    - Maintain altitude
    - Avoid excessive tilt/velocity
    - Smooth trajectory

Observations:
    - Position and orientation
    - Velocity (linear and angular)
    - Target waypoint
    - Distance to waypoint

Rewards:
    - Progress towards waypoint
    - Waypoint reached bonus
    - Stability (low tilt, smooth velocity)
    - Energy efficiency

Usage:
    from backend.tasks.flight_task import FlightTask

    waypoints = [
        np.array([1, 0, 2]),
        np.array([1, 1, 2.5]),
        np.array([0, 1, 2]),
    ]
    task = FlightTask(robot, waypoints=waypoints)
    obs = task.reset()

    while not done:
        # Use drone controller
        obs, reward, done, info = task.step(action)

References:
    - Quadcopter trajectory following
    - Waypoint navigation
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from backend.simulation.pybullet_interface import PyBulletRobotInterface
from backend.tasks.base_task import BaseTask, TaskConfig


@dataclass
class FlightTaskConfig(TaskConfig):
    """
    Configuration for flight task

    Attributes:
        waypoint_threshold: Distance to consider waypoint reached (meters)
        altitude_min: Minimum safe altitude (meters)
        altitude_max: Maximum altitude (meters)
        max_tilt_angle: Maximum tilt angle (radians)
        max_velocity: Maximum velocity (m/s)
        randomize_waypoints: Randomize waypoint locations on reset
        flight_area_size: Size of flight area (meters)
        stability_weight: Weight for stability reward
    """
    waypoint_threshold: float = 0.3  # 30cm
    altitude_min: float = 0.5        # 50cm
    altitude_max: float = 5.0        # 5m
    max_tilt_angle: float = 0.7      # ~40 degrees
    max_velocity: float = 5.0        # 5 m/s
    randomize_waypoints: bool = False
    flight_area_size: float = 10.0  # 10m x 10m x 3m
    stability_weight: float = 0.5


class FlightTask(BaseTask):
    """
    Waypoint navigation task for aerial robots (drones)

    The robot must fly through a series of 3D waypoints,
    maintaining stable flight and avoiding crashes.

    Features:
        - 3D waypoint following
        - Altitude limits
        - Tilt/velocity monitoring
        - Smooth trajectory rewards
    """

    def __init__(
        self,
        robot: PyBulletRobotInterface,
        waypoints: Optional[List[np.ndarray]] = None,
        config: Optional[FlightTaskConfig] = None
    ):
        """
        Initialize flight task

        Args:
            robot: PyBulletRobotInterface
            waypoints: List of 3D waypoint positions [[x, y, z], ...]
            config: FlightTaskConfig
        """
        super().__init__(robot, config or FlightTaskConfig())
        self.config: FlightTaskConfig  # Type hint

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

    # ========================================================================
    # Task Interface Implementation
    # ========================================================================

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get task observations

        Returns:
            Dictionary with:
                - "position": Current 3D position [x, y, z]
                - "orientation": Current orientation [w, x, y, z]
                - "linear_velocity": Linear velocity [vx, vy, vz]
                - "angular_velocity": Angular velocity [wx, wy, wz]
                - "target_waypoint": Current target [x, y, z]
                - "distance_to_waypoint": Scalar distance
                - "altitude": Current altitude (z position)
                - "roll": Roll angle (radians)
                - "pitch": Pitch angle (radians)
                - "yaw": Yaw angle (radians)
                - "waypoint_index": Current waypoint index
                - "progress": Fraction of waypoints completed
        """
        state = self.robot.get_state()

        # Extract orientation angles
        from backend.utils.math_utils import quat_to_euler
        roll, pitch, yaw = quat_to_euler(state.base_quat)

        # Get current target waypoint
        if self.current_waypoint_idx < len(self.waypoints):
            target_wp = self.waypoints[self.current_waypoint_idx]
        else:
            target_wp = self.waypoints[-1] if self.waypoints else np.array([0.0, 0.0, 2.0])

        # Compute error
        error = target_wp - state.base_pos
        distance = np.linalg.norm(error)

        obs = {
            "position": state.base_pos,
            "orientation": state.base_quat,
            "linear_velocity": np.zeros(3),  # Simplified - would need velocity tracking
            "angular_velocity": np.zeros(3),
            "target_waypoint": target_wp,
            "distance_to_waypoint": distance,
            "altitude": state.base_pos[2],
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "waypoint_index": self.current_waypoint_idx,
            "progress": self.waypoints_reached / max(1, self.total_waypoints)
        }

        return obs

    def compute_rewards(self) -> float:
        """
        Compute reward for current state

        Reward components:
            1. Progress towards waypoint (negative distance)
            2. Waypoint reached bonus
            3. Stability (low tilt angles)
            4. Altitude maintenance
            5. Smooth flight (low velocities)

        Returns:
            Scalar reward
        """
        obs = self.get_observations()

        reward = 0.0

        # 1. Distance reward (exponential falloff)
        dist_reward = np.exp(-obs["distance_to_waypoint"] / 1.0)
        reward += dist_reward * 2.0

        # 2. Waypoint reached bonus
        if obs["distance_to_waypoint"] < self.config.waypoint_threshold:
            reward += 10.0

        # 3. Stability reward (penalize excessive tilt)
        tilt_magnitude = np.sqrt(obs["roll"]**2 + obs["pitch"]**2)
        stability_reward = np.exp(-tilt_magnitude / 0.3)
        reward += stability_reward * self.config.stability_weight

        # 4. Altitude maintenance (penalize if too low/high)
        if obs["altitude"] < self.config.altitude_min:
            reward -= 2.0 * (self.config.altitude_min - obs["altitude"])
        elif obs["altitude"] > self.config.altitude_max:
            reward -= 1.0 * (obs["altitude"] - self.config.altitude_max)

        # 5. Velocity penalty (prefer smooth flight)
        # velocity_magnitude = np.linalg.norm(obs["linear_velocity"])
        # if velocity_magnitude > 2.0:  # Penalize high speeds
        #     reward -= 0.1 * (velocity_magnitude - 2.0)

        return reward

    def is_done(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check termination conditions

        Terminates on:
            - Success: All waypoints reached
            - Failure: Crashed (too low altitude)
            - Failure: Out of bounds (too high or far)
            - Failure: Excessive tilt
            - Timeout: Exceeded max steps

        Returns:
            (done, info) with termination reason
        """
        obs = self.get_observations()

        # Check success (all waypoints reached)
        if self.current_waypoint_idx >= len(self.waypoints):
            self.is_success = True
            if not hasattr(self, '_success_counted'):
                self.success_count += 1
                self._success_counted = True
            return True, {
                "success": True,
                "timeout": False,
                "crashed": False,
                "waypoints_reached": self.waypoints_reached,
                "total_waypoints": self.total_waypoints
            }

        # Check crash (too low)
        if obs["altitude"] < self.config.altitude_min * 0.5:
            return True, {
                "success": False,
                "timeout": False,
                "crashed": True,
                "reason": "altitude_too_low",
                "waypoints_reached": self.waypoints_reached
            }

        # Check out of bounds (too high)
        if obs["altitude"] > self.config.altitude_max:
            return True, {
                "success": False,
                "timeout": False,
                "crashed": True,
                "reason": "altitude_too_high",
                "waypoints_reached": self.waypoints_reached
            }

        # Check excessive tilt (lost control)
        tilt_magnitude = np.sqrt(obs["roll"]**2 + obs["pitch"]**2)
        if tilt_magnitude > self.config.max_tilt_angle:
            return True, {
                "success": False,
                "timeout": False,
                "crashed": True,
                "reason": "excessive_tilt",
                "waypoints_reached": self.waypoints_reached
            }

        # Check timeout
        if self.check_timeout():
            self.is_timeout = True
            return True, {
                "success": False,
                "timeout": True,
                "crashed": False,
                "waypoints_reached": self.waypoints_reached
            }

        return False, {
            "success": False,
            "timeout": False,
            "crashed": False,
            "waypoints_reached": self.waypoints_reached
        }

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset task to initial state

        - Reset robot to start position (in air)
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
        """Generate random 3D waypoints in the flight area"""
        num_waypoints = np.random.randint(3, 6)  # 3-5 waypoints

        self.waypoints = []
        for _ in range(num_waypoints):
            # Random position in flight area
            x = np.random.uniform(-self.config.flight_area_size / 2,
                                 self.config.flight_area_size / 2)
            y = np.random.uniform(-self.config.flight_area_size / 2,
                                 self.config.flight_area_size / 2)
            z = np.random.uniform(self.config.altitude_min + 0.5,
                                 min(self.config.altitude_max - 0.5, 3.0))

            self.waypoints.append(np.array([x, y, z]))

    def check_waypoint_reached(self) -> bool:
        """
        Check if current waypoint has been reached

        Returns:
            True if waypoint reached
        """
        obs = self.get_observations()
        return obs["distance_to_waypoint"] < self.config.waypoint_threshold

    def advance_waypoint(self):
        """Move to next waypoint"""
        if self.current_waypoint_idx < len(self.waypoints):
            self.waypoints_reached += 1
            self.current_waypoint_idx += 1

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Extended step with automatic waypoint advancement

        Args:
            action: Optional action (thrust + moments)

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

    def visualize_waypoints(self):
        """Print waypoint information (for debugging)"""
        print(f"\nFlight Task:")
        print(f"  Total waypoints: {len(self.waypoints)}")
        print(f"  Current waypoint: {self.current_waypoint_idx}")
        print(f"  Reached: {self.waypoints_reached}/{self.total_waypoints}")
        print(f"  Waypoints:")
        for i, wp in enumerate(self.waypoints):
            marker = " → " if i == self.current_waypoint_idx else "   "
            status = "✓" if i < self.current_waypoint_idx else " "
            print(f"    {marker}[{status}] {i}: [{wp[0]:6.2f}, {wp[1]:6.2f}, {wp[2]:6.2f}]")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from backend.assets.robot_configs import SIMPLE_QUADCOPTER_CONFIG
    from backend.simulation.pybullet_interface import PyBulletRobotInterface
    from backend.controllers import DroneController
    import time

    print("Flight Task Example")
    print("=" * 50)

    # Create robot
    robot = PyBulletRobotInterface(SIMPLE_QUADCOPTER_CONFIG, use_gui=True)

    # Create task with waypoints
    waypoints = [
        np.array([1.0, 0.0, 2.0]),
        np.array([1.0, 1.0, 2.5]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.0, 2.0]),
    ]

    config = FlightTaskConfig(
        waypoint_threshold=0.3,
        timeout_steps=2000
    )
    task = FlightTask(robot, waypoints=waypoints, config=config)

    # Create controller
    controller = DroneController(mass=1.5)

    # Reset task
    obs = task.reset()
    task.visualize_waypoints()

    done = False
    total_reward = 0.0

    print(f"\nStarting flight...")

    while not done:
        # Get current state
        current_pos = obs["position"]
        current_quat = obs["orientation"]
        current_vel = obs["linear_velocity"]
        target_wp = obs["target_waypoint"]

        # Compute control (hover to waypoint)
        thrust, moments = controller.compute_hover(
            current_pos, target_wp, current_vel, current_quat
        )

        # Step task (simplified - actual would apply thrust/moments)
        obs, reward, done, info = task.step()
        total_reward += reward

        # Print progress every 100 steps
        if task.step_count % 100 == 0:
            dist = obs["distance_to_waypoint"]
            alt = obs["altitude"]
            tilt = np.degrees(np.sqrt(obs["roll"]**2 + obs["pitch"]**2))
            print(f"  Step {task.step_count:4d} | "
                  f"WP {obs['waypoint_index']}/{len(waypoints)} | "
                  f"Dist: {dist:5.2f}m | "
                  f"Alt: {alt:5.2f}m | "
                  f"Tilt: {tilt:5.1f}°")

    # Episode results
    print(f"\nEpisode Results:")
    print(f"  Success: {info['success']}")
    print(f"  Waypoints reached: {info['waypoints_reached']}/{task.total_waypoints}")
    if 'crashed' in info and info['crashed']:
        print(f"  Crashed: Yes (reason: {info.get('reason', 'unknown')})")
    print(f"  Total steps: {task.step_count}")
    print(f"  Total reward: {total_reward:.2f}")

    input("\nPress Enter to exit...")
    robot.disconnect()
