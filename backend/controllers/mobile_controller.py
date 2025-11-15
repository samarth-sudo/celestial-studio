"""
Mobile Robot Controller

Controller for wheeled mobile robots with differential drive kinematics.
Converts velocity commands (linear, angular) into wheel velocities.

Supports:
    - Differential drive (2 wheels)
    - Trajectory tracking
    - Velocity control
    - Position control (with feedback)

Usage:
    from backend.controllers.mobile_controller import MobileController

    controller = MobileController(wheel_base=0.3, wheel_radius=0.05)
    left_vel, right_vel = controller.compute_wheel_velocities(
        linear_vel=0.5,   # 0.5 m/s forward
        angular_vel=0.2   # 0.2 rad/s turning
    )

References:
    - Differential drive kinematics
    - Pure pursuit for path following
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from backend.simulation.pybullet_interface import PyBulletRobotInterface


@dataclass
class MobileControllerConfig:
    """
    Configuration for mobile robot controller

    Attributes:
        wheel_base: Distance between left and right wheels (meters)
        wheel_radius: Wheel radius (meters)
        max_linear_velocity: Maximum linear speed (m/s)
        max_angular_velocity: Maximum angular speed (rad/s)
        position_gain: Proportional gain for position control
        heading_gain: Proportional gain for heading control
        lookahead_distance: Lookahead distance for pure pursuit (meters)
    """
    wheel_base: float = 0.3  # 30cm wheel base
    wheel_radius: float = 0.05  # 5cm wheel radius
    max_linear_velocity: float = 1.0  # 1 m/s
    max_angular_velocity: float = 2.0  # 2 rad/s (114 deg/s)
    position_gain: float = 1.0
    heading_gain: float = 2.0
    lookahead_distance: float = 0.5  # 50cm


class MobileController:
    """
    Differential drive mobile robot controller

    Converts high-level velocity commands to wheel velocities.
    Implements differential drive inverse kinematics.

    Kinematics:
        v = (v_left + v_right) / 2            # Linear velocity
        ω = (v_right - v_left) / wheel_base   # Angular velocity

        Inverse:
        v_left = v - (ω * wheel_base) / 2
        v_right = v + (ω * wheel_base) / 2
    """

    def __init__(self, config: Optional[MobileControllerConfig] = None):
        """
        Initialize mobile controller

        Args:
            config: MobileControllerConfig (uses defaults if None)
        """
        self.config = config or MobileControllerConfig()

    # ========================================================================
    # Velocity Control
    # ========================================================================

    def compute_wheel_velocities(
        self,
        linear_vel: float,
        angular_vel: float
    ) -> Tuple[float, float]:
        """
        Convert linear and angular velocities to wheel velocities

        Args:
            linear_vel: Desired linear velocity (m/s, positive = forward)
            angular_vel: Desired angular velocity (rad/s, positive = CCW)

        Returns:
            (left_wheel_vel, right_wheel_vel) in rad/s

        Example:
            >>> controller = MobileController()
            >>> left, right = controller.compute_wheel_velocities(0.5, 0.2)
            >>> print(f"Left: {left:.2f} rad/s, Right: {right:.2f} rad/s")
        """
        # Clip velocities
        linear_vel = np.clip(linear_vel, -self.config.max_linear_velocity,
                            self.config.max_linear_velocity)
        angular_vel = np.clip(angular_vel, -self.config.max_angular_velocity,
                             self.config.max_angular_velocity)

        # Differential drive inverse kinematics
        v_left = linear_vel - (angular_vel * self.config.wheel_base) / 2.0
        v_right = linear_vel + (angular_vel * self.config.wheel_base) / 2.0

        # Convert from m/s to rad/s
        left_wheel_vel = v_left / self.config.wheel_radius
        right_wheel_vel = v_right / self.config.wheel_radius

        return left_wheel_vel, right_wheel_vel

    def compute_forward_kinematics(
        self,
        left_wheel_vel: float,
        right_wheel_vel: float
    ) -> Tuple[float, float]:
        """
        Convert wheel velocities to linear and angular velocities

        Args:
            left_wheel_vel: Left wheel velocity (rad/s)
            right_wheel_vel: Right wheel velocity (rad/s)

        Returns:
            (linear_vel, angular_vel) in m/s and rad/s
        """
        # Convert to m/s
        v_left = left_wheel_vel * self.config.wheel_radius
        v_right = right_wheel_vel * self.config.wheel_radius

        # Forward kinematics
        linear_vel = (v_left + v_right) / 2.0
        angular_vel = (v_right - v_left) / self.config.wheel_base

        return linear_vel, angular_vel

    # ========================================================================
    # Position Control
    # ========================================================================

    def compute_velocity_to_goal(
        self,
        current_pos: np.ndarray,
        current_yaw: float,
        goal_pos: np.ndarray,
        goal_yaw: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute velocity commands to reach a goal position

        Simple proportional controller for position and heading.

        Args:
            current_pos: Current position [x, y] in world frame
            current_yaw: Current heading (radians)
            goal_pos: Goal position [x, y] in world frame
            goal_yaw: Goal heading (optional, None = face goal)

        Returns:
            (linear_vel, angular_vel) commands

        Example:
            >>> current = np.array([0.0, 0.0])
            >>> goal = np.array([1.0, 0.0])
            >>> v, w = controller.compute_velocity_to_goal(current, 0.0, goal)
        """
        # Compute error
        error = goal_pos - current_pos
        distance = np.linalg.norm(error)

        # If at goal, stop
        if distance < 0.01:  # 1cm threshold
            return 0.0, 0.0

        # Desired heading (towards goal)
        if goal_yaw is None:
            desired_yaw = np.arctan2(error[1], error[0])
        else:
            desired_yaw = goal_yaw

        # Heading error
        heading_error = self._normalize_angle(desired_yaw - current_yaw)

        # Proportional control
        linear_vel = self.config.position_gain * distance
        angular_vel = self.config.heading_gain * heading_error

        # If heading error is large, slow down forward motion
        if abs(heading_error) > np.radians(30):  # 30 degrees
            linear_vel *= 0.5

        return linear_vel, angular_vel

    # ========================================================================
    # Path Following (Pure Pursuit)
    # ========================================================================

    def pure_pursuit(
        self,
        current_pos: np.ndarray,
        current_yaw: float,
        path: List[np.ndarray]
    ) -> Tuple[float, float]:
        """
        Pure pursuit path following algorithm

        Tracks a path by aiming at a lookahead point.

        Args:
            current_pos: Current position [x, y]
            current_yaw: Current heading (radians)
            path: List of waypoints [[x, y], [x, y], ...]

        Returns:
            (linear_vel, angular_vel) commands

        Reference:
            Pure Pursuit algorithm for path tracking
        """
        if len(path) == 0:
            return 0.0, 0.0

        # Find lookahead point
        lookahead_point = self._find_lookahead_point(current_pos, path)

        if lookahead_point is None:
            # Reached end of path
            return 0.0, 0.0

        # Compute desired heading
        error = lookahead_point - current_pos
        desired_yaw = np.arctan2(error[1], error[0])

        # Heading error
        heading_error = self._normalize_angle(desired_yaw - current_yaw)

        # Pure pursuit control law
        linear_vel = self.config.max_linear_velocity

        # Curvature (simplified)
        curvature = 2.0 * np.sin(heading_error) / self.config.lookahead_distance
        angular_vel = curvature * linear_vel

        return linear_vel, angular_vel

    def _find_lookahead_point(
        self,
        current_pos: np.ndarray,
        path: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Find point on path at lookahead distance"""
        lookahead_dist = self.config.lookahead_distance

        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0

        for i, point in enumerate(path):
            dist = np.linalg.norm(point - current_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Search forward from closest point
        for i in range(closest_idx, len(path)):
            dist = np.linalg.norm(path[i] - current_pos)
            if dist >= lookahead_dist:
                return path[i]

        # Return last point if not found
        if len(path) > 0:
            return path[-1]

        return None

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

    def compute_and_apply(
        self,
        robot: PyBulletRobotInterface,
        linear_vel: float,
        angular_vel: float
    ):
        """
        Compute wheel velocities and apply to robot

        Args:
            robot: PyBulletRobotInterface
            linear_vel: Desired linear velocity (m/s)
            angular_vel: Desired angular velocity (rad/s)
        """
        left_vel, right_vel = self.compute_wheel_velocities(linear_vel, angular_vel)

        # Apply to robot (assumes first two joints are wheels)
        wheel_velocities = np.array([left_vel, right_vel])
        robot.set_joint_velocities(wheel_velocities)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Mobile Robot Controller Test")
    print("=" * 50)

    # Create controller
    config = MobileControllerConfig(
        wheel_base=0.3,
        wheel_radius=0.05,
        max_linear_velocity=1.0,
        max_angular_velocity=2.0
    )
    controller = MobileController(config)

    # Test 1: Forward motion
    print("\nTest 1: Forward motion")
    left, right = controller.compute_wheel_velocities(linear_vel=0.5, angular_vel=0.0)
    print(f"  Linear: 0.5 m/s, Angular: 0.0 rad/s")
    print(f"  Left wheel: {left:.2f} rad/s")
    print(f"  Right wheel: {right:.2f} rad/s")

    # Test 2: Turning in place
    print("\nTest 2: Turning in place (CCW)")
    left, right = controller.compute_wheel_velocities(linear_vel=0.0, angular_vel=1.0)
    print(f"  Linear: 0.0 m/s, Angular: 1.0 rad/s")
    print(f"  Left wheel: {left:.2f} rad/s")
    print(f"  Right wheel: {right:.2f} rad/s")

    # Test 3: Arc motion
    print("\nTest 3: Arc motion")
    left, right = controller.compute_wheel_velocities(linear_vel=0.5, angular_vel=0.5)
    print(f"  Linear: 0.5 m/s, Angular: 0.5 rad/s")
    print(f"  Left wheel: {left:.2f} rad/s")
    print(f"  Right wheel: {right:.2f} rad/s")

    # Test 4: Forward kinematics
    print("\nTest 4: Forward kinematics")
    linear, angular = controller.compute_forward_kinematics(left, right)
    print(f"  Wheel velocities: L={left:.2f}, R={right:.2f} rad/s")
    print(f"  Recovered: Linear={linear:.2f} m/s, Angular={angular:.2f} rad/s")

    # Test 5: Goal seeking
    print("\nTest 5: Goal seeking")
    current_pos = np.array([0.0, 0.0])
    current_yaw = 0.0
    goal_pos = np.array([1.0, 1.0])

    linear, angular = controller.compute_velocity_to_goal(
        current_pos, current_yaw, goal_pos
    )
    print(f"  Current: {current_pos}, Yaw: {np.degrees(current_yaw):.1f}°")
    print(f"  Goal: {goal_pos}")
    print(f"  Command: Linear={linear:.2f} m/s, Angular={angular:.2f} rad/s")

    # Test 6: Pure pursuit
    print("\nTest 6: Pure pursuit path following")
    path = [
        np.array([0.5, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.5, 0.5]),
        np.array([2.0, 1.0]),
    ]
    linear, angular = controller.pure_pursuit(current_pos, current_yaw, path)
    print(f"  Path: {len(path)} waypoints")
    print(f"  Command: Linear={linear:.2f} m/s, Angular={angular:.2f} rad/s")

    print("\n" + "=" * 50)
