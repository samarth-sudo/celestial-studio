"""
Drone Flight Controller

Simplified quadcopter controller for position and attitude control.
Converts position/velocity commands into thrust and moment commands.

Supports:
    - Hovering at target position
    - Waypoint navigation
    - Velocity control
    - Attitude stabilization (PID)

Usage:
    from backend.controllers.drone_controller import DroneController

    controller = DroneController(mass=1.5)
    thrust, moments = controller.compute_control(
        current_pos=[0, 0, 2],
        target_pos=[1, 0, 2],
        current_vel=[0, 0, 0],
        current_quat=[1, 0, 0, 0]
    )

References:
    - Cascade PID control for quadcopters
    - Simplified dynamics model
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from backend.utils.math_utils import quat_to_euler, euler_to_quat, normalize_quat


@dataclass
class DroneControllerConfig:
    """
    Configuration for drone controller

    Attributes:
        mass: Drone mass (kg)
        gravity: Gravitational acceleration (m/s²)
        max_thrust: Maximum total thrust (N)
        max_tilt_angle: Maximum tilt angle (radians)

        # Position PID gains
        pos_kp: Position proportional gain
        pos_kd: Position derivative gain
        pos_ki: Position integral gain

        # Attitude PID gains
        att_kp: Attitude proportional gain
        att_kd: Attitude derivative gain

        # Altitude PID gains (z-axis)
        alt_kp: Altitude proportional gain
        alt_kd: Altitude derivative gain
    """
    mass: float = 1.5  # kg
    gravity: float = 9.81  # m/s²
    max_thrust: float = 20.0  # N (enough for ~1.5kg with margin)
    max_tilt_angle: float = 0.5  # ~29 degrees

    # Position control gains
    pos_kp: float = 2.0
    pos_kd: float = 1.0
    pos_ki: float = 0.1

    # Attitude control gains
    att_kp: float = 3.0
    att_kd: float = 0.5

    # Altitude control gains
    alt_kp: float = 5.0
    alt_kd: float = 2.0


class DroneController:
    """
    Cascade PID controller for quadcopter drones

    Control hierarchy:
        1. Position controller → desired acceleration
        2. Acceleration → desired attitude (roll, pitch)
        3. Attitude controller → torques
        4. Altitude controller → thrust

    Simplified model for educational purposes.
    For real drones, use more sophisticated controllers like MPC or INDI.
    """

    def __init__(self, config: Optional[DroneControllerConfig] = None):
        """
        Initialize drone controller

        Args:
            config: DroneControllerConfig
        """
        self.config = config or DroneControllerConfig()

        # Integral error accumulator
        self.pos_error_integral = np.zeros(3)

        # Previous errors for derivative
        self.prev_pos_error = np.zeros(3)

    # ========================================================================
    # Main Control Interface
    # ========================================================================

    def compute_control(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        current_vel: np.ndarray,
        current_quat: np.ndarray,
        target_yaw: float = 0.0,
        dt: float = 1/240.0
    ) -> Tuple[float, np.ndarray]:
        """
        Compute thrust and moment commands for drone

        Args:
            current_pos: Current position [x, y, z]
            target_pos: Target position [x, y, z]
            current_vel: Current velocity [vx, vy, vz]
            current_quat: Current orientation [w, x, y, z]
            target_yaw: Desired yaw angle (radians)
            dt: Time step (seconds)

        Returns:
            (total_thrust, moments) where:
                - total_thrust: Total upward thrust (N)
                - moments: Torques [roll_torque, pitch_torque, yaw_torque] (Nm)

        Example:
            >>> controller = DroneController()
            >>> thrust, moments = controller.compute_control(
            ...     current_pos=np.array([0, 0, 1.8]),
            ...     target_pos=np.array([1, 0, 2.0]),
            ...     current_vel=np.array([0.1, 0, 0.05]),
            ...     current_quat=np.array([1, 0, 0, 0])
            ... )
        """
        # 1. Position control → desired acceleration
        desired_acc = self._position_control(
            current_pos, target_pos, current_vel, dt
        )

        # 2. Desired acceleration → desired attitude
        desired_roll, desired_pitch = self._acceleration_to_attitude(
            desired_acc, target_yaw
        )

        # 3. Current attitude
        current_roll, current_pitch, current_yaw = quat_to_euler(current_quat)

        # 4. Attitude control → moments
        moments = self._attitude_control(
            current_roll, current_pitch, current_yaw,
            desired_roll, desired_pitch, target_yaw
        )

        # 5. Altitude control → thrust
        thrust = self._altitude_control(
            current_pos[2], target_pos[2],
            current_vel[2],
            current_roll, current_pitch
        )

        return thrust, moments

    def compute_hover(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        current_vel: np.ndarray,
        current_quat: np.ndarray,
        dt: float = 1/240.0
    ) -> Tuple[float, np.ndarray]:
        """
        Simplified hovering control (alias for compute_control with yaw=0)

        Args:
            current_pos: Current position [x, y, z]
            target_pos: Target position [x, y, z]
            current_vel: Current velocity [vx, vy, vz]
            current_quat: Current orientation [w, x, y, z]
            dt: Time step

        Returns:
            (thrust, moments)
        """
        return self.compute_control(
            current_pos, target_pos, current_vel, current_quat,
            target_yaw=0.0, dt=dt
        )

    # ========================================================================
    # Inner Loop Controllers
    # ========================================================================

    def _position_control(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        current_vel: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        PID position controller

        Returns:
            Desired acceleration [ax, ay, az] in world frame
        """
        # Position error
        pos_error = target_pos - current_pos

        # Integral
        self.pos_error_integral += pos_error * dt
        self.pos_error_integral = np.clip(self.pos_error_integral, -1.0, 1.0)

        # Derivative (use velocity directly)
        vel_error = -current_vel  # Target velocity is zero (hovering)

        # PID output
        desired_acc = (
            self.config.pos_kp * pos_error +
            self.config.pos_ki * self.pos_error_integral +
            self.config.pos_kd * vel_error
        )

        return desired_acc

    def _acceleration_to_attitude(
        self,
        desired_acc: np.ndarray,
        yaw: float
    ) -> Tuple[float, float]:
        """
        Convert desired acceleration to desired roll/pitch angles

        Uses small angle approximation.

        Args:
            desired_acc: Desired acceleration [ax, ay, az]
            yaw: Current yaw angle

        Returns:
            (desired_roll, desired_pitch) in radians
        """
        # Rotate desired acceleration to body frame
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        acc_body_x = desired_acc[0] * cos_yaw + desired_acc[1] * sin_yaw
        acc_body_y = -desired_acc[0] * sin_yaw + desired_acc[1] * cos_yaw

        # Small angle approximation
        # For quadcopter: ax = g * tan(pitch) ≈ g * pitch
        #                 ay = -g * tan(roll) ≈ -g * roll
        desired_pitch = acc_body_x / self.config.gravity
        desired_roll = -acc_body_y / self.config.gravity

        # Clip to max tilt
        desired_pitch = np.clip(desired_pitch, -self.config.max_tilt_angle,
                               self.config.max_tilt_angle)
        desired_roll = np.clip(desired_roll, -self.config.max_tilt_angle,
                              self.config.max_tilt_angle)

        return desired_roll, desired_pitch

    def _attitude_control(
        self,
        current_roll: float,
        current_pitch: float,
        current_yaw: float,
        desired_roll: float,
        desired_pitch: float,
        desired_yaw: float
    ) -> np.ndarray:
        """
        PD attitude controller

        Args:
            current_roll, current_pitch, current_yaw: Current angles (rad)
            desired_roll, desired_pitch, desired_yaw: Desired angles (rad)

        Returns:
            Moments [Mx, My, Mz] in body frame (Nm)
        """
        # Angle errors
        roll_error = desired_roll - current_roll
        pitch_error = desired_pitch - current_pitch
        yaw_error = self._normalize_angle(desired_yaw - current_yaw)

        # PD control (simplified, no rate measurements)
        moments = np.array([
            self.config.att_kp * roll_error,
            self.config.att_kp * pitch_error,
            self.config.att_kp * yaw_error * 0.5  # Lower gain for yaw
        ])

        return moments

    def _altitude_control(
        self,
        current_z: float,
        target_z: float,
        current_vz: float,
        roll: float,
        pitch: float
    ) -> float:
        """
        PD altitude controller

        Args:
            current_z: Current altitude (m)
            target_z: Target altitude (m)
            current_vz: Current vertical velocity (m/s)
            roll, pitch: Current roll/pitch for thrust compensation

        Returns:
            Total thrust (N)
        """
        # Altitude error
        z_error = target_z - current_z

        # PD control
        thrust_acc = (
            self.config.alt_kp * z_error -
            self.config.alt_kd * current_vz +
            self.config.gravity  # Gravity compensation
        )

        # Compensate for tilt (thrust is reduced when tilted)
        cos_tilt = np.cos(roll) * np.cos(pitch)
        if cos_tilt > 0.1:  # Avoid division by zero
            thrust_acc = thrust_acc / cos_tilt

        # Convert to thrust force
        thrust = self.config.mass * thrust_acc

        # Clip to max thrust
        thrust = np.clip(thrust, 0.0, self.config.max_thrust)

        return thrust

    # ========================================================================
    # Waypoint Navigation
    # ========================================================================

    def follow_waypoints(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        current_quat: np.ndarray,
        waypoints: list,
        current_waypoint_idx: int,
        dt: float = 1/240.0
    ) -> Tuple[float, np.ndarray, int]:
        """
        Follow a series of waypoints

        Args:
            current_pos: Current position [x, y, z]
            current_vel: Current velocity [vx, vy, vz]
            current_quat: Current orientation [w, x, y, z]
            waypoints: List of waypoint positions [[x,y,z], ...]
            current_waypoint_idx: Index of current target waypoint
            dt: Time step

        Returns:
            (thrust, moments, next_waypoint_idx)
        """
        if current_waypoint_idx >= len(waypoints):
            # Reached end, hover at last position
            target_pos = waypoints[-1]
            next_idx = current_waypoint_idx
        else:
            target_pos = waypoints[current_waypoint_idx]

            # Check if reached current waypoint
            distance = np.linalg.norm(current_pos - target_pos)
            if distance < 0.1:  # 10cm threshold
                # Move to next waypoint
                next_idx = current_waypoint_idx + 1
            else:
                next_idx = current_waypoint_idx

        # Compute control to reach target
        thrust, moments = self.compute_hover(
            current_pos, target_pos, current_vel, current_quat, dt
        )

        return thrust, moments, next_idx

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

    def reset(self):
        """Reset integral accumulators"""
        self.pos_error_integral = np.zeros(3)
        self.prev_pos_error = np.zeros(3)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Drone Flight Controller Test")
    print("=" * 50)

    # Create controller
    config = DroneControllerConfig(
        mass=1.5,
        pos_kp=2.0,
        alt_kp=5.0
    )
    controller = DroneController(config)

    # Test 1: Hovering control
    print("\nTest 1: Hovering at target")
    current_pos = np.array([0.0, 0.0, 1.8])
    target_pos = np.array([0.0, 0.0, 2.0])
    current_vel = np.array([0.0, 0.0, 0.1])
    current_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity (level)

    thrust, moments = controller.compute_hover(
        current_pos, target_pos, current_vel, current_quat
    )

    print(f"  Current: {current_pos}, Target: {target_pos}")
    print(f"  Thrust: {thrust:.2f} N (hover={config.mass * config.gravity:.2f} N)")
    print(f"  Moments: [{moments[0]:.3f}, {moments[1]:.3f}, {moments[2]:.3f}] Nm")

    # Test 2: Moving to new position
    print("\nTest 2: Moving to new position")
    target_pos = np.array([1.0, 0.5, 2.0])

    thrust, moments = controller.compute_hover(
        current_pos, target_pos, current_vel, current_quat
    )

    print(f"  Current: {current_pos}, Target: {target_pos}")
    print(f"  Distance: {np.linalg.norm(target_pos - current_pos):.2f} m")
    print(f"  Thrust: {thrust:.2f} N")
    print(f"  Moments: [{moments[0]:.3f}, {moments[1]:.3f}, {moments[2]:.3f}] Nm")
    print(f"  → Controller will tilt forward/right to accelerate")

    # Test 3: Waypoint following
    print("\nTest 3: Waypoint navigation")
    waypoints = [
        np.array([1.0, 0.0, 2.0]),
        np.array([1.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.5]),
        np.array([0.0, 0.0, 2.0]),
    ]

    current_waypoint = 0
    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Starting at waypoint {current_waypoint}: {waypoints[current_waypoint]}")

    thrust, moments, next_waypoint = controller.follow_waypoints(
        current_pos, current_vel, current_quat,
        waypoints, current_waypoint
    )

    print(f"  Thrust: {thrust:.2f} N")
    print(f"  Next waypoint: {next_waypoint}")

    # Test 4: Tilted drone (attitude correction)
    print("\nTest 4: Attitude stabilization (tilted drone)")
    tilted_quat = euler_to_quat(0.2, 0.1, 0.0)  # 11° roll, 6° pitch

    thrust, moments = controller.compute_hover(
        current_pos, target_pos, current_vel, tilted_quat
    )

    roll, pitch, yaw = quat_to_euler(tilted_quat)
    print(f"  Current attitude: Roll={np.degrees(roll):.1f}°, "
          f"Pitch={np.degrees(pitch):.1f}°, Yaw={np.degrees(yaw):.1f}°")
    print(f"  Moments (to level): [{moments[0]:.3f}, {moments[1]:.3f}, {moments[2]:.3f}] Nm")

    print("\n" + "=" * 50)
