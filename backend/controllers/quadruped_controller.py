"""
Quadruped Gait Controller

Simplified gait controller for quadruped robots.
Generates leg trajectories for walking, trotting, and standing.

Supports:
    - Standing pose control
    - Walking gait (slow, stable)
    - Trotting gait (diagonal pairs)
    - Body height/velocity control

Usage:
    from backend.controllers.quadruped_controller import QuadrupedController, GaitType

    controller = QuadrupedController()
    joint_targets = controller.compute_gait(
        gait_type=GaitType.TROT,
        forward_velocity=0.5,  # 0.5 m/s
        time=current_time
    )

References:
    - Classic quadruped gaits
    - Simple swing/stance phase generation
    - Inspired by Isaac Lab locomotion controllers
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GaitType(Enum):
    """Quadruped gait types"""
    STAND = "stand"      # Standing still
    WALK = "walk"        # Walking (1 leg at a time)
    TROT = "trot"        # Trotting (diagonal pairs)
    PACE = "pace"        # Pacing (lateral pairs)
    BOUND = "bound"      # Bounding (front/back pairs)


@dataclass
class QuadrupedControllerConfig:
    """
    Configuration for quadruped controller

    Attributes:
        leg_order: Order of legs ["LF", "RF", "LH", "RH"]
        stance_height: Default standing height (meters)
        step_height: Foot lift height during swing (meters)
        step_length: Forward step length (meters)
        step_width: Lateral step width (meters)
        gait_frequency: Gait cycle frequency (Hz)
        duty_factor: Fraction of time foot in contact (0-1)
    """
    leg_order: List[str] = None  # Will be set in __post_init__
    stance_height: float = 0.3  # 30cm standing height
    step_height: float = 0.05   # 5cm foot clearance
    step_length: float = 0.1    # 10cm step
    step_width: float = 0.0     # No lateral motion
    gait_frequency: float = 1.0  # 1 Hz (1 step per second)
    duty_factor: float = 0.75    # 75% stance, 25% swing

    def __post_init__(self):
        if self.leg_order is None:
            self.leg_order = ["LF", "RF", "LH", "RH"]


class QuadrupedController:
    """
    Simplified quadruped gait controller

    Generates joint angle targets for quadruped locomotion using
    simplified gait patterns and inverse kinematics.

    Note: This is a simplified controller. For advanced locomotion,
    consider using full MPC or RL-based controllers.
    """

    def __init__(self, config: Optional[QuadrupedControllerConfig] = None):
        """
        Initialize quadruped controller

        Args:
            config: QuadrupedControllerConfig
        """
        self.config = config or QuadrupedControllerConfig()

        # Gait phase offsets for each leg
        self.phase_offsets = self._get_gait_phase_offsets(GaitType.TROT)

    # ========================================================================
    # Main Interface
    # ========================================================================

    def compute_gait(
        self,
        gait_type: GaitType,
        forward_velocity: float = 0.0,
        lateral_velocity: float = 0.0,
        yaw_rate: float = 0.0,
        time: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Compute foot positions for current gait

        Args:
            gait_type: Type of gait to use
            forward_velocity: Forward velocity (m/s)
            lateral_velocity: Lateral velocity (m/s)
            yaw_rate: Yaw rate (rad/s)
            time: Current time (seconds)

        Returns:
            Dictionary mapping leg names to foot positions [x, y, z]

        Example:
            >>> controller = QuadrupedController()
            >>> foot_pos = controller.compute_gait(
            ...     gait_type=GaitType.TROT,
            ...     forward_velocity=0.5,
            ...     time=1.0
            ... )
            >>> print(foot_pos["LF"])  # Left front foot position
        """
        # Update gait phase offsets if gait type changed
        self.phase_offsets = self._get_gait_phase_offsets(gait_type)

        # Compute foot positions for each leg
        foot_positions = {}

        for leg_name in self.config.leg_order:
            phase = self._get_leg_phase(leg_name, time)
            foot_pos = self._compute_foot_position(
                leg_name,
                phase,
                forward_velocity,
                lateral_velocity
            )
            foot_positions[leg_name] = foot_pos

        return foot_positions

    def compute_joint_angles(
        self,
        foot_positions: Dict[str, np.ndarray],
        base_height: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Convert foot positions to joint angles using IK

        Simple 3-DOF leg inverse kinematics (HAA-HFE-KFE).

        Args:
            foot_positions: Dict of foot positions per leg
            base_height: Body height (uses default if None)

        Returns:
            Dictionary mapping leg names to joint angles [HAA, HFE, KFE]

        Note:
            This is simplified planar IK. For precise control, use
            full 3D inverse kinematics or optimization-based methods.
        """
        if base_height is None:
            base_height = self.config.stance_height

        joint_angles = {}

        for leg_name, foot_pos in foot_positions.items():
            angles = self._leg_ik(leg_name, foot_pos, base_height)
            joint_angles[leg_name] = angles

        return joint_angles

    # ========================================================================
    # Gait Generation
    # ========================================================================

    def _get_gait_phase_offsets(self, gait_type: GaitType) -> Dict[str, float]:
        """
        Get phase offsets for each leg based on gait type

        Phase offset determines when each leg starts its cycle.

        Returns:
            Dictionary mapping leg names to phase offsets (0-1)
        """
        if gait_type == GaitType.STAND:
            # All legs on ground
            return {leg: 0.0 for leg in self.config.leg_order}

        elif gait_type == GaitType.WALK:
            # Walking: LF -> RH -> RF -> LH
            return {
                "LF": 0.0,
                "RH": 0.25,
                "RF": 0.5,
                "LH": 0.75
            }

        elif gait_type == GaitType.TROT:
            # Trotting: LF+RH together, RF+LH together
            return {
                "LF": 0.0,
                "RH": 0.0,
                "RF": 0.5,
                "LH": 0.5
            }

        elif gait_type == GaitType.PACE:
            # Pacing: Left pair, then right pair
            return {
                "LF": 0.0,
                "LH": 0.0,
                "RF": 0.5,
                "RH": 0.5
            }

        elif gait_type == GaitType.BOUND:
            # Bounding: Front pair, then hind pair
            return {
                "LF": 0.0,
                "RF": 0.0,
                "LH": 0.5,
                "RH": 0.5
            }

        return {leg: 0.0 for leg in self.config.leg_order}

    def _get_leg_phase(self, leg_name: str, time: float) -> float:
        """
        Get current phase (0-1) for a leg

        Args:
            leg_name: Name of leg
            time: Current time (seconds)

        Returns:
            Phase in range [0, 1]
        """
        # Gait cycle phase
        cycle_phase = (time * self.config.gait_frequency) % 1.0

        # Add leg offset
        leg_offset = self.phase_offsets.get(leg_name, 0.0)
        phase = (cycle_phase + leg_offset) % 1.0

        return phase

    def _compute_foot_position(
        self,
        leg_name: str,
        phase: float,
        forward_vel: float,
        lateral_vel: float
    ) -> np.ndarray:
        """
        Compute foot position for given phase

        Implements swing/stance trajectory.

        Args:
            leg_name: Name of leg
            phase: Gait phase (0-1)
            forward_vel: Forward velocity (m/s)
            lateral_vel: Lateral velocity (m/s)

        Returns:
            Foot position [x, y, z] in body frame
        """
        # Default foot position (stance)
        if leg_name == "LF":
            base_pos = np.array([0.2, 0.15, 0.0])   # Front left
        elif leg_name == "RF":
            base_pos = np.array([0.2, -0.15, 0.0])  # Front right
        elif leg_name == "LH":
            base_pos = np.array([-0.2, 0.15, 0.0])  # Hind left
        else:  # RH
            base_pos = np.array([-0.2, -0.15, 0.0]) # Hind right

        # Determine if in swing or stance phase
        if phase < self.config.duty_factor:
            # Stance phase: foot on ground, moving backward relative to body
            stance_progress = phase / self.config.duty_factor

            # Foot moves backward as body moves forward
            x_offset = self.config.step_length * (0.5 - stance_progress)
            y_offset = self.config.step_width * (0.5 - stance_progress) if lateral_vel != 0 else 0.0
            z_offset = 0.0  # On ground

        else:
            # Swing phase: foot in air, moving forward
            swing_progress = (phase - self.config.duty_factor) / (1.0 - self.config.duty_factor)

            # Foot swings forward with arc motion
            x_offset = self.config.step_length * (-0.5 + swing_progress)
            y_offset = 0.0
            z_offset = self.config.step_height * np.sin(swing_progress * np.pi)

        foot_pos = base_pos + np.array([x_offset, y_offset, z_offset])

        return foot_pos

    # ========================================================================
    # Inverse Kinematics
    # ========================================================================

    def _leg_ik(
        self,
        leg_name: str,
        foot_pos: np.ndarray,
        base_height: float
    ) -> np.ndarray:
        """
        Simple 3-DOF leg inverse kinematics

        Solves for joint angles [HAA, HFE, KFE] given foot position.

        Args:
            leg_name: Name of leg
            foot_pos: Desired foot position [x, y, z] in body frame
            base_height: Body height above ground

        Returns:
            Joint angles [HAA, HFE, KFE] in radians

        Note:
            Simplified planar IK. Assumes leg structure:
            - HAA: Hip abduction/adduction (y-axis rotation)
            - HFE: Hip flexion/extension (pitch)
            - KFE: Knee flexion/extension (pitch)
        """
        # Simplified leg dimensions (typical quadruped)
        hip_length = 0.08  # 8cm hip offset
        thigh_length = 0.2  # 20cm thigh
        shank_length = 0.2  # 20cm shank

        # Foot position relative to hip
        x, y, z = foot_pos
        z_from_hip = -(base_height + z)  # Flip z (down is positive)

        # HAA angle (hip abduction/adduction)
        # Simple approximation based on lateral distance
        haa = np.arctan2(abs(y) - hip_length, abs(z_from_hip))
        if "L" in leg_name:  # Left legs
            haa = -haa

        # Project to sagittal plane for HFE and KFE
        leg_extension = np.sqrt(x**2 + z_from_hip**2)

        # Use 2-link IK for HFE and KFE
        try:
            hfe, kfe = self._two_link_ik(leg_extension, thigh_length, shank_length)
        except ValueError:
            # Unreachable, use default angles
            hfe = 0.4
            kfe = -0.8

        # Adjust HFE based on forward/backward position
        if x > 0:  # Front legs
            hfe += np.arctan2(x, abs(z_from_hip))
        else:  # Hind legs
            hfe -= np.arctan2(abs(x), abs(z_from_hip))

        return np.array([haa, hfe, kfe])

    def _two_link_ik(
        self,
        target_distance: float,
        link1_length: float,
        link2_length: float
    ) -> Tuple[float, float]:
        """
        2-link planar inverse kinematics

        Args:
            target_distance: Distance to target
            link1_length: Length of first link
            link2_length: Length of second link

        Returns:
            (theta1, theta2) joint angles

        Raises:
            ValueError if target unreachable
        """
        # Check reachability
        max_reach = link1_length + link2_length
        min_reach = abs(link1_length - link2_length)

        if target_distance > max_reach or target_distance < min_reach:
            raise ValueError("Target unreachable")

        # Law of cosines
        cos_theta2 = (target_distance**2 - link1_length**2 - link2_length**2) / \
                     (2 * link1_length * link2_length)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

        theta2 = -np.arccos(cos_theta2)  # Negative for elbow-down config

        # Angle of first link
        k1 = link1_length + link2_length * np.cos(theta2)
        k2 = link2_length * np.sin(theta2)
        theta1 = np.arctan2(k2, k1)

        return theta1, theta2


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Quadruped Gait Controller Test")
    print("=" * 50)

    # Create controller
    config = QuadrupedControllerConfig(
        stance_height=0.3,
        step_height=0.05,
        step_length=0.1,
        gait_frequency=1.0
    )
    controller = QuadrupedController(config)

    # Test gaits
    for gait in [GaitType.STAND, GaitType.WALK, GaitType.TROT]:
        print(f"\n{gait.value.upper()} Gait:")
        print("-" * 50)

        # Simulate one gait cycle
        time_steps = [0.0, 0.25, 0.5, 0.75]

        for t in time_steps:
            print(f"\n  Time: {t:.2f}s")

            # Compute foot positions
            foot_pos = controller.compute_gait(
                gait_type=gait,
                forward_velocity=0.5,
                time=t
            )

            # Print which legs are in swing (z > 0)
            swing_legs = [leg for leg, pos in foot_pos.items() if pos[2] > 0.01]
            stance_legs = [leg for leg in controller.config.leg_order if leg not in swing_legs]

            print(f"    Swing:  {swing_legs if swing_legs else 'None'}")
            print(f"    Stance: {stance_legs}")

    # Test joint angle computation
    print("\n" + "=" * 50)
    print("Joint Angle Computation Test:")

    foot_positions = controller.compute_gait(
        gait_type=GaitType.STAND,
        time=0.0
    )

    joint_angles = controller.compute_joint_angles(foot_positions)

    print("\nStanding pose joint angles:")
    for leg, angles in joint_angles.items():
        print(f"  {leg}: HAA={np.degrees(angles[0]):6.1f}°, "
              f"HFE={np.degrees(angles[1]):6.1f}°, "
              f"KFE={np.degrees(angles[2]):6.1f}°")

    print("\n" + "=" * 50)
