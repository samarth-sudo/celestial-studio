"""
Robot Configuration System

Clean, modular robot definitions inspired by Isaac Lab.
Each robot configuration includes URDF path, initial pose, actuator settings, and end-effector info.

Usage:
    from backend.assets.robot_configs import FRANKA_PANDA_CONFIG
    robot = PyBulletRobotInterface(FRANKA_PANDA_CONFIG)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import re


class RobotType(Enum):
    """
    Robot type classification

    Types:
        MANIPULATOR: Fixed-base robotic arms (e.g., Franka Panda, UR5)
        MOBILE: Wheeled robots (e.g., TurtleBot, differential drive)
        QUADRUPED: Four-legged robots (e.g., ANYmal, Spot, Unitree)
        HUMANOID: Bipedal humanoid robots (e.g., Atlas, H1, G1)
        AERIAL: Flying robots/drones (e.g., quadcopters)
        MOBILE_MANIPULATOR: Mobile base + arm (e.g., Fetch, TIAGo)
    """
    MANIPULATOR = "manipulator"
    MOBILE = "mobile"
    QUADRUPED = "quadruped"
    HUMANOID = "humanoid"
    AERIAL = "aerial"
    MOBILE_MANIPULATOR = "mobile_manipulator"


@dataclass
class ActuatorGroupConfig:
    """
    Configuration for a group of joints with similar actuation properties

    Uses regex patterns to match joint names, allowing clean grouping.

    Example:
        shoulder = ActuatorGroupConfig(
            joint_pattern="panda_joint[1-4]",  # Matches joints 1-4
            max_force=87.0,
            kp=80.0,
            kd=4.0
        )
    """
    joint_pattern: str      # Regex pattern to match joint names
    max_force: float        # Maximum force/torque (N or Nm)
    kp: float              # Proportional gain for PD control
    kd: float              # Derivative gain for PD control

    def matches(self, joint_name: str) -> bool:
        """Check if joint name matches this actuator group pattern"""
        return bool(re.match(self.joint_pattern, joint_name))


@dataclass
class RobotConfig:
    """
    Complete robot configuration for all robot types

    Attributes:
        urdf_path: Path to URDF file (relative to project root)
        robot_type: Type of robot (manipulator, mobile, quadruped, etc.)
        initial_joint_pos: Dictionary mapping joint names (or patterns) to initial positions
        actuator_groups: Dictionary of actuator configurations by group name

        # Manipulation-specific
        ee_link_name: Name of end-effector link (for manipulators)
        ee_offset: Tool center point offset from EE link [x, y, z]

        # Locomotion-specific
        base_link_name: Name of base/trunk link (for mobile/legged robots)
        foot_link_names: Names of foot/contact links (for legged robots)
        wheel_joint_names: Names of wheel joints (for wheeled robots)

        # General
        base_position: Initial base position [x, y, z]
        base_orientation: Initial base orientation [roll, pitch, yaw]
        self_collision_enabled: Enable self-collision detection
        fixed_base: Whether robot base is fixed or floating

        # Physical properties
        total_mass: Total robot mass (kg) (optional, for drones/mobile)
        max_linear_velocity: Maximum base linear velocity (m/s)
        max_angular_velocity: Maximum base angular velocity (rad/s)

    Example (Manipulator):
        config = RobotConfig(
            urdf_path="backend/assets/urdf/franka_panda.urdf",
            robot_type=RobotType.MANIPULATOR,
            initial_joint_pos={"panda_joint1": 0.0, ...},
            actuator_groups={"arm": ActuatorGroupConfig(...)},
            ee_link_name="panda_hand",
            ee_offset=[0.0, 0.0, 0.107]
        )

    Example (Quadruped):
        config = RobotConfig(
            urdf_path="backend/assets/urdf/anymal_c.urdf",
            robot_type=RobotType.QUADRUPED,
            base_link_name="base",
            foot_link_names=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
            fixed_base=False
        )
    """
    urdf_path: str
    robot_type: RobotType
    initial_joint_pos: Dict[str, float]
    actuator_groups: Dict[str, ActuatorGroupConfig]

    # Manipulation-specific (optional)
    ee_link_name: Optional[str] = None
    ee_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Locomotion-specific (optional)
    base_link_name: Optional[str] = None
    foot_link_names: List[str] = field(default_factory=list)
    wheel_joint_names: List[str] = field(default_factory=list)

    # General
    base_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # Roll, pitch, yaw
    self_collision_enabled: bool = True
    fixed_base: bool = True

    # Physical properties (optional)
    total_mass: Optional[float] = None
    max_linear_velocity: float = 1.0   # m/s
    max_angular_velocity: float = 1.0  # rad/s

    def get_actuator_for_joint(self, joint_name: str) -> ActuatorGroupConfig:
        """
        Find actuator configuration for a specific joint

        Args:
            joint_name: Name of joint

        Returns:
            ActuatorGroupConfig for this joint

        Raises:
            ValueError if no matching actuator group found
        """
        for group_name, actuator in self.actuator_groups.items():
            if actuator.matches(joint_name):
                return actuator

        raise ValueError(f"No actuator group matches joint: {joint_name}")

    def expand_initial_positions(self, joint_names: List[str]) -> Dict[str, float]:
        """
        Expand initial joint positions, handling regex patterns

        Args:
            joint_names: List of actual joint names from URDF

        Returns:
            Dictionary mapping each joint name to initial position
        """
        expanded = {}

        for joint_name in joint_names:
            # First try exact match
            if joint_name in self.initial_joint_pos:
                expanded[joint_name] = self.initial_joint_pos[joint_name]
                continue

            # Try pattern matching
            for pattern, value in self.initial_joint_pos.items():
                if re.match(pattern, joint_name):
                    expanded[joint_name] = value
                    break

            # Default to 0.0 if no match found
            if joint_name not in expanded:
                expanded[joint_name] = 0.0

        return expanded


# ============================================================================
# Pre-configured Robots
# ============================================================================

FRANKA_PANDA_CONFIG = RobotConfig(
    urdf_path="backend/assets/urdf/franka_panda.urdf",
    robot_type=RobotType.MANIPULATOR,

    # Initial joint positions (home/ready pose)
    # From Isaac Lab: Arm extended forward, gripper open
    initial_joint_pos={
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,     # ~-32.6 degrees
        "panda_joint3": 0.0,
        "panda_joint4": -2.810,     # ~-161 degrees
        "panda_joint5": 0.0,
        "panda_joint6": 3.037,      # ~174 degrees
        "panda_joint7": 0.741,      # ~42.5 degrees
        "panda_finger_joint.*": 0.04,  # Gripper open (4cm)
    },

    # Actuator groups with PD gains from Isaac Lab
    actuator_groups={
        "panda_shoulder": ActuatorGroupConfig(
            joint_pattern=r"panda_joint[1-4]",  # Shoulder joints 1-4
            max_force=87.0,                      # Newton-meters
            kp=80.0,                             # Position gain
            kd=4.0                               # Velocity gain
        ),
        "panda_forearm": ActuatorGroupConfig(
            joint_pattern=r"panda_joint[5-7]",  # Forearm joints 5-7
            max_force=12.0,
            kp=80.0,
            kd=4.0
        ),
        "panda_hand": ActuatorGroupConfig(
            joint_pattern=r"panda_finger_joint.*",  # Gripper fingers
            max_force=200.0,
            kp=2000.0,                              # Stiff for grasping
            kd=100.0
        ),
    },

    # End-effector configuration
    ee_link_name="panda_hand",
    ee_offset=[0.0, 0.0, 0.107],  # TCP 10.7cm forward from hand frame

    # Base configuration
    base_position=[0.0, 0.0, 0.0],
    base_orientation=[0.0, 0.0, 0.0],
    fixed_base=True,
    self_collision_enabled=True
)
"""
Franka Emika Panda robot with Panda hand gripper
- 7-DOF manipulator + 2-DOF parallel gripper
- Payload: 3 kg
- Reach: 855 mm
- Best for: Research, pick-and-place, collaborative tasks
"""


FRANKA_PANDA_HIGH_PD_CONFIG = RobotConfig(
    urdf_path="backend/assets/urdf/franka_panda.urdf",
    robot_type=RobotType.MANIPULATOR,
    initial_joint_pos=FRANKA_PANDA_CONFIG.initial_joint_pos.copy(),

    # Higher PD gains for stiffer control (better for task-space IK)
    actuator_groups={
        "panda_shoulder": ActuatorGroupConfig(
            joint_pattern=r"panda_joint[1-4]",
            max_force=87.0,
            kp=400.0,  # 5x stiffer
            kd=80.0    # 20x damping
        ),
        "panda_forearm": ActuatorGroupConfig(
            joint_pattern=r"panda_joint[5-7]",
            max_force=12.0,
            kp=400.0,
            kd=80.0
        ),
        "panda_hand": ActuatorGroupConfig(
            joint_pattern=r"panda_finger_joint.*",
            max_force=200.0,
            kp=2000.0,
            kd=100.0
        ),
    },

    ee_link_name="panda_hand",
    ee_offset=[0.0, 0.0, 0.107],
    fixed_base=True,
)
"""
Franka Panda with high PD gains for task-space control
- Stiffer response
- Better for differential IK
- Use when end-effector precision is critical
"""


# ============================================================================
# Mobile Robots
# ============================================================================

SIMPLE_MOBILE_ROBOT_CONFIG = RobotConfig(
    urdf_path="backend/assets/urdf/mobile_robot.urdf",  # Placeholder
    robot_type=RobotType.MOBILE,

    # Initial joint positions (wheels at zero)
    initial_joint_pos={
        "wheel_left_joint": 0.0,
        "wheel_right_joint": 0.0,
    },

    # Wheel actuators
    actuator_groups={
        "wheels": ActuatorGroupConfig(
            joint_pattern=r"wheel_.*_joint",
            max_force=50.0,      # Wheel torque (Nm)
            kp=10.0,
            kd=1.0
        ),
    },

    # Locomotion config
    base_link_name="base_link",
    wheel_joint_names=["wheel_left_joint", "wheel_right_joint"],

    # Base configuration
    base_position=[0.0, 0.0, 0.1],  # Start 10cm above ground
    base_orientation=[0.0, 0.0, 0.0],
    fixed_base=False,  # Mobile base can move
    self_collision_enabled=False,

    # Physical properties
    total_mass=10.0,  # 10 kg
    max_linear_velocity=1.0,   # 1 m/s
    max_angular_velocity=1.5,  # ~86 deg/s
)
"""
Simple differential-drive mobile robot
- 2 wheels (left/right)
- Differential drive kinematics
- Best for: Navigation, path following
"""


# ============================================================================
# Quadruped Robots
# ============================================================================

ANYMAL_C_CONFIG = RobotConfig(
    urdf_path="backend/assets/urdf/anymal_c.urdf",  # From Isaac Lab
    robot_type=RobotType.QUADRUPED,

    # Initial joint positions (standing pose)
    # Hip-Abad-Extension (HAA-HFE-KFE) joints per leg
    initial_joint_pos={
        # Left Front (LF)
        "LF_HAA": 0.0,      # Hip abduction/adduction
        "LF_HFE": 0.4,      # Hip flexion/extension
        "LF_KFE": -0.8,     # Knee flexion/extension

        # Right Front (RF)
        "RF_HAA": 0.0,
        "RF_HFE": 0.4,
        "RF_KFE": -0.8,

        # Left Hind (LH)
        "LH_HAA": 0.0,
        "LH_HFE": -0.4,
        "LH_KFE": 0.8,

        # Right Hind (RH)
        "RH_HAA": 0.0,
        "RH_HFE": -0.4,
        "RH_KFE": 0.8,
    },

    # Leg actuators (from Isaac Lab)
    actuator_groups={
        "hip_abduction": ActuatorGroupConfig(
            joint_pattern=r".*_HAA",
            max_force=80.0,
            kp=80.0,
            kd=2.0
        ),
        "hip_flexion": ActuatorGroupConfig(
            joint_pattern=r".*_HFE",
            max_force=80.0,
            kp=80.0,
            kd=2.0
        ),
        "knee": ActuatorGroupConfig(
            joint_pattern=r".*_KFE",
            max_force=80.0,
            kp=80.0,
            kd=2.0
        ),
    },

    # Locomotion config
    base_link_name="base",
    foot_link_names=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],

    # Base configuration
    base_position=[0.0, 0.0, 0.5],  # Start 50cm above ground
    base_orientation=[0.0, 0.0, 0.0],
    fixed_base=False,
    self_collision_enabled=True,

    # Physical properties
    total_mass=50.0,  # 50 kg
    max_linear_velocity=2.0,   # 2 m/s
    max_angular_velocity=2.0,  # ~114 deg/s
)
"""
ANYmal C quadruped robot
- 12 DOF (3 per leg: HAA-HFE-KFE)
- Weight: 50 kg
- Best for: Rough terrain locomotion, research
Reference: https://github.com/isaac-sim/IsaacLab
"""


UNITREE_GO1_CONFIG = RobotConfig(
    urdf_path="backend/assets/urdf/go1.urdf",  # From Isaac Lab
    robot_type=RobotType.QUADRUPED,

    # Initial joint positions (standing pose)
    initial_joint_pos={
        # Front Left
        "FL_hip_joint": 0.0,
        "FL_thigh_joint": 0.9,
        "FL_calf_joint": -1.8,

        # Front Right
        "FR_hip_joint": 0.0,
        "FR_thigh_joint": 0.9,
        "FR_calf_joint": -1.8,

        # Rear Left
        "RL_hip_joint": 0.0,
        "RL_thigh_joint": 0.9,
        "RL_calf_joint": -1.8,

        # Rear Right
        "RR_hip_joint": 0.0,
        "RR_thigh_joint": 0.9,
        "RR_calf_joint": -1.8,
    },

    # Leg actuators
    actuator_groups={
        "hip": ActuatorGroupConfig(
            joint_pattern=r".*_hip_joint",
            max_force=23.7,  # Nm
            kp=20.0,
            kd=0.5
        ),
        "thigh": ActuatorGroupConfig(
            joint_pattern=r".*_thigh_joint",
            max_force=23.7,
            kp=20.0,
            kd=0.5
        ),
        "calf": ActuatorGroupConfig(
            joint_pattern=r".*_calf_joint",
            max_force=35.55,
            kp=20.0,
            kd=0.5
        ),
    },

    # Locomotion config
    base_link_name="trunk",
    foot_link_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],

    # Base configuration
    base_position=[0.0, 0.0, 0.4],
    base_orientation=[0.0, 0.0, 0.0],
    fixed_base=False,

    # Physical properties
    total_mass=12.0,  # 12 kg
    max_linear_velocity=3.0,   # 3 m/s
    max_angular_velocity=2.0,
)
"""
Unitree Go1 quadruped robot
- 12 DOF (3 per leg)
- Weight: 12 kg
- Compact, affordable platform
- Best for: Learning, experiments
"""


# ============================================================================
# Humanoid Robots
# ============================================================================

SIMPLE_HUMANOID_CONFIG = RobotConfig(
    urdf_path="backend/assets/urdf/humanoid.urdf",  # From Isaac Lab
    robot_type=RobotType.HUMANOID,

    # Initial joint positions (T-pose/standing)
    initial_joint_pos={
        # Legs (simplified)
        "left_hip_pitch": 0.0,
        "left_knee": 0.0,
        "left_ankle": 0.0,
        "right_hip_pitch": 0.0,
        "right_knee": 0.0,
        "right_ankle": 0.0,

        # Arms (T-pose)
        "left_shoulder_pitch": 0.0,
        "left_shoulder_roll": 1.57,  # 90 deg (arms out)
        "left_elbow": 0.0,
        "right_shoulder_pitch": 0.0,
        "right_shoulder_roll": -1.57,
        "right_elbow": 0.0,
    },

    # Actuator groups
    actuator_groups={
        "legs": ActuatorGroupConfig(
            joint_pattern=r".*(hip|knee|ankle).*",
            max_force=150.0,
            kp=100.0,
            kd=10.0
        ),
        "arms": ActuatorGroupConfig(
            joint_pattern=r".*(shoulder|elbow).*",
            max_force=50.0,
            kp=50.0,
            kd=5.0
        ),
    },

    # Locomotion config
    base_link_name="torso",
    foot_link_names=["left_foot", "right_foot"],

    # Base configuration
    base_position=[0.0, 0.0, 1.0],  # 1m standing height
    base_orientation=[0.0, 0.0, 0.0],
    fixed_base=False,

    # Physical properties
    total_mass=80.0,  # 80 kg
    max_linear_velocity=1.5,
    max_angular_velocity=1.0,
)
"""
Simple humanoid robot
- Bipedal walker
- Arms for manipulation
- Best for: Walking research, whole-body control
"""


# ============================================================================
# Aerial Robots (Drones)
# ============================================================================

SIMPLE_QUADCOPTER_CONFIG = RobotConfig(
    urdf_path="backend/assets/urdf/quadcopter.urdf",  # Placeholder
    robot_type=RobotType.AERIAL,

    # No joints for simple quadcopter (thrust only)
    initial_joint_pos={},

    # Motor actuators (for propellers if modeled)
    actuator_groups={
        "motors": ActuatorGroupConfig(
            joint_pattern=r"motor_.*",
            max_force=10.0,  # Motor thrust
            kp=1.0,
            kd=0.1
        ),
    },

    # Aerial config
    base_link_name="base_link",

    # Base configuration (start in air)
    base_position=[0.0, 0.0, 2.0],  # 2m above ground
    base_orientation=[0.0, 0.0, 0.0],
    fixed_base=False,

    # Physical properties
    total_mass=1.5,  # 1.5 kg
    max_linear_velocity=10.0,   # 10 m/s
    max_angular_velocity=5.0,   # ~286 deg/s
)
"""
Simple quadcopter drone
- 4 rotors (thrust vectoring)
- Weight: 1.5 kg
- Best for: Waypoint navigation, aerial surveys
"""


# ============================================================================
# Utility Functions
# ============================================================================

def list_available_robots() -> List[str]:
    """
    Get list of all pre-configured robots

    Returns:
        List of robot configuration names
    """
    return [
        # Manipulators
        "FRANKA_PANDA_CONFIG",
        "FRANKA_PANDA_HIGH_PD_CONFIG",

        # Mobile
        "SIMPLE_MOBILE_ROBOT_CONFIG",

        # Quadrupeds
        "ANYMAL_C_CONFIG",
        "UNITREE_GO1_CONFIG",

        # Humanoids
        "SIMPLE_HUMANOID_CONFIG",

        # Aerial
        "SIMPLE_QUADCOPTER_CONFIG",
    ]


def list_robots_by_type(robot_type: RobotType) -> List[str]:
    """
    Get list of robots of a specific type

    Args:
        robot_type: RobotType to filter by

    Returns:
        List of robot configuration names matching type
    """
    all_configs = get_all_robot_configs()
    return [name for name, config in all_configs.items()
            if config.robot_type == robot_type]


def get_all_robot_configs() -> Dict[str, RobotConfig]:
    """
    Get all robot configurations as a dictionary

    Returns:
        Dictionary mapping robot names to RobotConfig objects
    """
    return {
        # Manipulators
        "FRANKA_PANDA_CONFIG": FRANKA_PANDA_CONFIG,
        "FRANKA_PANDA_HIGH_PD_CONFIG": FRANKA_PANDA_HIGH_PD_CONFIG,

        # Mobile
        "SIMPLE_MOBILE_ROBOT_CONFIG": SIMPLE_MOBILE_ROBOT_CONFIG,

        # Quadrupeds
        "ANYMAL_C_CONFIG": ANYMAL_C_CONFIG,
        "UNITREE_GO1_CONFIG": UNITREE_GO1_CONFIG,

        # Humanoids
        "SIMPLE_HUMANOID_CONFIG": SIMPLE_HUMANOID_CONFIG,

        # Aerial
        "SIMPLE_QUADCOPTER_CONFIG": SIMPLE_QUADCOPTER_CONFIG,
    }


def get_robot_config(name: str) -> RobotConfig:
    """
    Get robot configuration by name

    Args:
        name: Robot configuration name (e.g., "FRANKA_PANDA_CONFIG")

    Returns:
        RobotConfig object

    Raises:
        ValueError if robot not found
    """
    configs = get_all_robot_configs()

    if name not in configs:
        raise ValueError(f"Unknown robot: {name}. Available: {list(configs.keys())}")

    return configs[name]


def print_robot_info(config: RobotConfig):
    """
    Print human-readable robot configuration info

    Args:
        config: RobotConfig to display
    """
    print(f"Robot Configuration")
    print(f"=" * 50)
    print(f"Type: {config.robot_type.value.upper()}")
    print(f"URDF Path: {config.urdf_path}")
    print(f"Fixed Base: {config.fixed_base}")

    # Type-specific info
    if config.robot_type == RobotType.MANIPULATOR:
        print(f"End-Effector Link: {config.ee_link_name}")
        print(f"EE Offset: {config.ee_offset}")

    elif config.robot_type in [RobotType.MOBILE, RobotType.QUADRUPED, RobotType.HUMANOID]:
        print(f"Base Link: {config.base_link_name}")
        if config.foot_link_names:
            print(f"Foot Links: {config.foot_link_names}")
        if config.wheel_joint_names:
            print(f"Wheel Joints: {config.wheel_joint_names}")

    if config.total_mass:
        print(f"Total Mass: {config.total_mass} kg")
    print(f"Max Linear Velocity: {config.max_linear_velocity} m/s")
    print(f"Max Angular Velocity: {config.max_angular_velocity} rad/s")

    print(f"\nActuator Groups:")
    for group_name, actuator in config.actuator_groups.items():
        print(f"  {group_name}:")
        print(f"    Pattern: {actuator.joint_pattern}")
        print(f"    Max Force: {actuator.max_force} N")
        print(f"    PD Gains: kp={actuator.kp}, kd={actuator.kd}")

    print(f"\nInitial Joint Positions:")
    for joint, pos in config.initial_joint_pos.items():
        print(f"  {joint}: {pos:.3f} rad")


# ============================================================================
# Template for Adding New Robots
# ============================================================================

"""
To add a new robot:

1. Get the URDF file (see ROBOT_ASSETS_GUIDE.md)

2. Create configuration:

MY_ROBOT_CONFIG = RobotConfig(
    urdf_path="backend/assets/urdf/my_robot.urdf",

    initial_joint_pos={
        "joint1": 0.0,
        "joint2": 1.57,  # 90 degrees
        # ... etc
    },

    actuator_groups={
        "arm": ActuatorGroupConfig(
            joint_pattern=r"joint[1-6]",
            max_force=100.0,
            kp=100.0,
            kd=10.0
        ),
    },

    ee_link_name="end_effector",
    ee_offset=[0.0, 0.0, 0.0],
)

3. Add to list_available_robots() and get_robot_config()

4. Test:
    python -c "from backend.assets.robot_configs import MY_ROBOT_CONFIG; print_robot_info(MY_ROBOT_CONFIG)"
"""


if __name__ == "__main__":
    # Example usage
    print("Available Robots:")
    for robot_name in list_available_robots():
        print(f"  - {robot_name}")

    print("\n" + "="*50 + "\n")
    print_robot_info(FRANKA_PANDA_CONFIG)
