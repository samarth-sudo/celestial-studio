"""
Robot Controllers

Collection of controllers for different robot types, inspired by Isaac Lab.

Available Controllers:
    - DifferentialIKController: Inverse kinematics for manipulators
    - MobileController: Differential drive control for wheeled robots
    - QuadrupedController: Gait generation for 4-legged robots
    - DroneController: Flight control for quadcopters

Usage:
    from backend.controllers import DifferentialIKController, MobileController
    from backend.controllers import QuadrupedController, DroneController

    # Manipulator
    ik_controller = DifferentialIKController()
    joint_vel = ik_controller.compute(robot, {"pos": [0.5, 0, 0.3]})

    # Mobile robot
    mobile_controller = MobileController()
    left, right = mobile_controller.compute_wheel_velocities(0.5, 0.2)

    # Quadruped
    quad_controller = QuadrupedController()
    foot_pos = quad_controller.compute_gait(GaitType.TROT, forward_velocity=0.5, time=t)

    # Drone
    drone_controller = DroneController(mass=1.5)
    thrust, moments = drone_controller.compute_hover(current_pos, target_pos, vel, quat)
"""

from backend.controllers.differential_ik import (
    DifferentialIKController,
    DifferentialIKConfig,
    solve_ik_position,
    solve_ik_pose
)

from backend.controllers.mobile_controller import (
    MobileController,
    MobileControllerConfig
)

from backend.controllers.quadruped_controller import (
    QuadrupedController,
    QuadrupedControllerConfig,
    GaitType
)

from backend.controllers.drone_controller import (
    DroneController,
    DroneControllerConfig
)

__all__ = [
    # Manipulator controllers
    "DifferentialIKController",
    "DifferentialIKConfig",
    "solve_ik_position",
    "solve_ik_pose",

    # Mobile robot controllers
    "MobileController",
    "MobileControllerConfig",

    # Quadruped controllers
    "QuadrupedController",
    "QuadrupedControllerConfig",
    "GaitType",

    # Drone controllers
    "DroneController",
    "DroneControllerConfig",
]
