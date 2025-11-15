"""
Robot Tasks for All Robot Types

Collection of tasks for robot learning and control,
inspired by Isaac Lab's task framework.

Available Tasks:

    Manipulation (Fixed-base arms):
    - ReachTask: Move end-effector to target pose
    - LiftTask: Pick and lift object to target height

    Locomotion (Mobile/Legged/Aerial):
    - NavigationTask: Waypoint navigation for mobile/legged robots
    - WalkingTask: Forward walking for quadrupeds/humanoids
    - FlightTask: 3D waypoint navigation for drones

Utilities:
    - BaseTask: Abstract base class for custom tasks
    - rewards: Reusable reward functions

Usage:
    # Manipulation
    from backend.tasks import ReachTask, LiftTask
    task = ReachTask(robot, target_pos=[0.5, 0, 0.3])

    # Locomotion
    from backend.tasks import NavigationTask, WalkingTask, FlightTask
    task = NavigationTask(robot, waypoints=[[1, 0], [1, 1], [0, 1]])
    task = WalkingTask(robot, target_distance=5.0)
    task = FlightTask(robot, waypoints=[[1, 0, 2], [1, 1, 2.5]])
"""

from backend.tasks.base_task import BaseTask, TaskConfig, ExampleReachTask

# Manipulation tasks
from backend.tasks.reach_task import ReachTask, ReachTaskConfig
from backend.tasks.lift_task import LiftTask, LiftTaskConfig

# Locomotion tasks
from backend.tasks.navigation_task import NavigationTask, NavigationTaskConfig
from backend.tasks.walking_task import WalkingTask, WalkingTaskConfig
from backend.tasks.flight_task import FlightTask, FlightTaskConfig

__all__ = [
    # Base classes
    "BaseTask",
    "TaskConfig",
    "ExampleReachTask",

    # Manipulation tasks
    "ReachTask",
    "ReachTaskConfig",
    "LiftTask",
    "LiftTaskConfig",

    # Locomotion tasks
    "NavigationTask",
    "NavigationTaskConfig",
    "WalkingTask",
    "WalkingTaskConfig",
    "FlightTask",
    "FlightTaskConfig",
]
