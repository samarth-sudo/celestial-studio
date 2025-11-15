"""
Multi-Robot System Module

Manages multiple robots in a single simulation environment.
"""

from .manager import MultiRobotManager, RobotInstance, get_multi_robot_manager, reset_multi_robot_manager

__all__ = [
    'MultiRobotManager',
    'RobotInstance',
    'get_multi_robot_manager',
    'reset_multi_robot_manager'
]
