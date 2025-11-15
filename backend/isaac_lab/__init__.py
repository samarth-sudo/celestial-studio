"""
Isaac Lab Integration Module

Cloud-based robotics simulation using NVIDIA Isaac Lab on Modal infrastructure.
Provides GPU-accelerated physics simulation, RL training, and video streaming.
"""

from .scene_converter import (
    SceneConverter,
    MultiRobotSceneConverter,
    convert_scene_to_isaac,
    suggest_training_task
)

from .task_mapper import (
    TaskMapper,
    IsaacLabTask,
    get_task_mapper
)

__all__ = [
    'SceneConverter',
    'MultiRobotSceneConverter',
    'convert_scene_to_isaac',
    'suggest_training_task',
    'TaskMapper',
    'IsaacLabTask',
    'get_task_mapper',
    'modal_runner',  # Modal functions accessed via module import
    'webrtc_server'  # WebRTC server module
]
