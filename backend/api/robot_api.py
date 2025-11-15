"""
Multi-Robot API Endpoints

FastAPI endpoints for controlling different robot types and running tasks.

Endpoints:
    - GET /api/robots/list - List all available robots
    - GET /api/robots/{robot_name}/info - Get robot configuration info
    - POST /api/robots/{robot_name}/simulate - Run simulation for a robot
    - GET /api/tasks/list - List all available tasks
    - POST /api/tasks/{task_name}/run - Run a specific task

Usage:
    # List robots
    GET /api/robots/list
    Response: {
        "robots": [
            {"name": "FRANKA_PANDA_CONFIG", "type": "manipulator", ...},
            ...
        ]
    }

    # Run simulation
    POST /api/robots/FRANKA_PANDA_CONFIG/simulate
    Body: {
        "task": "reach",
        "params": {"target_pos": [0.5, 0, 0.3]}
    }
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import numpy as np

try:
    from assets.robot_configs import (
        get_all_robot_configs,
        list_available_robots,
        list_robots_by_type,
        RobotType,
        RobotConfig
    )
except ImportError:
    from backend.assets.robot_configs import (
        get_all_robot_configs,
        list_available_robots,
        list_robots_by_type,
        RobotType,
        RobotConfig
    )

router = APIRouter(prefix="/api/robots", tags=["robots"])


# ============================================================================
# Request/Response Models
# ============================================================================

class RobotInfo(BaseModel):
    """Robot information model"""
    name: str
    type: str
    urdf_path: str
    fixed_base: bool
    num_actuator_groups: int
    ee_link_name: Optional[str] = None
    base_link_name: Optional[str] = None
    total_mass: Optional[float] = None
    max_linear_velocity: float
    max_angular_velocity: float


class SimulationRequest(BaseModel):
    """Simulation request model"""
    task_type: str  # "reach", "lift", "navigation", "walking", "flight"
    duration: float = 10.0  # seconds
    use_gui: bool = False
    params: Dict[str, Any] = {}


class SimulationResponse(BaseModel):
    """Simulation response model"""
    success: bool
    robot_name: str
    task_type: str
    steps: int
    total_reward: float
    final_state: Dict[str, Any]
    info: Dict[str, Any]


# ============================================================================
# Robot Endpoints
# ============================================================================

@router.get("/list", response_model=List[str])
async def list_robots():
    """
    Get list of all available robots

    Returns:
        List of robot configuration names
    """
    return list_available_robots()


@router.get("/list/{robot_type}", response_model=List[str])
async def list_robots_of_type(robot_type: str):
    """
    Get list of robots of a specific type

    Args:
        robot_type: Type of robot (manipulator, mobile, quadruped, humanoid, aerial)

    Returns:
        List of robot names matching the type
    """
    try:
        # Convert string to RobotType enum
        robot_type_enum = RobotType(robot_type.lower())
        return list_robots_by_type(robot_type_enum)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid robot type: {robot_type}. Valid types: manipulator, mobile, quadruped, humanoid, aerial"
        )


@router.get("/{robot_name}/info", response_model=RobotInfo)
async def get_robot_info(robot_name: str):
    """
    Get detailed information about a specific robot

    Args:
        robot_name: Name of robot configuration

    Returns:
        Robot configuration details
    """
    all_configs = get_all_robot_configs()

    if robot_name not in all_configs:
        raise HTTPException(
            status_code=404,
            detail=f"Robot not found: {robot_name}"
        )

    config = all_configs[robot_name]

    return RobotInfo(
        name=robot_name,
        type=config.robot_type.value,
        urdf_path=config.urdf_path,
        fixed_base=config.fixed_base,
        num_actuator_groups=len(config.actuator_groups),
        ee_link_name=config.ee_link_name,
        base_link_name=config.base_link_name,
        total_mass=config.total_mass,
        max_linear_velocity=config.max_linear_velocity,
        max_angular_velocity=config.max_angular_velocity
    )


@router.post("/{robot_name}/simulate", response_model=SimulationResponse)
async def simulate_robot(robot_name: str, request: SimulationRequest):
    """
    Run a simulation for a specific robot and task

    Args:
        robot_name: Name of robot configuration
        request: Simulation parameters

    Returns:
        Simulation results
    """
    # Get robot config
    all_configs = get_all_robot_configs()
    if robot_name not in all_configs:
        raise HTTPException(
            status_code=404,
            detail=f"Robot not found: {robot_name}"
        )

    config = all_configs[robot_name]

    # This is a placeholder - actual implementation would:
    # 1. Create PyBulletRobotInterface with config
    # 2. Create appropriate task based on task_type
    # 3. Create appropriate controller
    # 4. Run simulation loop
    # 5. Return results

    # For now, return mock response
    return SimulationResponse(
        success=True,
        robot_name=robot_name,
        task_type=request.task_type,
        steps=int(request.duration * 240),  # 240 Hz
        total_reward=100.0,
        final_state={
            "position": [0.5, 0.0, 0.3],
            "orientation": [1.0, 0.0, 0.0, 0.0]
        },
        info={
            "duration": request.duration,
            "robot_type": config.robot_type.value,
            "task_completed": True
        }
    )


# ============================================================================
# Task Endpoints
# ============================================================================

@router.get("/tasks/list", response_model=List[str])
async def list_tasks():
    """
    Get list of all available tasks

    Returns:
        List of task names
    """
    return [
        "reach",       # Manipulator: reach target pose
        "lift",        # Manipulator: pick and place
        "navigation",  # Mobile/Legged: waypoint following
        "walking",     # Legged: forward locomotion
        "flight",      # Aerial: 3D waypoint navigation
    ]


@router.get("/tasks/{robot_type}/compatible", response_model=List[str])
async def get_compatible_tasks(robot_type: str):
    """
    Get tasks compatible with a robot type

    Args:
        robot_type: Type of robot

    Returns:
        List of compatible task names
    """
    task_map = {
        "manipulator": ["reach", "lift"],
        "mobile": ["navigation"],
        "quadruped": ["navigation", "walking"],
        "humanoid": ["walking"],
        "aerial": ["flight"],
        "mobile_manipulator": ["navigation", "reach", "lift"]
    }

    if robot_type.lower() not in task_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid robot type: {robot_type}"
        )

    return task_map[robot_type.lower()]


# ============================================================================
# Controller Endpoints
# ============================================================================

@router.get("/controllers/list", response_model=List[Dict[str, str]])
async def list_controllers():
    """
    Get list of all available controllers

    Returns:
        List of controllers with their types and descriptions
    """
    return [
        {
            "name": "differential_ik",
            "type": "manipulator",
            "description": "Inverse kinematics for manipulator arms"
        },
        {
            "name": "mobile",
            "type": "mobile",
            "description": "Differential drive control for wheeled robots"
        },
        {
            "name": "quadruped",
            "type": "quadruped",
            "description": "Gait generation for 4-legged robots"
        },
        {
            "name": "drone",
            "type": "aerial",
            "description": "Flight control for quadcopters"
        }
    ]


# ============================================================================
# Example/Demo Endpoints
# ============================================================================

@router.get("/examples/{robot_type}", response_model=Dict[str, Any])
async def get_robot_examples(robot_type: str):
    """
    Get example code snippets for a robot type

    Args:
        robot_type: Type of robot

    Returns:
        Example code and usage instructions
    """
    examples = {
        "manipulator": {
            "description": "Franka Panda arm with IK control",
            "example_code": """
from backend.assets.robot_configs import FRANKA_PANDA_CONFIG
from backend.simulation.pybullet_interface import PyBulletRobotInterface
from backend.controllers import DifferentialIKController
from backend.tasks import ReachTask

# Create robot
robot = PyBulletRobotInterface(FRANKA_PANDA_CONFIG, use_gui=True)

# Create controller and task
controller = DifferentialIKController()
task = ReachTask(robot, target_pos=[0.5, 0, 0.3])

# Control loop
obs = task.reset()
while not done:
    joint_targets = controller.compute_joint_targets(robot, task.target_pose)
    obs, reward, done, info = task.step(joint_targets)
            """,
            "compatible_tasks": ["reach", "lift"]
        },
        "mobile": {
            "description": "Differential drive mobile robot",
            "example_code": """
from backend.controllers import MobileController
from backend.tasks import NavigationTask

controller = MobileController()
waypoints = [[1, 0], [1, 1], [0, 1]]
task = NavigationTask(robot, waypoints=waypoints)

# Navigation
obs = task.reset()
while not done:
    pos, yaw = obs["position"], obs["yaw"]
    target = obs["target_waypoint"]
    linear, angular = controller.compute_velocity_to_goal(pos, yaw, target)
    left, right = controller.compute_wheel_velocities(linear, angular)
    obs, reward, done, info = task.step([left, right])
            """,
            "compatible_tasks": ["navigation"]
        },
        "quadruped": {
            "description": "ANYmal quadruped with gait control",
            "example_code": """
from backend.controllers import QuadrupedController, GaitType
from backend.tasks import WalkingTask

controller = QuadrupedController()
task = WalkingTask(robot, target_distance=5.0)

# Walking
obs = task.reset()
time = 0.0
while not done:
    foot_pos = controller.compute_gait(
        gait_type=GaitType.TROT,
        forward_velocity=0.5,
        time=time
    )
    joint_angles = controller.compute_joint_angles(foot_pos)
    # Flatten and apply joint angles
    obs, reward, done, info = task.step(all_angles)
    time += 1/240.0
            """,
            "compatible_tasks": ["navigation", "walking"]
        },
        "aerial": {
            "description": "Quadcopter with flight control",
            "example_code": """
from backend.controllers import DroneController
from backend.tasks import FlightTask

controller = DroneController(mass=1.5)
waypoints = [[1, 0, 2], [1, 1, 2.5], [0, 1, 2]]
task = FlightTask(robot, waypoints=waypoints)

# Flight
obs = task.reset()
while not done:
    pos = obs["position"]
    quat = obs["orientation"]
    vel = obs["linear_velocity"]
    target = obs["target_waypoint"]

    thrust, moments = controller.compute_hover(pos, target, vel, quat)
    obs, reward, done, info = task.step()
            """,
            "compatible_tasks": ["flight"]
        }
    }

    if robot_type.lower() not in examples:
        raise HTTPException(
            status_code=404,
            detail=f"No examples found for robot type: {robot_type}"
        )

    return examples[robot_type.lower()]
