"""
Algorithm Templates for Celestial Studio - Python/Genesis Version

Research-grade algorithm templates for Genesis physics engine.
All templates use Python and Genesis APIs.

References:
- A* + DWA: "Robot obstacle avoidance optimization by A* and DWA fusion algorithm" (PLOS One, April 2024)
- FABRIK IK: "A Combined Inverse Kinematics Algorithm Using FABRIK with Optimization"
- Genesis Documentation: https://github.com/Genesis-Embodied-AI/genesis-doc
"""

from typing import Dict, List


class AlgorithmTemplates:
    """Collection of algorithm templates for different robotics tasks"""

    @staticmethod
    def get_all_templates() -> Dict[str, Dict[str, str]]:
        """Get all algorithm templates organized by category"""
        return {
            "path_planning": AlgorithmTemplates.path_planning_templates(),
            "obstacle_avoidance": AlgorithmTemplates.obstacle_avoidance_templates(),
            "inverse_kinematics": AlgorithmTemplates.inverse_kinematics_templates(),
            "computer_vision": AlgorithmTemplates.computer_vision_templates(),
            "motion_control": AlgorithmTemplates.motion_control_templates(),
        }

    @staticmethod
    def path_planning_templates() -> Dict[str, str]:
        """Path planning algorithm templates (Python/Genesis)"""
        return {
            "astar": """
# A* Path Planning Algorithm (Python/Genesis)
# Based on: "Robot obstacle avoidance optimization by A* and DWA fusion algorithm" (PLOS One, 2024)
#
# Finds optimal path from start to goal using grid-based search
# Time Complexity: O(b^d) where b=branching factor, d=depth
# Space Complexity: O(b^d)

import numpy as np
from typing import List, Tuple, Dict
import heapq

# Configuration parameters
GRID_SIZE = 0.5  # meters per grid cell
HEURISTIC_WEIGHT = 1.0  # A* heuristic weight (1.0 = optimal, >1.0 = faster)

def find_path(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Dict],
    robot_state: Dict
) -> List[np.ndarray]:
    \"\"\"
    Find path from start to goal using A* algorithm

    Args:
        start: Start position [x, y]
        goal: Goal position [x, y]
        obstacles: List of obstacles with 'position' and 'radius'
        robot_state: Current robot state

    Returns:
        List of waypoints from start to goal
    \"\"\"
    # Create grid
    grid_size = GRID_SIZE
    bounds = robot_state.get('bounds', {'min': [-10, -10], 'max': [10, 10]})

    # Convert continuous space to grid
    def world_to_grid(pos):
        return (
            int((pos[0] - bounds['min'][0]) / grid_size),
            int((pos[1] - bounds['min'][1]) / grid_size)
        )

    def grid_to_world(grid_pos):
        return np.array([
            grid_pos[0] * grid_size + bounds['min'][0],
            grid_pos[1] * grid_size + bounds['min'][1]
        ])

    # Check if cell is walkable
    def is_walkable(grid_pos):
        world_pos = grid_to_world(grid_pos)
        for obs in obstacles:
            obs_pos = np.array(obs['position'][:2])
            if np.linalg.norm(world_pos - obs_pos) < obs['radius'] + 0.3:
                return False
        return True

    # Heuristic (Euclidean distance)
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b)) * HEURISTIC_WEIGHT

    # A* algorithm
    start_grid = world_to_grid(start)
    goal_grid = world_to_grid(goal)

    open_set = []
    heapq.heappush(open_set, (0, start_grid))
    came_from = {}
    g_score = {start_grid: 0}
    f_score = {start_grid: heuristic(start_grid, goal_grid)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal_grid:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(grid_to_world(current))
                current = came_from[current]
            path.append(grid_to_world(start_grid))
            return path[::-1]

        # Check neighbors (8-connected)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not is_walkable(neighbor):
                continue

            tentative_g = g_score[current] + np.sqrt(dx**2 + dy**2)

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal_grid)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # No path found
    return [start, goal]
""",

            "rrt": """
# RRT (Rapidly-exploring Random Tree) Path Planning
# Python/Genesis implementation

import numpy as np
from typing import List, Dict

# Configuration
MAX_ITERATIONS = 1000
STEP_SIZE = 0.5  # meters
GOAL_THRESHOLD = 0.5  # meters

def find_path(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Dict],
    robot_state: Dict
) -> List[np.ndarray]:
    \"\"\"
    Find path using RRT algorithm

    Args:
        start: Start position [x, y]
        goal: Goal position [x, y]
        obstacles: List of obstacles
        robot_state: Robot state

    Returns:
        List of waypoints
    \"\"\"
    class Node:
        def __init__(self, pos):
            self.pos = np.array(pos)
            self.parent = None

    def is_collision_free(pos):
        for obs in obstacles:
            obs_pos = np.array(obs['position'][:2])
            if np.linalg.norm(pos - obs_pos) < obs['radius'] + 0.3:
                return False
        return True

    def nearest_node(tree, pos):
        return min(tree, key=lambda n: np.linalg.norm(n.pos - pos))

    def steer(from_pos, to_pos, step_size):
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        if dist < step_size:
            return to_pos
        return from_pos + direction / dist * step_size

    # Initialize tree
    tree = [Node(start)]
    bounds = robot_state.get('bounds', {'min': [-10, -10], 'max': [10, 10]})

    for i in range(MAX_ITERATIONS):
        # Sample random point (bias towards goal)
        if np.random.rand() < 0.1:
            rand_pos = goal
        else:
            rand_pos = np.array([
                np.random.uniform(bounds['min'][0], bounds['max'][0]),
                np.random.uniform(bounds['min'][1], bounds['max'][1])
            ])

        # Find nearest node
        nearest = nearest_node(tree, rand_pos)

        # Steer towards random point
        new_pos = steer(nearest.pos, rand_pos, STEP_SIZE)

        # Check if collision-free
        if is_collision_free(new_pos):
            new_node = Node(new_pos)
            new_node.parent = nearest
            tree.append(new_node)

            # Check if goal reached
            if np.linalg.norm(new_pos - goal) < GOAL_THRESHOLD:
                # Reconstruct path
                path = []
                node = new_node
                while node is not None:
                    path.append(node.pos)
                    node = node.parent
                return path[::-1]

    # No path found, return straight line
    return [start, goal]
"""
        }

    @staticmethod
    def obstacle_avoidance_templates() -> Dict[str, str]:
        """Obstacle avoidance algorithm templates (Python/Genesis)"""
        return {
            "dwa": """
# Dynamic Window Approach (DWA) Obstacle Avoidance
# Python/Genesis implementation

import numpy as np
from typing import List, Dict, Tuple

# Configuration
MAX_SPEED = 2.0  # m/s
MAX_ANGULAR_SPEED = 1.0  # rad/s
MAX_ACCEL = 0.5  # m/s^2
MAX_ANGULAR_ACCEL = 1.0  # rad/s^2
DT = 0.1  # Time step
PREDICT_TIME = 2.0  # Prediction horizon
ROBOT_RADIUS = 0.3  # meters

def compute_safe_velocity(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Dict],
    robot_state: Dict
) -> np.ndarray:
    \"\"\"
    Compute safe velocity using DWA

    Args:
        start: Current position [x, y]
        goal: Goal position [x, y]
        obstacles: List of obstacles
        robot_state: Current robot state with 'velocity' and 'angular_velocity'

    Returns:
        Safe velocity command [vx, vy, omega]
    \"\"\"
    current_vel = robot_state.get('velocity', np.zeros(3))
    current_omega = robot_state.get('angular_velocity', np.zeros(3))

    v = current_vel[0]  # Current linear velocity
    omega = current_omega[2]  # Current angular velocity

    # Dynamic window (achievable velocities)
    v_min = max(0, v - MAX_ACCEL * DT)
    v_max = min(MAX_SPEED, v + MAX_ACCEL * DT)
    omega_min = max(-MAX_ANGULAR_SPEED, omega - MAX_ANGULAR_ACCEL * DT)
    omega_max = min(MAX_ANGULAR_SPEED, omega + MAX_ANGULAR_ACCEL * DT)

    best_score = -float('inf')
    best_v, best_omega = v, omega

    # Sample velocity space
    for test_v in np.linspace(v_min, v_max, 10):
        for test_omega in np.linspace(omega_min, omega_max, 10):
            # Simulate trajectory
            trajectory = simulate_trajectory(start, test_v, test_omega, PREDICT_TIME)

            # Check collision
            if not is_trajectory_safe(trajectory, obstacles):
                continue

            # Score trajectory
            score = score_trajectory(trajectory, goal, test_v, test_omega)

            if score > best_score:
                best_score = score
                best_v = test_v
                best_omega = test_omega

    return np.array([best_v, 0, best_omega])

def simulate_trajectory(start, v, omega, time_horizon):
    \"\"\"Simulate robot trajectory\"\"\"
    trajectory = [start[:2].copy()]
    pos = start[:2].copy()
    theta = 0.0

    steps = int(time_horizon / DT)
    for _ in range(steps):
        pos[0] += v * np.cos(theta) * DT
        pos[1] += v * np.sin(theta) * DT
        theta += omega * DT
        trajectory.append(pos.copy())

    return np.array(trajectory)

def is_trajectory_safe(trajectory, obstacles):
    \"\"\"Check if trajectory collides with obstacles\"\"\"
    for pos in trajectory:
        for obs in obstacles:
            obs_pos = np.array(obs['position'][:2])
            if np.linalg.norm(pos - obs_pos) < ROBOT_RADIUS + obs['radius']:
                return False
    return True

def score_trajectory(trajectory, goal, v, omega):
    \"\"\"Score trajectory based on goal proximity and smoothness\"\"\"
    # Distance to goal
    goal_dist = np.linalg.norm(trajectory[-1] - goal[:2])
    goal_score = 1.0 / (1.0 + goal_dist)

    # Velocity score (prefer higher velocities)
    vel_score = v / MAX_SPEED

    # Smoothness score (prefer lower angular velocity)
    smooth_score = 1.0 - abs(omega) / MAX_ANGULAR_SPEED

    # Combined score
    return 2.0 * goal_score + 1.0 * vel_score + 0.5 * smooth_score
"""
        }

    @staticmethod
    def inverse_kinematics_templates() -> Dict[str, str]:
        """Inverse kinematics algorithm templates (Python/Genesis)"""
        return {
            "fabrik": """
# FABRIK (Forward And Backward Reaching Inverse Kinematics)
# Python/Genesis implementation

import numpy as np
from typing import List, Dict

# Configuration
MAX_ITERATIONS = 100
TOLERANCE = 0.01  # meters
LINK_LENGTHS = [0.3, 0.25, 0.2, 0.15]  # Example link lengths

def solve_ik(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Dict],
    robot_state: Dict
) -> np.ndarray:
    \"\"\"
    Solve inverse kinematics using FABRIK

    Args:
        start: Base position [x, y, z]
        goal: Target end-effector position [x, y, z]
        obstacles: Not used for IK
        robot_state: Current joint angles

    Returns:
        Joint angles to reach target
    \"\"\"
    # Get current joint positions
    link_lengths = robot_state.get('link_lengths', LINK_LENGTHS)
    n_joints = len(link_lengths)

    # Initialize joint positions
    base = start.copy()
    target = goal.copy()

    # Initialize joints along straight line from base to target
    direction = target - base
    total_length = np.sum(link_lengths)
    dist = np.linalg.norm(direction)

    # Check if target is reachable
    if dist > total_length:
        target = base + direction / dist * total_length * 0.95

    # Initialize joint positions
    joints = [base.copy()]
    for i, length in enumerate(link_lengths):
        ratio = (i + 1) / n_joints
        joints.append(base + direction * ratio)

    # FABRIK algorithm
    for iteration in range(MAX_ITERATIONS):
        # Forward reaching
        joints[-1] = target.copy()
        for i in range(n_joints - 1, 0, -1):
            direction = joints[i] - joints[i + 1]
            joints[i] = joints[i + 1] + direction / np.linalg.norm(direction) * link_lengths[i]

        # Backward reaching
        joints[0] = base.copy()
        for i in range(n_joints):
            direction = joints[i + 1] - joints[i]
            dist = np.linalg.norm(direction)
            if dist > 0:
                joints[i + 1] = joints[i] + direction / dist * link_lengths[i]

        # Check convergence
        if np.linalg.norm(joints[-1] - target) < TOLERANCE:
            break

    # Convert joint positions to angles
    angles = []
    for i in range(n_joints):
        vec1 = joints[i + 1] - joints[i]
        angle = np.arctan2(vec1[1], vec1[0])
        angles.append(angle)

    return np.array(angles)
"""
        }

    @staticmethod
    def computer_vision_templates() -> Dict[str, str]:
        """Computer vision algorithm templates (Python/Genesis)"""
        return {
            "object_detection": """
# Object Detection using Genesis Camera
# Python/Genesis implementation

import numpy as np
from typing import List, Dict

def process_vision(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Dict],
    robot_state: Dict
) -> Dict:
    \"\"\"
    Process vision data from Genesis camera

    Args:
        start: Camera position
        goal: Not used
        obstacles: Scene obstacles
        robot_state: Contains camera data if available

    Returns:
        Dictionary with detected objects
    \"\"\"
    # Get camera data from robot state
    rgb_frame = robot_state.get('camera_rgb', None)
    depth_frame = robot_state.get('camera_depth', None)

    detected_objects = []

    if rgb_frame is not None:
        # Simple color-based detection (placeholder)
        # In real implementation, would use YOLO, etc.
        height, width = rgb_frame.shape[:2]

        # Detect red objects (example)
        red_mask = (rgb_frame[:, :, 0] > 150) & (rgb_frame[:, :, 1] < 100) & (rgb_frame[:, :, 2] < 100)
        red_pixels = np.argwhere(red_mask)

        if len(red_pixels) > 0:
            # Get bounding box
            y_min, x_min = red_pixels.min(axis=0)
            y_max, x_max = red_pixels.max(axis=0)

            detected_objects.append({
                'class': 'red_object',
                'bbox': [x_min, y_min, x_max, y_max],
                'confidence': 0.85
            })

    if depth_frame is not None:
        # Estimate distance to objects
        for obj in detected_objects:
            x_center = (obj['bbox'][0] + obj['bbox'][2]) // 2
            y_center = (obj['bbox'][1] + obj['bbox'][3]) // 2
            obj['distance'] = float(depth_frame[y_center, x_center])

    return {
        'objects': detected_objects,
        'frame_width': rgb_frame.shape[1] if rgb_frame is not None else 0,
        'frame_height': rgb_frame.shape[0] if rgb_frame is not None else 0
    }
"""
        }

    @staticmethod
    def motion_control_templates() -> Dict[str, str]:
        """Motion control algorithm templates (Python/Genesis)"""
        return {
            "pd_controller": """
# PD Controller for Robot Control
# Python/Genesis implementation

import numpy as np
from typing import List, Dict

# Configuration
KP = 10.0  # Proportional gain
KD = 2.0   # Derivative gain

def compute_control(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Dict],
    robot_state: Dict
) -> np.ndarray:
    \"\"\"
    Compute control command using PD controller

    Args:
        start: Current position [x, y]
        goal: Target position [x, y]
        obstacles: Not used
        robot_state: Current velocity and acceleration

    Returns:
        Control command [vx, vy, omega]
    \"\"\"
    # Position error
    error = goal[:2] - start[:2]

    # Velocity (derivative of position)
    current_vel = robot_state.get('velocity', np.zeros(3))[:2]

    # PD control
    control = KP * error - KD * current_vel

    # Limit maximum velocity
    max_vel = 2.0
    vel_magnitude = np.linalg.norm(control)
    if vel_magnitude > max_vel:
        control = control / vel_magnitude * max_vel

    # Compute angular velocity (turn towards goal)
    current_heading = robot_state.get('heading', 0.0)
    desired_heading = np.arctan2(error[1], error[0])
    heading_error = desired_heading - current_heading

    # Normalize angle to [-pi, pi]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    omega = KP * heading_error

    return np.array([control[0], control[1], omega])
"""
        }


def get_algorithm_list() -> List[Dict]:
    """Get list of all available algorithms with metadata"""
    algorithms = []

    templates = AlgorithmTemplates.get_all_templates()

    for category, algorithms_dict in templates.items():
        for algo_name in algorithms_dict.keys():
            algorithms.append({
                "id": f"{category}_{algo_name}",
                "name": algo_name.upper().replace("_", " "),
                "category": category.replace("_", " ").title(),
                "description": f"{algo_name} algorithm for {category}",
                "language": "python",
                "framework": "genesis"
            })

    return algorithms
