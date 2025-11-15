"""
Multi-Robot Manager

Manages multiple robots in a single simulation scene.
Handles robot instantiation, positioning, and coordination.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class RobotInstance:
    """Single robot instance in the scene"""
    id: str
    name: str
    type: str  # 'urdf_custom', 'mobile_robot', 'robotic_arm', etc.
    scene_config: Dict[str, Any]
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # RPY
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'scene_config': self.scene_config,
            'position': self.position,
            'orientation': self.orientation,
            'active': self.active,
            'created_at': self.created_at,
            'metadata': self.metadata
        }


class MultiRobotManager:
    """Manage multiple robots in a simulation"""

    def __init__(self):
        self.robots: Dict[str, RobotInstance] = {}
        self.scene_id = str(uuid.uuid4())

    def add_robot(self,
                  scene_config: Dict[str, Any],
                  name: Optional[str] = None,
                  position: Optional[List[float]] = None,
                  orientation: Optional[List[float]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> RobotInstance:
        """
        Add a robot to the scene

        Args:
            scene_config: Robot scene configuration (from URDF parser/generator)
            name: Optional custom name for the robot
            position: [x, y, z] position in the scene
            orientation: [roll, pitch, yaw] orientation
            metadata: Additional metadata

        Returns:
            RobotInstance object
        """
        robot_id = str(uuid.uuid4())

        # Extract robot type from scene config
        robot_type = scene_config.get('robot', {}).get('type', 'unknown')
        robot_name = name or scene_config.get('robot', {}).get('name', f'robot_{len(self.robots)}')

        # Create robot instance
        robot = RobotInstance(
            id=robot_id,
            name=robot_name,
            type=robot_type,
            scene_config=scene_config,
            position=position or [0.0, 0.0, 0.0],
            orientation=orientation or [0.0, 0.0, 0.0],
            metadata=metadata or {}
        )

        self.robots[robot_id] = robot

        print(f"Added robot: {robot_name} (ID: {robot_id[:8]}...)")

        return robot

    def remove_robot(self, robot_id: str) -> bool:
        """Remove a robot from the scene"""
        if robot_id in self.robots:
            robot_name = self.robots[robot_id].name
            del self.robots[robot_id]
            print(f"Removed robot: {robot_name} (ID: {robot_id[:8]}...)")
            return True
        return False

    def get_robot(self, robot_id: str) -> Optional[RobotInstance]:
        """Get robot by ID"""
        return self.robots.get(robot_id)

    def list_robots(self) -> List[RobotInstance]:
        """List all robots in the scene"""
        return list(self.robots.values())

    def update_robot_position(self, robot_id: str, position: List[float], orientation: Optional[List[float]] = None) -> bool:
        """Update robot position and orientation"""
        if robot_id in self.robots:
            self.robots[robot_id].position = position
            if orientation:
                self.robots[robot_id].orientation = orientation
            return True
        return False

    def set_robot_active(self, robot_id: str, active: bool) -> bool:
        """Set robot active/inactive state"""
        if robot_id in self.robots:
            self.robots[robot_id].active = active
            return True
        return False

    def get_combined_scene(self) -> Dict[str, Any]:
        """
        Get combined scene configuration for all robots

        Returns a scene config with all robots positioned correctly
        """
        all_robots = []

        for robot in self.robots.values():
            if robot.active:
                robot_data = {
                    'id': robot.id,
                    'name': robot.name,
                    'type': robot.type,
                    'position': robot.position,
                    'orientation': robot.orientation,
                    'config': robot.scene_config
                }
                all_robots.append(robot_data)

        return {
            'scene_id': self.scene_id,
            'robot_count': len(all_robots),
            'robots': all_robots,
            'environment': {
                'floor': {'size': [20, 20]},
                'walls': False,
                'lighting': 'default'
            }
        }

    def clear_scene(self):
        """Remove all robots from the scene"""
        count = len(self.robots)
        self.robots.clear()
        print(f"Cleared scene: removed {count} robots")

    def get_robot_by_name(self, name: str) -> Optional[RobotInstance]:
        """Find robot by name"""
        for robot in self.robots.values():
            if robot.name == name:
                return robot
        return None

    def check_collision(self, robot_id: str, new_position: List[float], safety_distance: float = 1.0) -> bool:
        """
        Check if moving a robot would cause collision with other robots

        Args:
            robot_id: ID of robot to move
            new_position: Proposed new position [x, y, z]
            safety_distance: Minimum safe distance between robots

        Returns:
            True if collision detected, False otherwise
        """
        if robot_id not in self.robots:
            return False

        # Calculate distance to all other robots
        for other_id, other_robot in self.robots.items():
            if other_id == robot_id or not other_robot.active:
                continue

            # Simple Euclidean distance check (2D for ground robots)
            dx = new_position[0] - other_robot.position[0]
            dy = new_position[1] - other_robot.position[1]
            distance = (dx**2 + dy**2) ** 0.5

            if distance < safety_distance:
                return True  # Collision detected

        return False  # No collision

    def suggest_safe_position(self, robot_id: str, preferred_position: List[float],
                            safety_distance: float = 1.0,
                            max_attempts: int = 20) -> Optional[List[float]]:
        """
        Suggest a safe position near the preferred position

        Tries to find an unoccupied position by spiraling out from the preferred location
        """
        import math

        if robot_id not in self.robots:
            return None

        # Check if preferred position is safe
        if not self.check_collision(robot_id, preferred_position, safety_distance):
            return preferred_position

        # Try positions in a spiral pattern
        angle_step = 2 * math.pi / 8  # 8 directions
        radius_step = safety_distance * 0.5

        for attempt in range(max_attempts):
            radius = safety_distance + (attempt // 8) * radius_step
            angle = (attempt % 8) * angle_step

            test_x = preferred_position[0] + radius * math.cos(angle)
            test_y = preferred_position[1] + radius * math.sin(angle)
            test_position = [test_x, test_y, preferred_position[2]]

            if not self.check_collision(robot_id, test_position, safety_distance):
                return test_position

        # Could not find safe position
        return None

    def get_scene_bounds(self) -> Dict[str, List[float]]:
        """Calculate bounding box of all robots"""
        if not self.robots:
            return {'min': [0, 0, 0], 'max': [0, 0, 0]}

        positions = [robot.position for robot in self.robots.values() if robot.active]

        if not positions:
            return {'min': [0, 0, 0], 'max': [0, 0, 0]}

        min_x = min(p[0] for p in positions)
        max_x = max(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] for p in positions)
        min_z = min(p[2] for p in positions)
        max_z = max(p[2] for p in positions)

        return {
            'min': [min_x, min_y, min_z],
            'max': [max_x, max_y, max_z],
            'center': [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
        }


# Global manager instance (in production, use proper state management)
_multi_robot_manager: Optional[MultiRobotManager] = None


def get_multi_robot_manager() -> MultiRobotManager:
    """Get singleton multi-robot manager instance"""
    global _multi_robot_manager
    if _multi_robot_manager is None:
        _multi_robot_manager = MultiRobotManager()
    return _multi_robot_manager


def reset_multi_robot_manager():
    """Reset the multi-robot manager (useful for testing)"""
    global _multi_robot_manager
    _multi_robot_manager = MultiRobotManager()
