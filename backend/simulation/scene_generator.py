import random
from typing import Dict, List


class SceneGenerator:
    """Generates 3D scene configurations from user requirements"""

    def generate_from_requirements(self, requirements: Dict) -> Dict:
        """
        Transform requirements into Three.js scene configuration

        Args:
            requirements: Dict with robot_type, task, environment, objects, etc.

        Returns:
            Complete scene config ready for frontend rendering
        """

        robot_type = requirements.get('robot_type', 'mobile')
        environment = requirements.get('environment', 'warehouse')
        task = requirements.get('task', 'navigate')
        objects_list = requirements.get('objects', ['boxes'])

        scene_config = {
            "environment": self._create_environment(environment),
            "robot": self._create_robot(robot_type, task),
            "objects": [],
            "lighting": self._create_lighting(environment),
            "camera": {
                "position": [15, 15, 15],
                "lookAt": [0, 0, 0],
                "fov": 50
            }
        }

        # Add objects based on requirements
        for obj_type in objects_list:
            scene_config['objects'].extend(
                self._create_objects(obj_type, environment)
            )

        # Add task-specific elements
        if 'pick' in task.lower():
            scene_config['task_markers'] = self._create_pick_targets()

        if 'navigate' in task.lower() or 'move' in task.lower():
            scene_config['waypoints'] = self._create_waypoints()

        return scene_config

    def _create_environment(self, env_type: str) -> Dict:
        """Create environment-specific floor and walls"""

        environments = {
            "warehouse": {
                "floor": {
                    "size": [20, 20],
                    "color": "#808080",
                    "texture": "concrete",
                    "gridLines": True
                },
                "walls": True,
                "wallHeight": 5,
                "wallColor": "#A0A0A0"
            },
            "office": {
                "floor": {
                    "size": [15, 15],
                    "color": "#C0C0C0",
                    "texture": "carpet",
                    "gridLines": False
                },
                "walls": True,
                "wallHeight": 3,
                "wallColor": "#D0D0D0"
            },
            "outdoor": {
                "floor": {
                    "size": [30, 30],
                    "color": "#228B22",
                    "texture": "grass",
                    "gridLines": False
                },
                "walls": False,
                "wallHeight": 0,
                "skybox": True
            },
            "factory": {
                "floor": {
                    "size": [25, 25],
                    "color": "#696969",
                    "texture": "metal",
                    "gridLines": True
                },
                "walls": True,
                "wallHeight": 6,
                "wallColor": "#808080"
            }
        }

        return environments.get(env_type, environments["warehouse"])

    def _create_robot(self, robot_type: str, task: str) -> Dict:
        """Create robot configuration"""

        robots = {
            "mobile": {
                "type": "mobile_robot",
                "model": "differential_drive",
                "dimensions": [0.4, 0.3, 0.4],  # width, height, depth
                "color": "#4169E1",
                "position": [0, 0.15, 0],
                "wheels": {
                    "radius": 0.1,
                    "width": 0.05,
                    "color": "#000000"
                },
                "hasGripper": 'pick' in task.lower()
            },
            "arm": {
                "type": "robotic_arm",
                "dof": 6,
                "segments": [
                    {"length": 0.5, "color": "#FF6347"},
                    {"length": 0.4, "color": "#FF7F50"},
                    {"length": 0.3, "color": "#FFA07A"}
                ],
                "base_position": [0, 0, 0],
                "end_effector": "gripper"
            },
            "drone": {
                "type": "quadcopter",
                "body_size": 0.3,
                "propeller_count": 4,
                "color": "#00CED1",
                "position": [0, 2, 0]
            },
            "humanoid": {
                "type": "humanoid",
                "height": 1.7,
                "color": "#FFD700",
                "position": [0, 0.85, 0]
            }
        }

        return robots.get(robot_type, robots["mobile"])

    def _create_objects(self, obj_type: str, environment: str) -> List[Dict]:
        """Create objects appropriate for environment"""

        # Adjust counts based on environment
        if environment == "warehouse":
            box_count = 10  # More boxes for warehouse
            shelf_count = 4  # More shelves for warehouse
            pallet_count = 4  # More pallets for warehouse
        else:
            box_count = 5
            shelf_count = 3
            pallet_count = 3

        if obj_type == "boxes" or obj_type == "box":
            objects = self._create_boxes(count=box_count)
            # Add shelving units and pallets for warehouse environment
            if environment == "warehouse":
                objects.extend(self._create_shelves(count=shelf_count))
                objects.extend(self._create_pallets(count=pallet_count))
            return objects
        elif obj_type == "shelves" or obj_type == "shelf":
            return self._create_shelves(count=shelf_count)
        elif obj_type == "obstacles" or obj_type == "obstacle":
            return self._create_obstacles(count=4)
        elif obj_type == "people" or obj_type == "person":
            return self._create_people(count=2)
        elif obj_type == "pallets" or obj_type == "pallet":
            return self._create_pallets(count=pallet_count)
        else:
            return []

    def _create_boxes(self, count: int) -> List[Dict]:
        """Generate box objects"""
        boxes = []
        for i in range(count):
            position = self._random_position(min_dist=2)
            boxes.append({
                "type": "box",
                "id": f"box_{i}",
                "position": position,
                "size": [0.3, 0.3, 0.3],
                "color": self._random_box_color(),
                "physics": {
                    "mass": 1.0,
                    "friction": 0.5,
                    "restitution": 0.3
                },
                "interactive": True
            })
        return boxes

    def _create_obstacles(self, count: int) -> List[Dict]:
        """Generate obstacle objects (cylinders and boxes)"""
        obstacles = []
        for i in range(count):
            position = self._random_position(min_dist=3)
            is_cylinder = i % 2 == 0

            if is_cylinder:
                obstacles.append({
                    "type": "cylinder",
                    "id": f"obstacle_{i}",
                    "position": position,
                    "size": [0.3, 1.5, 0.3],  # radius, height, radius
                    "color": "#FF4500",
                    "physics": {"mass": 10, "friction": 0.7, "static": True}
                })
            else:
                obstacles.append({
                    "type": "box",
                    "id": f"obstacle_{i}",
                    "position": position,
                    "size": [0.8, 1.2, 0.8],
                    "color": "#FF6347",
                    "physics": {"mass": 10, "friction": 0.7, "static": True}
                })
        return obstacles

    def _create_shelves(self, count: int) -> List[Dict]:
        """Generate shelf units"""
        shelves = []
        for i in range(count):
            # Place shelves along edges
            if i == 0:
                position = [8, 1.5, 0]
            elif i == 1:
                position = [-8, 1.5, 0]
            else:
                position = [0, 1.5, 8]

            shelves.append({
                "type": "shelf",
                "id": f"shelf_{i}",
                "position": position,
                "size": [2, 3, 0.5],  # width, height, depth
                "color": "#8B4513",
                "shelves": 3,  # number of shelf levels
                "physics": {"mass": 50, "friction": 0.8, "static": True}
            })
        return shelves

    def _create_people(self, count: int) -> List[Dict]:
        """Generate simple human representations (cylinders)"""
        people = []
        for i in range(count):
            position = self._random_position(min_dist=4)
            position[1] = 0.9  # Half of height

            people.append({
                "type": "person",
                "id": f"person_{i}",
                "position": position,
                "height": 1.8,
                "radius": 0.3,
                "color": "#4169E1",
                "physics": {"mass": 70, "friction": 0.6, "static": True}
            })
        return people

    def _create_pallets(self, count: int) -> List[Dict]:
        """Generate warehouse pallets"""
        pallets = []
        for i in range(count):
            position = self._random_position(min_dist=2.5)
            position[1] = 0.075  # Half height

            pallets.append({
                "type": "pallet",
                "id": f"pallet_{i}",
                "position": position,
                "size": [1.2, 0.15, 0.8],
                "color": "#CD853F",
                "physics": {"mass": 15, "friction": 0.7, "static": True}
            })
        return pallets

    def _create_pick_targets(self) -> List[Dict]:
        """Create target markers for pick-and-place tasks"""
        return [
            {
                "type": "target_zone",
                "id": "drop_zone",
                "position": [5, 0.05, 5],
                "size": [1.5, 0.1, 1.5],
                "color": "#00FF00",
                "opacity": 0.5
            }
        ]

    def _create_waypoints(self) -> List[Dict]:
        """Create waypoints for navigation tasks"""
        waypoints = []
        positions = [[3, 0.2, 3], [3, 0.2, -3], [-3, 0.2, -3], [-3, 0.2, 3]]

        for i, pos in enumerate(positions):
            waypoints.append({
                "type": "waypoint",
                "id": f"waypoint_{i}",
                "position": pos,
                "radius": 0.3,
                "color": "#FFD700",
                "number": i + 1
            })
        return waypoints

    def _create_lighting(self, environment: str) -> Dict:
        """Create lighting configuration"""
        return {
            "ambient": {
                "intensity": 0.5,
                "color": "#FFFFFF"
            },
            "directional": {
                "position": [10, 10, 5],
                "intensity": 1.0,
                "color": "#FFFFFF",
                "castShadow": True
            },
            "hemisphere": environment == "outdoor"
        }

    def _random_position(self, min_dist: float = 2) -> List[float]:
        """Generate random position avoiding center"""
        while True:
            x = random.uniform(-8, 8)
            z = random.uniform(-8, 8)

            # Avoid placing too close to center (robot start position)
            if abs(x) > min_dist or abs(z) > min_dist:
                return [x, 0, z]

    def _random_box_color(self) -> str:
        """Get random box color from common warehouse colors"""
        colors = ["#8B4513", "#A0522D", "#D2691E", "#CD853F", "#DEB887"]
        return random.choice(colors)
