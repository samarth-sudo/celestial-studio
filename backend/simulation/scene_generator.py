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
        robot_dof = requirements.get('robot_dof', None)
        environment = requirements.get('environment', 'warehouse')
        task = requirements.get('task', 'navigate')
        objects_list = requirements.get('objects', ['boxes'])

        scene_config = {
            "environment": self._create_environment(environment),
            "robot": self._create_robot(robot_type, task, robot_dof),
            "objects": [],
            "lighting": self._create_lighting(environment),
            "camera": {
                "position": [15, 15, 15],
                "lookAt": [0, 0, 0],
                "fov": 50
            }
        }

        # Check if we have task-specific objects
        task_objects = requirements.get('task_objects', [])

        if task_objects:
            # Create task-specific objects with exact colors and counts
            scene_config['objects'].extend(
                self._create_task_objects(task_objects, environment)
            )
        else:
            # Add objects based on requirements (old behavior)
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
            "tabletop": {
                "floor": {
                    "size": [3, 3],  # Smaller 3x3m workspace for manipulation
                    "color": "#D2B48C",  # Tan/wood color
                    "texture": "wood",
                    "gridLines": False
                },
                "walls": False,  # No walls for tabletop
                "wallHeight": 0,
                "wallColor": "#FFFFFF"
            },
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

    def _create_robot(self, robot_type: str, task: str, robot_dof: int = None) -> Dict:
        """Create robot configuration"""

        # For arm robots, use custom DOF if provided
        if robot_type == "arm" and robot_dof:
            # Generate segments based on DOF
            segments = []
            segment_colors = ["#FF6347", "#FF7F50", "#FFA07A", "#FFB6C1", "#FF69B4", "#FF1493"]
            for i in range(robot_dof):
                segments.append({
                    "length": 0.5 - (i * 0.05),  # Decreasing length
                    "color": segment_colors[i % len(segment_colors)]
                })

            arm_config = {
                "type": "robotic_arm",
                "dof": robot_dof,
                "segments": segments,
                "base_position": [0, 0, 0],
                "end_effector": "gripper"
            }
            return arm_config

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

    def _create_boxes(self, count: int, color: str = None) -> List[Dict]:
        """Generate box objects"""
        boxes = []
        for i in range(count):
            position = self._random_position(min_dist=2)
            boxes.append({
                "type": "box",
                "id": f"box_{i}",
                "position": position,
                "size": [0.3, 0.3, 0.3],
                "color": color if color else self._random_box_color(),
                "physics": {
                    "mass": 1.0,
                    "friction": 0.5,
                    "restitution": 0.3
                },
                "interactive": True
            })
        return boxes

    def _create_task_objects(self, task_objects: List[Dict], environment: str) -> List[Dict]:
        """Create objects based on task specifications with exact colors and counts"""
        all_objects = []
        object_counter = 0

        # Position objects in a cluster near the robot for easy reach
        # Y = 0.15 (half of cube height 0.3) so cubes rest ON ground, not IN it
        cluster_positions = [
            [2, 0.15, 2], [2, 0.15, -2], [-2, 0.15, 2], [-2, 0.15, -2],
            [3, 0.15, 0], [0, 0.15, 3], [-3, 0.15, 0], [0, 0.15, -3],
            [2.5, 0.15, 2.5], [2.5, 0.15, -2.5], [-2.5, 0.15, 2.5], [-2.5, 0.15, -2.5]
        ]

        position_index = 0

        for obj_spec in task_objects:
            obj_type = obj_spec.get('type', 'box')
            color_spec = obj_spec.get('color', 'mixed')
            count = obj_spec.get('count', 1)
            role = obj_spec.get('role', 'distractor')

            # Handle None count by defaulting to 1
            if count is None or not isinstance(count, int):
                count = 1

            for i in range(count):
                # Get position from cluster
                if position_index < len(cluster_positions):
                    position = cluster_positions[position_index].copy()
                    position_index += 1
                else:
                    position = self._random_position(min_dist=2)

                # Determine color
                if color_spec == 'mixed' or color_spec is None:
                    # For mixed colors or None, use different bright colors for EACH object
                    # Use object_counter to ensure unique colors across all mixed objects
                    color = self._get_task_color(object_counter)
                else:
                    # Use specified color
                    color = self._color_name_to_hex(color_spec)

                # Create object based on type
                obj = {
                    "type": obj_type,
                    "id": f"{obj_type}_{object_counter}",
                    "position": position,
                    "size": [0.3, 0.3, 0.3],
                    "color": color,
                    "physics": {
                        "mass": 1.0,
                        "friction": 0.5,
                        "restitution": 0.3
                    },
                    "interactive": True,
                    "role": role  # Mark as target or distractor for algorithms
                }

                all_objects.append(obj)
                object_counter += 1

        return all_objects

    def _get_task_color(self, index: int) -> str:
        """Get bright, distinct colors for task objects"""
        task_colors = [
            "#FF0000",  # Red
            "#0000FF",  # Blue
            "#00FF00",  # Green
            "#FFFF00",  # Yellow
            "#FF8800",  # Orange
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#8B00FF",  # Purple
            "#FF1493",  # Deep Pink
            "#32CD32",  # Lime Green
            "#FFD700",  # Gold
            "#00CED1"   # Dark Turquoise
        ]
        return task_colors[index % len(task_colors)]

    def _color_name_to_hex(self, color_name: str) -> str:
        """Convert color name to hex code"""
        color_map = {
            "red": "#FF0000",
            "blue": "#0000FF",
            "green": "#00FF00",
            "yellow": "#FFFF00",
            "orange": "#FF8800",
            "purple": "#8B00FF",
            "magenta": "#FF00FF",
            "cyan": "#00FFFF",
            "pink": "#FF1493",
            "lime": "#32CD32",
            "gold": "#FFD700",
            "turquoise": "#00CED1",
            "brown": "#8B4513",
            "white": "#FFFFFF",
            "black": "#000000",
            "gray": "#808080",
            "grey": "#808080"
        }
        # Handle None color_name by returning default color
        if color_name is None:
            return "#FF0000"  # Default to red
        return color_map.get(color_name.lower(), "#FF0000")  # Default to red if unknown

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
                # Y = 0.15 (half of cube height) for ground contact
                return [x, 0.15, z]

    def _random_box_color(self) -> str:
        """Get random box color from common warehouse colors"""
        colors = ["#8B4513", "#A0522D", "#D2691E", "#CD853F", "#DEB887"]
        return random.choice(colors)
