"""
Scene Converter

Converts Celestial Studio JSON scene configurations to Isaac Lab USD format.
Maps robot types, objects, and environments to Isaac Lab assets.
"""

from typing import Dict, Any, List, Optional
import json


class SceneConverter:
    """Convert Celestial Studio scenes to Isaac Lab format"""

    # Map Celestial robot types to Isaac Lab asset paths
    ROBOT_TYPE_MAPPING = {
        'mobile_robot': '/Isaac/Robots/Carter/carter_v1.usd',
        'robotic_arm': '/Isaac/Robots/Franka/franka.usd',
        'drone': '/Isaac/Robots/Crazyflie/cf2x.usd',
        'quadruped': '/Isaac/Robots/ANYbotics/anymal_c.usd',
        'humanoid': '/Isaac/Robots/Humanoid/humanoid.usd',
        'urdf_custom': None  # Will be handled specially
    }

    # Isaac Lab task mapping for common scenarios
    TASK_MAPPING = {
        'navigation': 'Isaac-Velocity-Flat-Anymal-C-v0',
        'manipulation': 'Isaac-Reach-Franka-v0',
        'pick_and_place': 'Isaac-Lift-Cube-Franka-v0',
        'warehouse': 'Isaac-Velocity-Flat-Anymal-C-v0',  # Use mobile robot
        'inspection': 'Isaac-Velocity-Flat-Anymal-C-v0',
        'assembly': 'Isaac-Reach-Franka-v0',
    }

    def __init__(self):
        self.warnings = []

    def convert(self, scene_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Celestial scene config to Isaac Lab scene format

        Args:
            scene_config: Scene configuration from Celestial Studio

        Returns:
            Isaac Lab compatible scene configuration
        """
        self.warnings = []

        isaac_scene = {
            'scene': {
                'num_envs': 1,  # Single environment by default
                'env_spacing': 2.5,
                'replicate_physics': True
            },
            'robot': self._convert_robot(scene_config.get('robot', {})),
            'environment': self._convert_environment(scene_config.get('environment', {})),
            'objects': self._convert_objects(scene_config.get('objects', [])),
            'lighting': self._convert_lighting(scene_config.get('lighting', {})),
            'camera': self._convert_camera(scene_config.get('camera', {})),
            'metadata': {
                'source': 'celestial_studio',
                'original_config': scene_config,
                'warnings': self.warnings
            }
        }

        return isaac_scene

    def _convert_robot(self, robot_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert robot configuration"""
        robot_type = robot_config.get('type', 'mobile_robot')

        isaac_robot = {
            'type': robot_type,
            'asset_path': self.ROBOT_TYPE_MAPPING.get(robot_type),
            'spawn_position': robot_config.get('position', [0.0, 0.0, 0.5]),
            'spawn_orientation': robot_config.get('orientation', [0.0, 0.0, 0.0]),
            'articulation': {
                'enabled': True,
                'self_collisions': False,
                'fix_base_link': robot_type == 'robotic_arm'
            }
        }

        # Handle custom URDF robots
        if robot_type == 'urdf_custom':
            urdf_path = robot_config.get('urdf_path')
            if urdf_path:
                isaac_robot['urdf_path'] = urdf_path
                isaac_robot['asset_path'] = None
            else:
                self.warnings.append("Custom URDF robot specified but no urdf_path provided")

        # Add control configuration
        if 'control' in robot_config:
            isaac_robot['control'] = self._convert_control(robot_config['control'])

        # Add sensors
        if 'sensors' in robot_config:
            isaac_robot['sensors'] = self._convert_sensors(robot_config['sensors'])

        return isaac_robot

    def _convert_environment(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert environment configuration"""
        isaac_env = {
            'ground_plane': {
                'enabled': True,
                'size': env_config.get('floor', {}).get('size', [20.0, 20.0]),
                'physics_material': {
                    'static_friction': 0.5,
                    'dynamic_friction': 0.5,
                    'restitution': 0.0
                }
            },
            'sky_dome': {
                'enabled': True,
                'texture': 'kloofendal_48d_partly_cloudy'
            }
        }

        # Add walls if specified
        if env_config.get('walls', False):
            floor_size = env_config.get('floor', {}).get('size', [20.0, 20.0])
            isaac_env['walls'] = self._create_walls(floor_size)

        return isaac_env

    def _convert_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert objects to Isaac Lab format"""
        isaac_objects = []

        for obj in objects:
            obj_type = obj.get('type', 'box')

            isaac_obj = {
                'name': obj.get('name', f'{obj_type}_{len(isaac_objects)}'),
                'type': obj_type,
                'position': obj.get('position', [0.0, 0.0, 0.0]),
                'orientation': obj.get('orientation', [0.0, 0.0, 0.0]),
                'scale': obj.get('scale', [1.0, 1.0, 1.0]),
                'physics': {
                    'enabled': obj.get('physics_enabled', True),
                    'mass': obj.get('mass', 1.0),
                    'static': obj.get('static', False)
                }
            }

            # Map object types to USD primitives or assets
            if obj_type == 'box':
                isaac_obj['usd_path'] = '/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd'
            elif obj_type == 'sphere':
                isaac_obj['primitive'] = 'sphere'
                isaac_obj['radius'] = obj.get('radius', 0.5)
            elif obj_type == 'cylinder':
                isaac_obj['primitive'] = 'cylinder'
                isaac_obj['radius'] = obj.get('radius', 0.5)
                isaac_obj['height'] = obj.get('height', 1.0)
            elif obj_type == 'custom_mesh':
                isaac_obj['mesh_path'] = obj.get('mesh_path')

            # Add visual properties
            if 'color' in obj:
                isaac_obj['visual'] = {
                    'color': obj['color'],
                    'metallic': obj.get('metallic', 0.0),
                    'roughness': obj.get('roughness', 0.5)
                }

            isaac_objects.append(isaac_obj)

        return isaac_objects

    def _convert_lighting(self, lighting_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert lighting configuration"""
        return {
            'dome_light': {
                'enabled': True,
                'intensity': lighting_config.get('intensity', 1000.0),
                'texture': lighting_config.get('hdri', 'kloofendal_48d_partly_cloudy')
            },
            'distant_light': {
                'enabled': True,
                'intensity': 3000.0,
                'angle': 0.53,  # Sun angle
                'direction': [-1.0, -1.0, -1.0]
            }
        }

    def _convert_camera(self, camera_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert camera configuration"""
        return {
            'position': camera_config.get('position', [5.0, 5.0, 5.0]),
            'target': camera_config.get('target', [0.0, 0.0, 0.0]),
            'resolution': camera_config.get('resolution', [1920, 1080]),
            'fov': camera_config.get('fov', 60.0),
            'clipping_range': camera_config.get('clipping_range', [0.1, 1000.0])
        }

    def _convert_control(self, control_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert control configuration"""
        return {
            'type': control_config.get('type', 'velocity'),
            'stiffness': control_config.get('stiffness', 0.0),
            'damping': control_config.get('damping', 0.0),
            'max_effort': control_config.get('max_effort', 100.0)
        }

    def _convert_sensors(self, sensors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert sensor configurations"""
        isaac_sensors = []

        for sensor in sensors:
            sensor_type = sensor.get('type')

            isaac_sensor = {
                'type': sensor_type,
                'enabled': sensor.get('enabled', True),
                'update_rate': sensor.get('update_rate', 30.0)
            }

            if sensor_type == 'camera':
                isaac_sensor.update({
                    'resolution': sensor.get('resolution', [640, 480]),
                    'position': sensor.get('position', [0.0, 0.0, 0.0]),
                    'orientation': sensor.get('orientation', [0.0, 0.0, 0.0])
                })
            elif sensor_type == 'lidar':
                isaac_sensor.update({
                    'num_rays': sensor.get('num_rays', 360),
                    'range': sensor.get('range', 10.0),
                    'position': sensor.get('position', [0.0, 0.0, 0.0])
                })
            elif sensor_type == 'imu':
                isaac_sensor.update({
                    'position': sensor.get('position', [0.0, 0.0, 0.0])
                })

            isaac_sensors.append(isaac_sensor)

        return isaac_sensors

    def _create_walls(self, floor_size: List[float]) -> List[Dict[str, Any]]:
        """Create wall objects around the environment"""
        width, depth = floor_size
        wall_height = 2.0
        wall_thickness = 0.2

        walls = [
            {
                'name': 'wall_north',
                'primitive': 'box',
                'position': [0.0, depth / 2, wall_height / 2],
                'scale': [width, wall_thickness, wall_height],
                'physics': {'static': True}
            },
            {
                'name': 'wall_south',
                'primitive': 'box',
                'position': [0.0, -depth / 2, wall_height / 2],
                'scale': [width, wall_thickness, wall_height],
                'physics': {'static': True}
            },
            {
                'name': 'wall_east',
                'primitive': 'box',
                'position': [width / 2, 0.0, wall_height / 2],
                'scale': [wall_thickness, depth, wall_height],
                'physics': {'static': True}
            },
            {
                'name': 'wall_west',
                'primitive': 'box',
                'position': [-width / 2, 0.0, wall_height / 2],
                'scale': [wall_thickness, depth, wall_height],
                'physics': {'static': True}
            }
        ]

        return walls

    def suggest_isaac_task(self, scene_config: Dict[str, Any]) -> Optional[str]:
        """
        Suggest an Isaac Lab pre-built task based on scene configuration

        Args:
            scene_config: Celestial scene configuration

        Returns:
            Isaac Lab task name if match found, None otherwise
        """
        # Extract task type from scene
        task_type = scene_config.get('task', {}).get('type')
        robot_type = scene_config.get('robot', {}).get('type')
        environment = scene_config.get('environment', {}).get('type', '')

        # Try direct task mapping
        if task_type in self.TASK_MAPPING:
            return self.TASK_MAPPING[task_type]

        # Try robot + environment heuristics
        if robot_type == 'robotic_arm':
            if 'pick' in str(scene_config).lower() or 'grasp' in str(scene_config).lower():
                return 'Isaac-Lift-Cube-Franka-v0'
            return 'Isaac-Reach-Franka-v0'

        if robot_type == 'mobile_robot' or robot_type == 'quadruped':
            return 'Isaac-Velocity-Flat-Anymal-C-v0'

        if robot_type == 'humanoid':
            return 'Isaac-Velocity-Rough-Humanoid-v0'

        return None

    def validate_scene(self, isaac_scene: Dict[str, Any]) -> List[str]:
        """
        Validate converted Isaac Lab scene

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check robot configuration
        if not isaac_scene.get('robot'):
            errors.append("No robot specified in scene")
        elif not isaac_scene['robot'].get('asset_path') and not isaac_scene['robot'].get('urdf_path'):
            errors.append("Robot has no asset_path or urdf_path")

        # Check environment
        if not isaac_scene.get('environment'):
            errors.append("No environment specified")

        # Check object positions are valid
        for obj in isaac_scene.get('objects', []):
            pos = obj.get('position', [0, 0, 0])
            if len(pos) != 3:
                errors.append(f"Object {obj.get('name')} has invalid position: {pos}")

        return errors


class MultiRobotSceneConverter(SceneConverter):
    """Extended converter for multi-robot scenes"""

    def convert_multi_robot(self, multi_robot_scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert multi-robot scene to Isaac Lab format

        Args:
            multi_robot_scene: Multi-robot scene from MultiRobotManager.get_combined_scene()

        Returns:
            Isaac Lab multi-robot scene configuration
        """
        robots = multi_robot_scene.get('robots', [])

        isaac_scene = {
            'scene': {
                'num_envs': 1,
                'env_spacing': 2.5,
                'replicate_physics': True
            },
            'robots': [],
            'environment': self._convert_environment(multi_robot_scene.get('environment', {})),
            'objects': [],
            'metadata': {
                'source': 'celestial_studio_multi_robot',
                'scene_id': multi_robot_scene.get('scene_id'),
                'robot_count': len(robots)
            }
        }

        # Convert each robot
        for robot_data in robots:
            robot_config = robot_data.get('config', {}).get('robot', {})
            robot_config['position'] = robot_data.get('position', [0, 0, 0])
            robot_config['orientation'] = robot_data.get('orientation', [0, 0, 0])

            isaac_robot = self._convert_robot(robot_config)
            isaac_robot['id'] = robot_data.get('id')
            isaac_robot['name'] = robot_data.get('name')

            isaac_scene['robots'].append(isaac_robot)

        # Collect all objects from all robot scenes
        for robot_data in robots:
            objects = robot_data.get('config', {}).get('objects', [])
            isaac_scene['objects'].extend(self._convert_objects(objects))

        return isaac_scene


def convert_scene_to_isaac(scene_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to convert scene configuration

    Args:
        scene_config: Celestial scene configuration

    Returns:
        Isaac Lab scene configuration
    """
    converter = SceneConverter()
    isaac_scene = converter.convert(scene_config)

    # Validate the scene
    errors = converter.validate_scene(isaac_scene)
    if errors:
        print(f"Warning: Scene validation found {len(errors)} issues:")
        for error in errors:
            print(f"  - {error}")

    return isaac_scene


def suggest_training_task(scene_config: Dict[str, Any]) -> Optional[str]:
    """
    Suggest a pre-built Isaac Lab training task

    Args:
        scene_config: Celestial scene configuration

    Returns:
        Task name or None
    """
    converter = SceneConverter()
    return converter.suggest_isaac_task(scene_config)
