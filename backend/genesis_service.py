"""
Genesis Simulation Service
Manages Genesis physics engine, scene creation, robot control, and real-time streaming.
"""

import asyncio
import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from fastapi import WebSocket
from PIL import Image

try:
    import genesis as gs
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False
    logging.warning("Genesis not available. Install with: pip install genesis-world")


logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available Genesis backends"""
    METAL = "metal"  # Apple Silicon GPU
    CUDA = "cuda"    # NVIDIA GPU
    CPU = "cpu"      # CPU fallback
    VULKAN = "vulkan"  # Cross-platform GPU


class RobotType(Enum):
    """Supported robot types - using Genesis built-in assets"""
    # Simple primitives for testing
    MOBILE_ROBOT = "mobile"
    ROBOTIC_ARM = "arm"
    DRONE = "drone"

    # Genesis built-in robots (MJCF/XML)
    FRANKA = "franka"  # Franka Emika Panda arm
    ANT = "ant"  # Ant quadruped
    HUMANOID = "humanoid"  # Humanoid robot

    # Genesis built-in robots (URDF)
    GO2 = "go2"  # Unitree Go2 quadruped
    ANYMAL = "anymal"  # ANYmal C quadruped
    KUKA = "kuka"  # KUKA iiwa arm
    UR5E = "ur5e"  # Universal Robots UR5e arm
    SHADOW_HAND = "shadow_hand"  # Shadow dexterous hand
    CRAZYFLIE = "crazyflie"  # Crazyflie 2.X drone


@dataclass
class GenesisConfig:
    """Genesis initialization configuration"""
    backend: BackendType = BackendType.METAL
    fps: int = 60
    precision: str = "32"  # "16" or "32" bit
    show_viewer: bool = False  # Headless by default
    logging_level: str = "warning"

    # Rendering settings
    render_width: int = 1920
    render_height: int = 1080
    render_fps: int = 60

    # Video streaming
    video_quality: int = 85  # JPEG quality (0-100)
    stream_fps: int = 30  # Stream at lower FPS to reduce bandwidth

    # Simulation
    substeps: int = 10
    gravity: Tuple[float, float, float] = (0, 0, -9.81)


@dataclass
class RobotState:
    """Robot state information"""
    robot_id: str
    robot_type: RobotType
    position: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    algorithm: Optional[str] = None
    algorithm_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObstacleState:
    """Obstacle information"""
    obstacle_id: str
    position: np.ndarray
    size: np.ndarray
    rotation: np.ndarray


class GenesisSimulation:
    """
    Genesis simulation manager
    Handles scene creation, robot control, physics stepping, and rendering.
    Uses Genesis built-in assets with relative paths (e.g., "xml/franka_emika_panda/panda.xml")
    """

    def __init__(self, config: GenesisConfig):
        self.config = config
        self.scene: Optional[gs.Scene] = None
        self.robots: Dict[str, Any] = {}  # robot_id -> entity
        self.robot_types: Dict[str, RobotType] = {}  # robot_id -> robot_type
        self.obstacles: Dict[str, Any] = {}
        self.algorithms: Dict[str, Dict[str, Any]] = {}  # robot_id -> {function, type, params, goal}

        self.is_initialized = False
        self.is_running = False
        self.step_count = 0
        self.start_time = 0.0

        # Cameras for rendering (Genesis requires explicit camera objects)
        self.cameras: Dict[str, Any] = {}  # camera_name -> camera object
        self.active_camera: str = "main"  # Currently active camera

        # Frame capture
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time = 0.0

        # Connected clients
        self.websocket_clients: List[WebSocket] = []

        # Auto-detect backend if not specified
        if self.config.backend == BackendType.METAL:
            self._auto_detect_backend()

    def _auto_detect_backend(self):
        """Auto-detect best available backend"""
        import platform
        import subprocess

        system = platform.system()

        # Check for Apple Silicon
        if system == "Darwin":
            try:
                # Check for Apple Silicon
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True
                )
                if "Apple" in result.stdout:
                    self.config.backend = BackendType.METAL
                    logger.info("🍎 Auto-detected Apple Silicon - using Metal backend")
                    return
            except:
                pass

        # Check for NVIDIA GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.config.backend = BackendType.CUDA
                logger.info("🎮 Auto-detected NVIDIA GPU - using CUDA backend")
                return
        except:
            pass

        # Fallback to CPU
        self.config.backend = BackendType.CPU
        logger.info("💻 Using CPU backend (slower)")

    def _create_cameras(self):
        """
        Create multiple cameras for different viewpoints

        Cameras:
        - main: Third-person view (default)
        - fpv: First-person view (attached to robot)
        - top: Top-down view
        - debug: Debug/close-up view
        """
        if not GENESIS_AVAILABLE or self.scene is None:
            return

        logger.info("Creating cameras for rendering...")

        # Main camera - third-person view
        self.cameras["main"] = self.scene.add_camera(
            res=(self.config.render_width, self.config.render_height),
            pos=(3.5, 0.0, 2.5),
            lookat=(0.0, 0.0, 0.5),
            fov=40,
            GUI=False,  # CRITICAL: Don't show in viewer, just render to buffer
        )
        logger.info("✅ Main camera created (third-person)")

        # FPV camera - will be attached to robot later
        self.cameras["fpv"] = self.scene.add_camera(
            res=(1280, 720),  # Lower res for FPV (faster)
            pos=(0.0, 0.0, 0.5),  # Default position, will be attached
            lookat=(1.0, 0.0, 0.5),
            fov=60,  # Wider FOV for FPV
            GUI=False,
        )
        logger.info("✅ FPV camera created (will attach to robot)")

        # Top-down camera
        self.cameras["top"] = self.scene.add_camera(
            res=(1280, 720),
            pos=(0.0, 0.0, 5.0),  # 5m above ground
            lookat=(0.0, 0.0, 0.0),
            fov=60,
            GUI=False,
        )
        logger.info("✅ Top-down camera created")

        # Debug camera - close-up view
        self.cameras["debug"] = self.scene.add_camera(
            res=(640, 480),  # Low res for debug
            pos=(1.0, 1.0, 1.0),
            lookat=(0.0, 0.0, 0.5),
            fov=45,
            GUI=False,
        )
        logger.info("✅ Debug camera created")

        logger.info(f"📷 Created {len(self.cameras)} cameras")

    def _create_scene(self):
        """Create Genesis scene (separated for reusability)"""
        if not GENESIS_AVAILABLE:
            raise RuntimeError("Genesis not available")

        logger.info("Creating Genesis scene...")

        # Create scene with viewer options
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1.0 / self.config.fps,
                substeps=self.config.substeps,
                gravity=self.config.gravity,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                max_FPS=self.config.render_fps,
                res=(self.config.render_width, self.config.render_height),
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=1.0,
                show_link_frame=False,
                show_cameras=False,
            ),
            show_viewer=self.config.show_viewer,
        )

        # Add ground plane
        self.scene.add_entity(gs.morphs.Plane())

        # Create cameras for rendering (CRITICAL: cameras must be created for frame capture)
        self._create_cameras()

        logger.info("✅ Scene created successfully")

    def initialize(self):
        """Initialize Genesis engine"""
        if not GENESIS_AVAILABLE:
            raise RuntimeError("Genesis not installed. Install with: pip install genesis-world")

        # Check if Genesis is ALREADY initialized globally
        # Genesis maintains a global _initialized flag that prevents multiple gs.init() calls
        if hasattr(gs, '_initialized') and gs._initialized:
            logger.warning("⚠️ Genesis already initialized globally, skipping gs.init()")
            self.is_initialized = True

            # Still create scene if needed
            if self.scene is None:
                self._create_scene()
            return

        logger.info(f"Initializing Genesis with {self.config.backend.value} backend...")

        # Map backend enum to Genesis backend
        backend_map = {
            BackendType.METAL: gs.metal,
            BackendType.CUDA: gs.cuda,
            BackendType.CPU: gs.cpu,
            BackendType.VULKAN: gs.vulkan,
        }

        backend = backend_map.get(self.config.backend, gs.cpu)

        # Initialize Genesis (ONLY if not already initialized)
        gs.init(
            backend=backend,
            precision=self.config.precision,
            logging_level=self.config.logging_level,
        )

        # Create scene
        self._create_scene()

        self.is_initialized = True
        logger.info("✅ Genesis initialized successfully")

    @classmethod
    def discover_available_models(cls) -> Dict[str, List[Dict[str, str]]]:
        """
        Get all available Genesis built-in robot models

        Returns:
            Dictionary with categories (urdf, xml) and model information
        """
        models = {
            "urdf": [
                {
                    "name": "Unitree Go2",
                    "path": "urdf/go2/urdf/go2.urdf",
                    "type": "urdf",
                    "enum": "GO2",
                    "description": "Quadruped robot"
                },
                {
                    "name": "ANYmal C",
                    "path": "urdf/anymal_c/urdf/anymal_c.urdf",
                    "type": "urdf",
                    "enum": "ANYMAL",
                    "description": "Quadruped robot"
                },
                {
                    "name": "KUKA iiwa",
                    "path": "urdf/kuka_iiwa/model.urdf",
                    "type": "urdf",
                    "enum": "KUKA",
                    "description": "7-DOF robotic arm"
                },
                {
                    "name": "Shadow Hand",
                    "path": "urdf/shadow_hand/shadow_hand.urdf",
                    "type": "urdf",
                    "enum": "SHADOW_HAND",
                    "description": "Dexterous robotic hand"
                },
                {
                    "name": "Crazyflie 2.X",
                    "path": "urdf/drones/cf2x.urdf",
                    "type": "urdf",
                    "enum": "CRAZYFLIE",
                    "description": "Nano quadcopter drone"
                },
            ],
            "xml": [
                {
                    "name": "Franka Panda",
                    "path": "xml/franka_emika_panda/panda.xml",
                    "type": "mjcf",
                    "enum": "FRANKA",
                    "description": "7-DOF collaborative arm"
                },
                {
                    "name": "Ant",
                    "path": "xml/ant.xml",
                    "type": "mjcf",
                    "enum": "ANT",
                    "description": "Quadruped ant robot"
                },
                {
                    "name": "Humanoid",
                    "path": "xml/humanoid.xml",
                    "type": "mjcf",
                    "enum": "HUMANOID",
                    "description": "Bipedal humanoid robot"
                },
                {
                    "name": "UR5e",
                    "path": "xml/universal_robots_ur5e/ur5e.xml",
                    "type": "mjcf",
                    "enum": "UR5E",
                    "description": "6-DOF collaborative arm"
                },
            ],
        }

        return models

    def add_robot(self, robot_id: str, robot_type: RobotType, position: Tuple[float, float, float] = (0, 0, 0.5)) -> Any:
        """Add a robot to the scene using Genesis built-in assets"""
        if not self.is_initialized:
            self.initialize()

        logger.info(f"Adding robot: {robot_id} ({robot_type.value}) at position {position}")

        robot = None

        try:
            # Genesis built-in robots (MJCF/XML format)
            if robot_type == RobotType.FRANKA:
                robot = self.scene.add_entity(
                    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=position)
                )
                logger.info(f"✅ Loaded Franka Panda (MJCF)")

            elif robot_type == RobotType.ANT:
                robot = self.scene.add_entity(
                    gs.morphs.MJCF(file="xml/ant.xml", pos=position)
                )
                logger.info(f"✅ Loaded Ant quadruped (MJCF)")

            elif robot_type == RobotType.HUMANOID:
                robot = self.scene.add_entity(
                    gs.morphs.MJCF(file="xml/humanoid.xml", pos=position)
                )
                logger.info(f"✅ Loaded Humanoid robot (MJCF)")

            # Genesis built-in robots (URDF format)
            elif robot_type == RobotType.GO2:
                robot = self.scene.add_entity(
                    gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=position)
                )
                logger.info(f"✅ Loaded Unitree Go2 (URDF)")

            elif robot_type == RobotType.ANYMAL:
                robot = self.scene.add_entity(
                    gs.morphs.URDF(file="urdf/anymal_c/urdf/anymal_c.urdf", pos=position)
                )
                logger.info(f"✅ Loaded ANYmal C (URDF)")

            elif robot_type == RobotType.KUKA:
                robot = self.scene.add_entity(
                    gs.morphs.URDF(file="urdf/kuka_iiwa/model.urdf", pos=position)
                )
                logger.info(f"✅ Loaded KUKA iiwa (URDF)")

            elif robot_type == RobotType.UR5E:
                robot = self.scene.add_entity(
                    gs.morphs.MJCF(file="xml/universal_robots_ur5e/ur5e.xml", pos=position)
                )
                logger.info(f"✅ Loaded UR5e arm (MJCF)")

            elif robot_type == RobotType.SHADOW_HAND:
                robot = self.scene.add_entity(
                    gs.morphs.URDF(file="urdf/shadow_hand/shadow_hand.urdf", pos=position)
                )
                logger.info(f"✅ Loaded Shadow Hand (URDF)")

            elif robot_type == RobotType.CRAZYFLIE:
                robot = self.scene.add_entity(
                    gs.morphs.URDF(file="urdf/drones/cf2x.urdf", pos=position)
                )
                logger.info(f"✅ Loaded Crazyflie 2.X (URDF)")

            # Simple primitives for algorithm testing
            elif robot_type == RobotType.MOBILE_ROBOT:
                robot = self._create_mobile_robot(position)

            elif robot_type == RobotType.ROBOTIC_ARM:
                # Use simple 2-link arm from Genesis
                try:
                    robot = self.scene.add_entity(
                        gs.morphs.URDF(file="urdf/simple/two_link_arm.urdf", pos=position)
                    )
                    logger.info(f"✅ Loaded simple 2-link arm (URDF)")
                except Exception as e:
                    logger.warning(f"Could not load simple arm: {e}, using primitive")
                    robot = self._create_robotic_arm(position)

            elif robot_type == RobotType.DRONE:
                robot = self._create_drone(position)

        except Exception as e:
            logger.error(f"Failed to load robot {robot_type.value}: {e}")
            # Try fallback primitives
            if robot_type in [RobotType.MOBILE_ROBOT, RobotType.ROBOTIC_ARM, RobotType.DRONE]:
                if robot_type == RobotType.MOBILE_ROBOT:
                    robot = self._create_mobile_robot(position)
                elif robot_type == RobotType.ROBOTIC_ARM:
                    robot = self._create_robotic_arm(position)
                else:
                    robot = self._create_drone(position)

        if robot is not None:
            self.robots[robot_id] = robot
            self.robot_types[robot_id] = robot_type  # Store robot type for algorithm execution
            logger.info(f"✅ Robot {robot_id} added successfully")
        else:
            logger.error(f"❌ Failed to create robot {robot_id}")

        return robot

    def _create_mobile_robot(self, position: Tuple[float, float, float]) -> Any:
        """Create a simple 4-wheeled mobile robot"""
        # For now, create a simple box
        # TODO: Create proper URDF with wheels
        robot = self.scene.add_entity(
            gs.morphs.Box(
                pos=position,
                size=(0.5, 0.3, 0.2),
                fixed=False,
            )
        )
        return robot

    def _create_robotic_arm(self, position: Tuple[float, float, float]) -> Any:
        """Create a simple robotic arm"""
        # TODO: Create proper URDF with joints
        robot = self.scene.add_entity(
            gs.morphs.Box(
                pos=position,
                size=(0.1, 0.1, 0.5),
                fixed=False,
            )
        )
        return robot

    def _create_drone(self, position: Tuple[float, float, float]) -> Any:
        """Create a quadcopter drone (Crazyflie) using Genesis built-in asset"""
        try:
            robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/drones/cf2x.urdf",  # Genesis relative path
                    pos=position,
                )
            )
            logger.info(f"✅ Loaded Crazyflie drone (URDF)")
            return robot
        except Exception as e:
            logger.warning(f"Could not load drone model: {e}, falling back to simple sphere")
            # Fallback to simple sphere
            robot = self.scene.add_entity(
                gs.morphs.Sphere(pos=position, radius=0.2, fixed=False)
            )
            return robot

    def add_obstacle(self, obstacle_id: str, position: Tuple[float, float, float], size: Tuple[float, float, float]) -> Any:
        """Add an obstacle to the scene"""
        if not self.is_initialized:
            self.initialize()

        obstacle = self.scene.add_entity(
            gs.morphs.Box(
                pos=position,
                size=size,
                fixed=True,
            )
        )

        self.obstacles[obstacle_id] = obstacle
        logger.info(f"Added obstacle: {obstacle_id}")
        return obstacle

    def build_scene(self):
        """Build the scene (required before stepping)"""
        if not self.is_initialized:
            raise RuntimeError("Scene not initialized")

        logger.info("Building Genesis scene...")
        self.scene.build()
        logger.info("✅ Scene built successfully")

    def set_algorithm(
        self,
        robot_id: str,
        algorithm: Callable,
        algorithm_type: str,
        params: Dict[str, Any] = None,
        goal: Optional[np.ndarray] = None
    ):
        """
        Set algorithm for a robot

        Args:
            robot_id: ID of the robot
            algorithm: Compiled algorithm function
            algorithm_type: Type of algorithm (path_planning, obstacle_avoidance, etc.)
            params: Algorithm parameters
            goal: Goal position/state for the robot (if applicable)
        """
        if robot_id not in self.robots:
            logger.error(f"Robot {robot_id} not found")
            return

        self.algorithms[robot_id] = {
            "function": algorithm,
            "type": algorithm_type,
            "params": params or {},
            "goal": goal
        }
        logger.info(f"Algorithm ({algorithm_type}) set for robot {robot_id}")

    def start(self):
        """Start simulation"""
        if not self.is_initialized:
            self.build_scene()

        self.is_running = True
        self.start_time = time.time()
        logger.info("▶️  Simulation started")

    def stop(self):
        """Stop simulation"""
        self.is_running = False
        logger.info("⏸️  Simulation stopped")

    def step(self) -> Dict[str, Any]:
        """
        Step the simulation forward one timestep
        Returns state information
        """
        if not self.is_running:
            return {}

        # Execute algorithms for each robot
        for robot_id, algo_data in self.algorithms.items():
            if robot_id in self.robots:
                try:
                    robot = self.robots[robot_id]
                    algorithm_func = algo_data["function"]
                    algorithm_type = algo_data["type"]
                    algo_params = algo_data["params"]
                    goal = algo_data.get("goal")

                    # Extract robot state from Genesis entity
                    robot_state = self._extract_robot_state(robot, robot_id)

                    # Prepare obstacle information
                    obstacles = self._get_obstacles_list()

                    # Execute algorithm based on type
                    result = self._execute_algorithm(
                        algorithm_func,
                        algorithm_type,
                        robot_state,
                        obstacles,
                        goal,
                        algo_params
                    )

                    # Apply algorithm result to robot
                    if result is not None:
                        self._apply_algorithm_result(
                            robot,
                            robot_id,
                            algorithm_type,
                            result
                        )

                except Exception as e:
                    logger.error(f"Algorithm execution failed for {robot_id}: {e}")

        # Step physics
        self.scene.step()
        self.step_count += 1

        # Capture frame every N steps based on stream FPS
        frame_interval = self.config.render_fps // self.config.stream_fps
        if self.step_count % frame_interval == 0:
            self.last_frame = self._capture_frame()
            self.last_frame_time = time.time()

        # Get state
        state = self._get_state()

        return state

    async def run_async(self, duration: Optional[float] = None):
        """
        Run simulation asynchronously
        Allows other async tasks to run concurrently
        """
        self.start()

        start = time.time()

        while self.is_running:
            # Step simulation
            state = self.step()

            # Broadcast state to WebSocket clients
            await self._broadcast_state(state)

            # Check duration
            if duration is not None and (time.time() - start) >= duration:
                break

            # Sleep to maintain FPS
            await asyncio.sleep(1.0 / self.config.fps)

        self.stop()

    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture current frame from active Genesis camera

        Returns:
            RGB frame as numpy array (H, W, 3) with dtype uint8, or None if capture fails
        """
        try:
            # Get active camera
            if self.active_camera not in self.cameras:
                logger.warning(f"Active camera '{self.active_camera}' not found, using 'main'")
                self.active_camera = "main"

            if "main" not in self.cameras:
                logger.error("No cameras available for frame capture")
                return None

            camera = self.cameras[self.active_camera]

            # Render frame from camera (CORRECT Genesis API)
            # Returns: (rgb, depth, segmentation, normal) tuple
            # We only need RGB for video streaming
            rgb, _, _, _ = camera.render(
                rgb=True,
                depth=False,
                segmentation=False,
                normal=False
            )

            # rgb is numpy array with shape (H, W, 3) and dtype uint8
            return rgb

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None

    def get_frame_jpeg(self) -> Optional[bytes]:
        """Get current frame as JPEG bytes"""
        if self.last_frame is None:
            return None

        try:
            # Convert numpy array to PIL Image
            if self.last_frame.dtype != np.uint8:
                frame = (self.last_frame * 255).astype(np.uint8)
            else:
                frame = self.last_frame

            image = Image.fromarray(frame)

            # Compress to JPEG
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.config.video_quality)
            buffer.seek(0)

            return buffer.read()
        except Exception as e:
            logger.error(f"JPEG encoding failed: {e}")
            return None

    def get_frame_base64(self) -> Optional[str]:
        """Get current frame as base64-encoded JPEG"""
        jpeg_bytes = self.get_frame_jpeg()
        if jpeg_bytes is None:
            return None

        return base64.b64encode(jpeg_bytes).decode('utf-8')

    def set_active_camera(self, camera_name: str) -> bool:
        """
        Switch to a different camera

        Args:
            camera_name: Camera name ("main", "fpv", "top", "debug")

        Returns:
            True if camera switched successfully, False otherwise
        """
        if camera_name in self.cameras:
            self.active_camera = camera_name
            logger.info(f"📷 Switched to {camera_name} camera")
            return True
        else:
            logger.warning(f"Camera '{camera_name}' not found. Available: {list(self.cameras.keys())}")
            return False

    def get_available_cameras(self) -> List[str]:
        """Get list of available camera names"""
        return list(self.cameras.keys())

    def attach_fpv_camera(self, robot_id: str) -> bool:
        """
        Attach FPV camera to a robot for first-person view

        Args:
            robot_id: ID of robot to attach camera to

        Returns:
            True if attached successfully, False otherwise
        """
        if robot_id not in self.robots:
            logger.error(f"Robot {robot_id} not found")
            return False

        if "fpv" not in self.cameras:
            logger.error("FPV camera not available")
            return False

        try:
            robot = self.robots[robot_id]
            fpv_camera = self.cameras["fpv"]

            # Create offset transformation for camera mount
            # Position camera slightly forward and up from robot base
            import genesis.utils.geom as gu
            trans = np.array([0.2, 0.0, 0.3])  # 0.2m forward, 0.3m up
            R = np.eye(3)  # No rotation offset
            offset_T = gu.trans_R_to_T(trans, R)

            # Attach camera to robot base link
            # Note: This works for entities with links. For simple boxes, we'll use follow_entity
            if hasattr(robot, 'links') and len(robot.links) > 0:
                fpv_camera.attach(robot.links[0], offset_T)
                logger.info(f"📷 FPV camera attached to {robot_id} (rigid mount)")
            else:
                # Fallback: use follow_entity for simple geometries
                fpv_camera.follow_entity(robot, fix_orientation=False)
                logger.info(f"📷 FPV camera following {robot_id} (tracking mode)")

            return True

        except Exception as e:
            logger.error(f"Failed to attach FPV camera: {e}")
            return False

    def apply_robot_action(
        self,
        robot_id: str,
        action: List[float],
        robot_type: Optional[RobotType] = None
    ) -> bool:
        """
        Apply action to robot (for teleoperation)

        Args:
            robot_id: ID of robot
            action: Action vector (meaning depends on robot type)
                - Mobile: [vx, vy, vz] linear velocity
                - Arm: joint positions or velocities
                - Drone: [thrust, roll, pitch, yaw]
            robot_type: Type of robot (auto-detected if None)

        Returns:
            True if action applied successfully
        """
        if robot_id not in self.robots:
            logger.error(f"Robot {robot_id} not found")
            return False

        try:
            robot = self.robots[robot_id]

            # For simple geometries (Box, Sphere), directly set velocity
            if hasattr(robot, 'set_vel'):
                # Mobile robot: action = [vx, vy, angular_vel]
                if len(action) >= 2:
                    robot.set_vel(np.array([action[0], action[1], 0.0]))

                    # Set angular velocity if provided
                    if len(action) >= 3 and hasattr(robot, 'set_ang'):
                        robot.set_ang(np.array([0.0, 0.0, action[2]]))

                logger.debug(f"Applied velocity to {robot_id}: {action}")
                return True

            # For robots with DOFs (URDF/MJCF), use joint control
            elif hasattr(robot, 'control_dofs_position'):
                # Get number of DOFs
                if hasattr(robot, 'n_dofs'):
                    n_dofs = robot.n_dofs
                    dofs_idx = list(range(n_dofs))

                    # Trim action to match DOFs
                    action_array = np.array(action[:n_dofs])

                    # Apply position control
                    robot.control_dofs_position(action_array, dofs_idx)
                    logger.debug(f"Applied joint positions to {robot_id}")
                    return True
                else:
                    logger.warning(f"Robot {robot_id} has DOF control but n_dofs not found")
                    return False

            else:
                logger.warning(f"Robot {robot_id} doesn't support known control methods")
                return False

        except Exception as e:
            logger.error(f"Failed to apply action to {robot_id}: {e}")
            return False

    @staticmethod
    def _tensor_to_list(value) -> List[float]:
        """
        Convert Genesis Tensor, numpy array, or other types to JSON-serializable list

        Args:
            value: Genesis Tensor, numpy array, list, tuple, or other numeric type

        Returns:
            JSON-serializable list of floats
        """
        import numpy as np

        # Genesis Tensor - check for to_numpy() method
        if hasattr(value, 'to_numpy'):
            return value.to_numpy().tolist()

        # Genesis Tensor - alternative method
        if hasattr(value, 'numpy'):
            return value.numpy().tolist()

        # Numpy array
        if isinstance(value, np.ndarray):
            return value.tolist()

        # Already a list or tuple
        if isinstance(value, (list, tuple)):
            return list(value)

        # Single numeric value
        if isinstance(value, (int, float)):
            return [float(value)]

        # Fallback to zero vector
        logger.warning(f"Could not convert {type(value)} to list, using default [0, 0, 0]")
        return [0, 0, 0]

    def get_robot_state(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current state of a robot

        Returns:
            Dictionary with position, velocity, orientation, etc.
        """
        if robot_id not in self.robots:
            return None

        try:
            robot = self.robots[robot_id]
            state = {}

            # Get position
            if hasattr(robot, 'get_pos'):
                pos = robot.get_pos()
                state['position'] = self._tensor_to_list(pos)
            else:
                state['position'] = [0, 0, 0]

            # Get velocity
            if hasattr(robot, 'get_vel'):
                vel = robot.get_vel()
                state['velocity'] = self._tensor_to_list(vel)
            else:
                state['velocity'] = [0, 0, 0]

            # Get quaternion
            if hasattr(robot, 'get_quat'):
                quat = robot.get_quat()
                state['quaternion'] = self._tensor_to_list(quat)
            else:
                state['quaternion'] = [0, 0, 0, 1]

            # Get angular velocity
            if hasattr(robot, 'get_ang'):
                ang = robot.get_ang()
                state['angular_velocity'] = self._tensor_to_list(ang)
            else:
                state['angular_velocity'] = [0, 0, 0]

            # Get DOF states if available
            if hasattr(robot, 'get_dofs_position') and hasattr(robot, 'n_dofs'):
                n_dofs = robot.n_dofs
                dofs_idx = list(range(n_dofs))
                joint_pos = robot.get_dofs_position(dofs_idx)
                state['joint_positions'] = self._tensor_to_list(joint_pos)

                if hasattr(robot, 'get_dofs_velocity'):
                    joint_vel = robot.get_dofs_velocity(dofs_idx)
                    state['joint_velocities'] = self._tensor_to_list(joint_vel)

            return state

        except Exception as e:
            logger.error(f"Failed to get state for {robot_id}: {e}")
            return None

    def set_robot_position(
        self,
        robot_id: str,
        position: Tuple[float, float, float],
        quaternion: Optional[Tuple[float, float, float, float]] = None
    ) -> bool:
        """
        Set robot position (hard reset, violates physics)

        Args:
            robot_id: ID of robot
            position: (x, y, z) position
            quaternion: (x, y, z, w) orientation (optional)

        Returns:
            True if position set successfully
        """
        if robot_id not in self.robots:
            logger.error(f"Robot {robot_id} not found")
            return False

        try:
            robot = self.robots[robot_id]

            # Set position
            if hasattr(robot, 'set_pos'):
                robot.set_pos(np.array(position))

            # Set quaternion if provided
            if quaternion and hasattr(robot, 'set_quat'):
                robot.set_quat(np.array(quaternion))

            logger.info(f"Set position for {robot_id}: {position}")
            return True

        except Exception as e:
            logger.error(f"Failed to set position for {robot_id}: {e}")
            return False

    def _extract_robot_state(self, robot: Any, robot_id: str) -> Dict[str, Any]:
        """
        Extract current state from Genesis robot entity

        Returns dict with: position, velocity, orientation, joint_angles (for arms)
        """
        try:
            # Get basic state (available for all entities)
            position = robot.get_pos()  # Returns np.ndarray (x, y, z)
            velocity = robot.get_vel()  # Returns np.ndarray (vx, vy, vz)

            state = {
                "position": position,
                "velocity": velocity,
                "robot_id": robot_id
            }

            # For articulated robots (arms, hands), get joint states
            robot_type = self.robot_types.get(robot_id)
            if robot_type in [RobotType.FRANKA, RobotType.KUKA, RobotType.UR5E,
                             RobotType.ROBOTIC_ARM, RobotType.SHADOW_HAND]:
                try:
                    # Get all DOF indices
                    n_dofs = robot.n_dofs
                    dof_indices = list(range(n_dofs))

                    # Get joint positions and velocities
                    joint_positions = robot.get_dofs_position(dofs_idx_local=dof_indices)
                    joint_velocities = robot.get_dofs_velocity(dofs_idx_local=dof_indices)

                    state["joint_angles"] = joint_positions
                    state["joint_velocities"] = joint_velocities
                except Exception as e:
                    logger.debug(f"Could not get joint state: {e}")

            return state

        except Exception as e:
            logger.error(f"Failed to extract robot state: {e}")
            return {"position": np.zeros(3), "velocity": np.zeros(3)}

    def _get_obstacles_list(self) -> List[Dict]:
        """Get list of obstacles in format expected by algorithms"""
        obstacles = []
        for obs_id, obs_entity in self.obstacles.items():
            try:
                position = obs_entity.get_pos()
                # Estimate radius from entity (approximate for now)
                radius = 0.5  # Default radius
                obstacles.append({
                    "id": obs_id,
                    "position": position,
                    "radius": radius
                })
            except Exception as e:
                logger.debug(f"Could not get obstacle {obs_id} state: {e}")

        return obstacles

    def _execute_algorithm(
        self,
        algorithm_func: Callable,
        algorithm_type: str,
        robot_state: Dict[str, Any],
        obstacles: List[Dict],
        goal: Optional[np.ndarray],
        params: Dict[str, Any]
    ) -> Any:
        """
        Execute algorithm function with appropriate parameters based on type

        Returns algorithm output (varies by type)
        """
        try:
            if algorithm_type == "path_planning":
                # find_path(start, goal, obstacles, robot_state) -> List[np.ndarray]
                if goal is None:
                    return None
                return algorithm_func(
                    robot_state["position"],
                    goal,
                    obstacles,
                    robot_state
                )

            elif algorithm_type == "obstacle_avoidance":
                # compute_safe_velocity(current_pos, current_vel, obstacles, goal, max_speed, params) -> np.ndarray
                max_speed = params.get("max_speed", 2.0)
                goal_pos = goal if goal is not None else robot_state["position"] + np.array([1, 0, 0])
                return algorithm_func(
                    robot_state["position"],
                    robot_state["velocity"],
                    obstacles,
                    goal_pos,
                    max_speed,
                    params
                )

            elif algorithm_type == "inverse_kinematics":
                # solve_ik(target_pos, current_angles, link_lengths, params) -> np.ndarray
                if goal is None or "joint_angles" not in robot_state:
                    return None
                link_lengths = params.get("link_lengths", np.array([1.0, 0.8, 0.6, 0.4]))
                return algorithm_func(
                    goal,
                    robot_state["joint_angles"],
                    link_lengths,
                    params
                )

            elif algorithm_type == "computer_vision":
                # process_vision(camera_state, scene_objects, params) -> List[Dict]
                # For now, pass basic camera info
                camera_state = {
                    "robot_position": robot_state["position"],
                    "robot_velocity": robot_state["velocity"]
                }
                scene_objects = obstacles  # Obstacles as scene objects for now
                return algorithm_func(camera_state, scene_objects, params)

            else:
                logger.warning(f"Unknown algorithm type: {algorithm_type}")
                return None

        except Exception as e:
            logger.error(f"Algorithm execution error: {e}")
            return None

    def _apply_algorithm_result(
        self,
        robot: Any,
        robot_id: str,
        algorithm_type: str,
        result: Any
    ):
        """
        Apply algorithm output to Genesis robot

        Different algorithms produce different outputs:
        - path_planning: List of waypoints
        - obstacle_avoidance: Velocity vector
        - inverse_kinematics: Joint angles
        - computer_vision: Detected objects (no direct control)
        """
        try:
            robot_type = self.robot_types.get(robot_id)

            if algorithm_type == "path_planning":
                # Result is List[np.ndarray] of waypoints
                if isinstance(result, list) and len(result) > 0:
                    # Navigate to first waypoint
                    target = result[0]
                    current_pos = robot.get_pos()
                    direction = target - current_pos
                    distance = np.linalg.norm(direction)

                    if distance > 0.1:  # Not at waypoint yet
                        direction = direction / distance  # Normalize
                        velocity = direction * 1.0  # 1 m/s speed

                        # Apply velocity
                        robot.set_velocity(
                            linear=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
                            angular=(0, 0, 0)
                        )

            elif algorithm_type == "obstacle_avoidance":
                # Result is np.ndarray velocity vector
                if isinstance(result, np.ndarray) and len(result) >= 2:
                    # Apply safe velocity (assume XZ plane for mobile robots)
                    robot.set_velocity(
                        linear=(float(result[0]), 0, float(result[1])),
                        angular=(0, 0, 0)
                    )

            elif algorithm_type == "inverse_kinematics":
                # Result is np.ndarray of joint angles
                if isinstance(result, np.ndarray) and robot_type in [
                    RobotType.FRANKA, RobotType.KUKA, RobotType.UR5E,
                    RobotType.ROBOTIC_ARM, RobotType.SHADOW_HAND
                ]:
                    # Apply joint angles as position targets
                    n_dofs = min(len(result), robot.n_dofs)
                    dof_indices = list(range(n_dofs))
                    robot.control_dofs_position(result[:n_dofs], dof_indices)

            elif algorithm_type == "computer_vision":
                # Computer vision doesn't directly control robot
                # Results would be used by other algorithms
                pass

        except Exception as e:
            logger.error(f"Failed to apply algorithm result: {e}")

    def _get_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        state = {
            'timestamp': time.time(),
            'step': self.step_count,
            'fps': self.step_count / max(0.001, time.time() - self.start_time),
            'robots': {},
            'obstacles': {},
        }

        # Get robot states
        for robot_id in self.robots.keys():
            robot_state = self.get_robot_state(robot_id)
            if robot_state:
                state['robots'][robot_id] = robot_state

        return state

    async def _broadcast_state(self, state: Dict[str, Any]):
        """Broadcast state to all connected WebSocket clients"""
        if not self.websocket_clients:
            return

        message = json.dumps({
            'type': 'state_update',
            'data': state
        })

        # Remove disconnected clients
        disconnected = []

        for client in self.websocket_clients:
            try:
                await client.send_text(message)
            except Exception as e:
                logger.warning(f"Client disconnected: {e}")
                disconnected.append(client)

        # Clean up disconnected clients
        for client in disconnected:
            self.websocket_clients.remove(client)

    async def add_websocket_client(self, websocket: WebSocket):
        """Add a WebSocket client"""
        await websocket.accept()
        self.websocket_clients.append(websocket)
        logger.info(f"WebSocket client connected ({len(self.websocket_clients)} total)")

        # Send initial state
        state = self._get_state()
        await websocket.send_json({
            'type': 'initial_state',
            'data': state
        })

    def remove_websocket_client(self, websocket: WebSocket):
        """Remove a WebSocket client"""
        if websocket in self.websocket_clients:
            self.websocket_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected ({len(self.websocket_clients)} remaining)")

    def reset_scene(self):
        """
        Reset scene for a new simulation
        This clears all robots and obstacles, and resets the scene state
        WITHOUT destroying the Genesis engine (avoids "already initialized" error)
        """
        logger.info("🔄 Resetting scene for new simulation...")

        # Stop simulation if running
        was_running = self.is_running
        if was_running:
            self.stop()

        # Clear robot and obstacle dictionaries
        self.robots.clear()
        self.obstacles.clear()
        self.algorithms.clear()

        # Reset counters
        self.step_count = 0
        self.start_time = time.time()

        # If scene exists, reset it using Genesis's reset API
        if self.scene is not None:
            try:
                # Genesis scenes have a reset() method that resets state without destroying
                if hasattr(self.scene, 'reset'):
                    self.scene.reset()
                    logger.info("✅ Scene reset using Genesis API")
                else:
                    # Fallback: destroy and recreate scene
                    logger.info("Scene reset not available, recreating scene...")
                    self.scene = None
                    self._create_scene()
            except Exception as e:
                logger.warning(f"Scene reset failed: {e}, recreating scene...")
                self.scene = None
                self._create_scene()
        else:
            # No scene exists, create new one
            self._create_scene()

        logger.info("✅ Scene reset complete, ready for new simulation")

        # Restart if was running
        if was_running:
            self.start()

    def reset(self):
        """Reset simulation to initial state (deprecated, use reset_scene instead)"""
        self.step_count = 0
        self.start_time = time.time()
        logger.info("🔄 Simulation reset")

    def destroy(self):
        """
        Clean up all resources and destroy Genesis scene
        Note: This does NOT destroy the Genesis engine itself (gs.destroy())
        """
        logger.info("🗑️ Destroying simulation...")

        # Stop simulation
        self.stop()

        # Clear all data structures
        self.robots.clear()
        self.obstacles.clear()
        self.algorithms.clear()
        self.cameras.clear()
        self.websocket_clients.clear()

        # Destroy scene if it exists
        if self.scene is not None:
            try:
                # Delete scene object (Genesis handles cleanup internally)
                del self.scene
                self.scene = None
                logger.info("✅ Scene destroyed")
            except Exception as e:
                logger.warning(f"Scene destruction warning: {e}")
                self.scene = None

        # Reset frame capture
        self.last_frame = None
        self.last_frame_time = 0.0

        # Reset initialization flag (allows re-initialization of scene, not Genesis engine)
        self.is_initialized = False

        logger.info("✅ Simulation destroyed successfully")


# Global simulation manager (singleton)
_simulation: Optional[GenesisSimulation] = None


def get_simulation(config: Optional[GenesisConfig] = None) -> GenesisSimulation:
    """Get or create global simulation instance"""
    global _simulation

    if _simulation is None:
        if config is None:
            config = GenesisConfig()
        _simulation = GenesisSimulation(config)

    return _simulation


def reset_simulation():
    """Reset global simulation instance"""
    global _simulation

    if _simulation is not None:
        _simulation.destroy()
        _simulation = None
