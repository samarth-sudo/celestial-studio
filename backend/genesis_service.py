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
    """Supported robot types"""
    MOBILE_ROBOT = "mobile"
    ROBOTIC_ARM = "arm"
    DRONE = "drone"
    FRANKA = "franka"
    GO2 = "go2"


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
    Handles scene creation, robot control, physics stepping, and rendering
    """

    # Path to local robot model assets
    ASSETS_DIR = Path(__file__).parent / "genesis_assets"

    def __init__(self, config: GenesisConfig):
        self.config = config
        self.scene: Optional[gs.Scene] = None
        self.robots: Dict[str, Any] = {}  # robot_id -> entity
        self.obstacles: Dict[str, Any] = {}
        self.algorithms: Dict[str, Callable] = {}  # robot_id -> algorithm function

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
        Discover all available robot models in the local assets directory

        Returns:
            Dictionary with categories (urdf, xml) and model information
        """
        models = {
            "urdf": [],
            "xml": [],
        }

        # Discover URDF models
        urdf_dir = cls.ASSETS_DIR / "urdf"
        if urdf_dir.exists():
            for model_dir in urdf_dir.iterdir():
                if model_dir.is_dir():
                    # Look for .urdf files
                    urdf_files = list(model_dir.rglob("*.urdf"))
                    for urdf_file in urdf_files:
                        models["urdf"].append({
                            "name": model_dir.name,
                            "path": str(urdf_file.relative_to(cls.ASSETS_DIR)),
                            "full_path": str(urdf_file),
                            "type": "urdf"
                        })

        # Discover MJCF/XML models
        xml_dir = cls.ASSETS_DIR / "xml"
        if xml_dir.exists():
            for model_dir in xml_dir.iterdir():
                if model_dir.is_dir():
                    # Look for .xml files
                    xml_files = list(model_dir.glob("*.xml"))
                    for xml_file in xml_files:
                        models["xml"].append({
                            "name": model_dir.name,
                            "path": str(xml_file.relative_to(cls.ASSETS_DIR)),
                            "full_path": str(xml_file),
                            "type": "mjcf"
                        })

        return models

    def add_robot(self, robot_id: str, robot_type: RobotType, position: Tuple[float, float, float] = (0, 0, 0.5)) -> Any:
        """Add a robot to the scene"""
        if not self.is_initialized:
            self.initialize()

        logger.info(f"Adding robot: {robot_id} ({robot_type.value}) at position {position}")

        robot = None

        if robot_type == RobotType.MOBILE_ROBOT:
            # Create simple mobile robot (box with 4 wheels)
            robot = self._create_mobile_robot(position)

        elif robot_type == RobotType.ROBOTIC_ARM:
            # Use Franka Panda arm if available
            robot = self._create_robotic_arm(position)

        elif robot_type == RobotType.DRONE:
            # Create quadcopter
            robot = self._create_drone(position)

        elif robot_type == RobotType.FRANKA:
            # Load Franka Panda from local MJCF assets
            try:
                franka_path = self.ASSETS_DIR / "xml" / "franka_emika_panda" / "panda.xml"
                if franka_path.exists():
                    robot = self.scene.add_entity(
                        gs.morphs.MJCF(
                            file=str(franka_path),
                            pos=position,
                        )
                    )
                    logger.info(f"✅ Loaded Franka Panda from {franka_path}")
                else:
                    logger.warning(f"Franka model not found at {franka_path}, falling back to simple arm")
                    robot = self._create_robotic_arm(position)
            except Exception as e:
                logger.warning(f"Could not load Franka model: {e}, falling back to simple arm")
                robot = self._create_robotic_arm(position)

        elif robot_type == RobotType.GO2:
            # Load Unitree Go2 quadruped from local URDF assets
            try:
                go2_path = self.ASSETS_DIR / "urdf" / "go2" / "urdf" / "go2.urdf"
                if go2_path.exists():
                    robot = self.scene.add_entity(
                        gs.morphs.URDF(
                            file=str(go2_path),
                            pos=position,
                        )
                    )
                    logger.info(f"✅ Loaded Go2 quadruped from {go2_path}")
                else:
                    logger.warning(f"Go2 model not found at {go2_path}")
                    robot = None
            except Exception as e:
                logger.warning(f"Could not load Go2 model: {e}")
                robot = None

        if robot is not None:
            self.robots[robot_id] = robot
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
        """Create a quadcopter drone (Crazyflie)"""
        # Try to load Crazyflie drone from local assets
        try:
            drone_path = self.ASSETS_DIR / "urdf" / "drones" / "cf2x.urdf"
            if drone_path.exists():
                robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=str(drone_path),
                        pos=position,
                    )
                )
                logger.info(f"✅ Loaded Crazyflie drone from {drone_path}")
                return robot
        except Exception as e:
            logger.warning(f"Could not load drone model: {e}, falling back to simple sphere")

        # Fallback to simple sphere
        robot = self.scene.add_entity(
            gs.morphs.Sphere(
                pos=position,
                radius=0.2,
                fixed=False,
            )
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

    def set_algorithm(self, robot_id: str, algorithm: Callable, params: Dict[str, Any] = None):
        """Set algorithm for a robot"""
        if robot_id not in self.robots:
            logger.error(f"Robot {robot_id} not found")
            return

        self.algorithms[robot_id] = algorithm
        logger.info(f"Algorithm set for robot {robot_id}")

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
        for robot_id, algorithm in self.algorithms.items():
            if robot_id in self.robots:
                try:
                    # Get robot state
                    robot = self.robots[robot_id]
                    # TODO: Extract actual state from Genesis entity

                    # Execute algorithm
                    # TODO: Apply algorithm output to robot
                    pass
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
                state['position'] = pos.tolist() if isinstance(pos, np.ndarray) else list(pos)
            else:
                state['position'] = [0, 0, 0]

            # Get velocity
            if hasattr(robot, 'get_vel'):
                vel = robot.get_vel()
                state['velocity'] = vel.tolist() if isinstance(vel, np.ndarray) else list(vel)
            else:
                state['velocity'] = [0, 0, 0]

            # Get quaternion
            if hasattr(robot, 'get_quat'):
                quat = robot.get_quat()
                state['quaternion'] = quat.tolist() if isinstance(quat, np.ndarray) else list(quat)
            else:
                state['quaternion'] = [0, 0, 0, 1]

            # Get angular velocity
            if hasattr(robot, 'get_ang'):
                ang = robot.get_ang()
                state['angular_velocity'] = ang.tolist() if isinstance(ang, np.ndarray) else list(ang)
            else:
                state['angular_velocity'] = [0, 0, 0]

            # Get DOF states if available
            if hasattr(robot, 'get_dofs_position') and hasattr(robot, 'n_dofs'):
                n_dofs = robot.n_dofs
                dofs_idx = list(range(n_dofs))
                joint_pos = robot.get_dofs_position(dofs_idx)
                state['joint_positions'] = joint_pos.tolist() if isinstance(joint_pos, np.ndarray) else list(joint_pos)

                if hasattr(robot, 'get_dofs_velocity'):
                    joint_vel = robot.get_dofs_velocity(dofs_idx)
                    state['joint_velocities'] = joint_vel.tolist() if isinstance(joint_vel, np.ndarray) else list(joint_vel)

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
