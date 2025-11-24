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

    def initialize(self):
        """Initialize Genesis engine"""
        if not GENESIS_AVAILABLE:
            raise RuntimeError("Genesis not installed. Install with: pip install genesis-world")

        logger.info(f"Initializing Genesis with {self.config.backend.value} backend...")

        # Map backend enum to Genesis backend
        backend_map = {
            BackendType.METAL: gs.metal,
            BackendType.CUDA: gs.cuda,
            BackendType.CPU: gs.cpu,
            BackendType.VULKAN: gs.vulkan,
        }

        backend = backend_map.get(self.config.backend, gs.cpu)

        # Initialize Genesis
        gs.init(
            backend=backend,
            precision=self.config.precision,
            logging_level=self.config.logging_level,
        )

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

        self.is_initialized = True
        logger.info("✅ Genesis initialized successfully")

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
            # Load Franka Panda from MJCF
            try:
                robot = self.scene.add_entity(
                    gs.morphs.MJCF(
                        file="xml/franka_emika_panda/panda.xml",
                        pos=position,
                    )
                )
            except Exception as e:
                logger.warning(f"Could not load Franka model: {e}, falling back to simple arm")
                robot = self._create_robotic_arm(position)

        elif robot_type == RobotType.GO2:
            # Load Unitree Go2 quadruped
            try:
                robot = self.scene.add_entity(
                    gs.morphs.MJCF(
                        file="xml/unitree_go2/go2.xml",
                        pos=position,
                    )
                )
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
        """Create a quadcopter drone"""
        # TODO: Create proper URDF with propellers
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
        """Capture current frame from Genesis viewer"""
        try:
            # Get frame from scene
            # Note: This requires Genesis viewer to be running (even headless)
            if hasattr(self.scene, 'viewer') and self.scene.viewer is not None:
                frame = self.scene.viewer.render()
                return frame
            else:
                logger.warning("Viewer not available for frame capture")
                return None
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
        for robot_id, robot in self.robots.items():
            try:
                # TODO: Extract actual state from Genesis entity
                # For now, return placeholder
                state['robots'][robot_id] = {
                    'position': [0, 0, 0],
                    'orientation': [0, 0, 0, 1],
                    'velocity': [0, 0, 0],
                }
            except Exception as e:
                logger.error(f"Failed to get state for {robot_id}: {e}")

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

    def reset(self):
        """Reset simulation to initial state"""
        self.step_count = 0
        self.start_time = time.time()
        # TODO: Reset scene state
        logger.info("🔄 Simulation reset")

    def destroy(self):
        """Clean up resources"""
        self.stop()
        self.robots.clear()
        self.obstacles.clear()
        self.algorithms.clear()
        self.websocket_clients.clear()

        if self.scene is not None:
            # TODO: Properly destroy Genesis scene
            self.scene = None

        logger.info("🗑️  Simulation destroyed")


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
