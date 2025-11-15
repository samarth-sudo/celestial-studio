"""
Modal Isaac Lab Runner

Runs Isaac Lab simulations and training on Modal's cloud infrastructure with GPU acceleration.
"""

import modal
from typing import Dict, Any, Optional, List
import json

# Create Modal stub
stub = modal.Stub("celestial-isaac-lab")

# Create Modal volume for persistent storage (models, recordings, etc.)
volume = modal.Volume.from_name("celestial-isaac-models", create_if_missing=True)

# Isaac Lab container image
# Using NVIDIA Isaac Sim base with Isaac Lab installed
isaac_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/isaac-sim:2024.1.0",
        add_python="3.10"
    )
    .apt_install([
        "git",
        "wget",
        "build-essential",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "xvfb",  # Virtual display
        "ffmpeg",  # Video encoding
    ])
    .run_commands([
        # Clone Isaac Lab
        "cd /tmp && git clone https://github.com/isaac-sim/IsaacLab.git",
        "cd /tmp/IsaacLab && git checkout v1.0.0",

        # Install Isaac Lab
        "cd /tmp/IsaacLab && pip install -e .",

        # Install RL frameworks
        "pip install stable-baselines3==2.2.1",
        "pip install rl-games==1.6.1",
        "pip install rsl-rl==2.0.0",
        "pip install tensorboard==2.15.1",

        # Install video/streaming tools
        "pip install opencv-python==4.9.0.80",
        "pip install av==12.0.0",
        "pip install aiortc==1.9.0",
    ])
    .env({
        "ACCEPT_EULA": "Y",  # Accept NVIDIA EULA
        "ISAAC_LAB_PATH": "/tmp/IsaacLab",
    })
)


@stub.function(
    image=isaac_image,
    gpu=modal.gpu.A10G(),  # NVIDIA A10G GPU
    memory=65536,  # 64GB RAM
    timeout=7200,  # 2 hour timeout
    volumes={"/models": volume},
    env={
        "CUDA_VISIBLE_DEVICES": "0",
        "DISPLAY": ":99",
    }
)
def run_isaac_simulation(
    scene_config: Dict[str, Any],
    duration: float = 10.0,
    record_video: bool = True,
    headless: bool = True,
    fps: int = 30
) -> Dict[str, Any]:
    """
    Run Isaac Lab simulation from scene configuration

    Args:
        scene_config: Scene configuration (from frontend)
        duration: Simulation duration in seconds
        record_video: Whether to record video
        headless: Run without GUI
        fps: Target frames per second

    Returns:
        Dict with simulation results, metrics, and paths
    """
    import subprocess
    import sys
    import time
    import numpy as np
    from pathlib import Path

    print(f"Starting Isaac Lab simulation (duration: {duration}s, fps: {fps})")

    # Start virtual display for rendering (even in headless mode)
    if headless:
        print("Starting Xvfb virtual display...")
        subprocess.Popen([
            "Xvfb", ":99",
            "-screen", "0", "1920x1080x24",
            "-ac", "+extension", "GLX", "+render", "-noreset"
        ])
        time.sleep(2)  # Wait for display to initialize

    # Add Isaac Lab to path
    sys.path.append("/tmp/IsaacLab")

    try:
        # Import Isaac Lab (must be after setting up display)
        from omni.isaac.lab.app import AppLauncher

        # Launch Isaac Sim application
        print("Launching Isaac Sim...")
        app_launcher = AppLauncher(headless=headless)
        simulation_app = app_launcher.app

        # Now import Isaac Lab modules (must be after app launch)
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.scene import InteractiveScene
        from omni.isaac.lab.utils import convert_dict_to_backend

        # Convert scene config to Isaac Lab format
        print("Converting scene configuration...")
        isaac_scene = _convert_scene_config_to_isaac(scene_config)

        # Create simulation context
        print("Creating simulation context...")
        sim_cfg = sim_utils.SimulationCfg(dt=1.0/fps, substeps=1)
        sim = sim_utils.SimulationContext(sim_cfg)

        # Set up the scene
        scene = InteractiveScene(isaac_scene)

        # Initialize video recorder if requested
        video_path = None
        if record_video:
            import cv2
            video_path = "/models/simulation.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (1920, 1080))

        # Run simulation loop
        print(f"Running simulation for {duration} seconds...")
        total_frames = int(duration * fps)
        metrics = {
            "frames_simulated": 0,
            "average_fps": 0,
            "physics_steps": 0
        }

        start_time = time.time()

        for frame in range(total_frames):
            # Step physics
            sim.step()
            metrics["physics_steps"] += 1

            # Capture frame for video
            if record_video and frame % (60 // fps) == 0:
                # Get viewport image
                viewport = sim.render()
                if viewport is not None:
                    video_writer.write(viewport)

            metrics["frames_simulated"] = frame + 1

            # Progress update
            if frame % fps == 0:
                elapsed = time.time() - start_time
                progress = (frame / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame}/{total_frames} frames, {elapsed:.1f}s elapsed)")

        # Finalize
        if record_video:
            video_writer.release()
            print(f"Video saved to {video_path}")

        elapsed_time = time.time() - start_time
        metrics["average_fps"] = total_frames / elapsed_time

        print(f"Simulation complete! Avg FPS: {metrics['average_fps']:.1f}")

        # Clean up
        simulation_app.close()

        return {
            "success": True,
            "metrics": metrics,
            "video_path": video_path if record_video else None,
            "duration": elapsed_time
        }

    except Exception as e:
        print(f"Simulation error: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@stub.function(
    image=isaac_image,
    gpu=modal.gpu.A10G(),
    memory=65536,
    timeout=7200,
    volumes={"/models": volume},
    env={"CUDA_VISIBLE_DEVICES": "0", "DISPLAY": ":99"}
)
def train_isaac_policy(
    task_name: str,
    num_envs: int = 2048,
    max_iterations: int = 1000,
    algorithm: str = "PPO",
    headless: bool = True
) -> Dict[str, Any]:
    """
    Train robot policy using Isaac Lab with RL

    Args:
        task_name: Isaac Lab task name (e.g., "Isaac-Reach-Franka-v0")
        num_envs: Number of parallel environments
        max_iterations: Maximum training iterations
        algorithm: RL algorithm ("PPO", "SAC", "RSL")
        headless: Run without GUI

    Returns:
        Dict with training results and model path
    """
    import subprocess
    import sys
    import time
    from pathlib import Path

    print(f"Starting Isaac Lab training: {task_name}")
    print(f"Config: {num_envs} envs, {max_iterations} iterations, {algorithm}")

    # Start virtual display
    if headless:
        print("Starting Xvfb...")
        subprocess.Popen([
            "Xvfb", ":99",
            "-screen", "0", "1024x768x24",
            "-ac", "+extension", "GLX", "+render", "-noreset"
        ])
        time.sleep(2)

    # Add Isaac Lab to path
    sys.path.append("/tmp/IsaacLab")

    try:
        # Map algorithm to training workflow
        workflow_map = {
            "PPO": "rl_games",
            "SAC": "sb3",
            "RSL": "rsl_rl"
        }
        workflow = workflow_map.get(algorithm, "rl_games")

        # Training command
        train_script = f"/tmp/IsaacLab/source/standalone/workflows/{workflow}/train.py"

        cmd = [
            "python",
            train_script,
            "--task", task_name,
            "--num_envs", str(num_envs),
            "--max_iterations", str(max_iterations),
            "--headless" if headless else ""
        ]

        print(f"Running command: {' '.join(cmd)}")

        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7000  # Slightly less than function timeout
        )

        if result.returncode != 0:
            print(f"Training failed with code {result.returncode}")
            print("STDERR:", result.stderr)
            return {
                "success": False,
                "error": result.stderr,
                "logs": result.stdout
            }

        # Find trained model
        log_dir = Path(f"/tmp/IsaacLab/logs/{workflow}")
        model_files = list(log_dir.glob("**/model.pth"))

        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

            # Copy to persistent volume
            import shutil
            model_save_path = f"/models/{task_name}_trained.pth"
            shutil.copy(latest_model, model_save_path)
            print(f"Model saved to {model_save_path}")

            return {
                "success": True,
                "model_path": model_save_path,
                "logs": result.stdout,
                "task": task_name,
                "algorithm": algorithm
            }
        else:
            return {
                "success": False,
                "error": "No model file found after training",
                "logs": result.stdout
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Training timeout (2 hours exceeded)"
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def _convert_scene_config_to_isaac(scene_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Celestial Studio scene config to Isaac Lab format

    Uses the SceneConverter module for proper conversion
    """
    from .scene_converter import convert_scene_to_isaac

    return convert_scene_to_isaac(scene_config)


@stub.local_entrypoint()
def main():
    """
    Local entrypoint for testing
    """
    # Test simulation
    test_scene = {
        "robot": {"type": "mobile_robot"},
        "environment": {"floor": {"size": [20, 20]}},
        "objects": []
    }

    print("Testing Isaac Lab simulation...")
    result = run_isaac_simulation.remote(test_scene, duration=5.0)
    print("Result:", json.dumps(result, indent=2))
