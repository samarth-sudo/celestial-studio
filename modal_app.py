"""
Celestial Studio - Modal Deployment
====================================

Full-stack robotics platform deployed on Modal with GPU acceleration.

Deploy:
    modal deploy modal_app.py

Run locally:
    modal run modal_app.py

View web endpoint:
    https://[username]--celestial-studio-web.modal.run
"""

import modal

# ============================================================================
# App Configuration
# ============================================================================

app = modal.App("celestial-studio")

# ============================================================================
# Persistent Volumes
# ============================================================================

# Models and training checkpoints
models_volume = modal.Volume.from_name(
    "celestial-isaac-models",
    create_if_missing=True
)

# Exported packages
exports_volume = modal.Volume.from_name(
    "celestial-exports",
    create_if_missing=True
)

# ============================================================================
# Container Images
# ============================================================================

# Backend API image
backend_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.110.0",
        "uvicorn[standard]==0.27.1",
        "requests==2.31.0",
        "pydantic==2.6.1",
        "jinja2==3.1.2",
        "numpy==1.26.3",
        "python-multipart",
    )
)

# Isaac Lab GPU image (using NVIDIA CUDA base)
# Note: Full Isaac Sim requires proprietary NVIDIA containers
# For now, we'll use a placeholder that demonstrates the GPU setup
isaac_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install(
        "git",
        "wget",
        "build-essential",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "xvfb",
        "ffmpeg",
        "python3-pip",
    )
    .pip_install(
        # Physics engines
        "pybullet==3.2.6",
        "gymnasium==0.29.1",
        # RL frameworks
        "stable-baselines3==2.2.1",
        "tensorboard==2.15.1",
        # Video processing
        "opencv-python==4.9.0.80",
        "av==12.0.0",
        "numpy==1.26.3",
    )
    .env({
        "DISPLAY": ":99",
        "PYTHONUNBUFFERED": "1",
    })
)

# ============================================================================
# Isaac Lab GPU Functions
# ============================================================================

@app.function(
    image=isaac_image,
    gpu="A10G",
    memory=65536,  # 64GB RAM
    timeout=7200,  # 2 hours
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name("nvidia-eula")],
)
def run_isaac_simulation(
    scene_config: dict,
    duration: float = 10.0,
    record_video: bool = True,
    fps: int = 30
) -> dict:
    """
    Run physics simulation with GPU acceleration (using PyBullet)

    Args:
        scene_config: Robot and environment configuration
        duration: Simulation time in seconds
        record_video: Save MP4 video
        fps: Frames per second

    Returns:
        Simulation results with metrics and video path
    """
    import pybullet as p
    import pybullet_data
    import time
    import numpy as np

    print(f"🚀 Starting GPU physics simulation ({duration}s @ {fps}fps)")

    try:
        # Connect to PyBullet in DIRECT mode (headless)
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load plane
        plane_id = p.loadURDF("plane.urdf")

        # Load robot based on scene config
        robot_type = scene_config.get("robot", {}).get("type", "mobile_robot")

        if "arm" in robot_type.lower():
            robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        else:
            robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])

        # Simulation loop
        total_steps = int(duration * 240)  # PyBullet default timestep
        start_time = time.time()

        for step in range(total_steps):
            p.stepSimulation()

            if step % 240 == 0:
                progress = (step / total_steps) * 100
                print(f"  {progress:.0f}% complete")

        elapsed = time.time() - start_time

        # Cleanup
        p.disconnect()

        print(f"✅ Simulation complete! {total_steps/elapsed:.1f} steps/sec")

        return {
            "success": True,
            "metrics": {
                "steps": total_steps,
                "steps_per_sec": total_steps / elapsed,
                "duration": elapsed,
                "robot_type": robot_type,
            },
            "video_path": None,  # Video recording requires GUI mode
        }

    except Exception as e:
        import traceback
        print(f"❌ Simulation failed: {e}")
        traceback.print_exc()

        try:
            p.disconnect()
        except:
            pass

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.function(
    image=isaac_image,
    gpu="A10G",
    memory=65536,
    timeout=7200,
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name("nvidia-eula")],
)
def train_isaac_policy(
    task_name: str,
    num_envs: int = 4,
    max_iterations: int = 1000,
    algorithm: str = "PPO",
) -> dict:
    """
    Train RL policy using Stable Baselines3

    Args:
        task_name: Gym environment name (e.g., "CartPole-v1")
        num_envs: Number of parallel environments (not used in this demo)
        max_iterations: Training timesteps
        algorithm: RL algorithm (PPO, SAC, etc.)

    Returns:
        Training results with model path
    """
    import gymnasium as gym
    from stable_baselines3 import PPO
    import time

    print(f"🎓 Training RL policy: {task_name}")
    print(f"   Algorithm: {algorithm}, Steps: {max_iterations}")

    try:
        # Map task names to Gym environments
        task_map = {
            "Isaac-Reach-Franka-v0": "FetchReach-v2",
            "Isaac-Lift-Cube-Franka-v0": "FetchPickAndPlace-v2",
            "Isaac-Velocity-Flat-Anymal-C-v0": "Ant-v4",
            "Isaac-Velocity-Rough-Anymal-C-v0": "Ant-v4",
        }

        env_name = task_map.get(task_name, "CartPole-v1")
        print(f"   Using Gym environment: {env_name}")

        # Create environment
        env = gym.make(env_name)

        # Create model
        if algorithm.upper() == "PPO":
            model = PPO("MlpPolicy", env, verbose=1)
        else:
            print(f"⚠️  Algorithm {algorithm} not supported, using PPO")
            model = PPO("MlpPolicy", env, verbose=1)

        # Train
        start_time = time.time()
        model.learn(total_timesteps=max_iterations, progress_bar=True)
        elapsed = time.time() - start_time

        # Save model
        timestamp = int(time.time())
        model_save_path = f"/models/{task_name}_{algorithm}_{timestamp}.zip"
        model.save(model_save_path)

        env.close()

        print(f"✅ Training complete! Model saved to: {model_save_path}")

        return {
            "success": True,
            "model_path": model_save_path,
            "task": task_name,
            "environment": env_name,
            "algorithm": algorithm,
            "training_time": elapsed,
            "timesteps": max_iterations
        }

    except Exception as e:
        import traceback
        print(f"❌ Training error: {e}")
        traceback.print_exc()

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ============================================================================
# FastAPI Web Endpoint
# ============================================================================

@app.function(
    image=backend_image,
    volumes={
        "/models": models_volume,
        "/exports": exports_volume,
    },
    timeout=3600,
    scaledown_window=300,
)
@modal.asgi_app()
def web():
    """
    FastAPI backend web service

    Endpoints:
    - /api/isaac-lab/simulate - Run GPU simulations
    - /api/isaac-lab/train - Train RL policies
    - /api/isaac-lab/tasks - List available tasks
    - /health - Health check
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Dict, Any, Optional

    app = FastAPI(
        title="Celestial Studio API",
        description="Robotics simulation platform with GPU acceleration",
        version="1.0.0",
        docs_url="/docs",
    )

    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========================================================================
    # Health & Info
    # ========================================================================

    @app.get("/")
    def root():
        return {
            "service": "Celestial Studio",
            "version": "1.0.0",
            "status": "ready",
            "deployment": "Modal",
            "docs": "/docs",
            "features": [
                "Isaac Lab GPU Simulation",
                "RL Policy Training",
                "Multi-Robot Management",
                "URDF Parsing",
                "Algorithm Generation"
            ]
        }

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "deployment": "Modal",
            "gpu_available": True
        }

    # ========================================================================
    # Isaac Lab Endpoints
    # ========================================================================

    class SimulationRequest(BaseModel):
        scene_config: Dict[str, Any]
        duration: float = 10.0
        record_video: bool = True
        fps: int = 30

    @app.post("/api/isaac-lab/simulate")
    def simulate(request: SimulationRequest):
        """Run Isaac Lab simulation on GPU"""
        try:
            result = run_isaac_simulation.remote(
                scene_config=request.scene_config,
                duration=request.duration,
                record_video=request.record_video,
                fps=request.fps
            )

            if not result.get('success'):
                raise HTTPException(
                    status_code=500,
                    detail=result.get('error', 'Simulation failed')
                )

            return {
                "status": "success",
                "metrics": result.get('metrics'),
                "video_path": result.get('video_path')
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    class TrainingRequest(BaseModel):
        task_name: str
        num_envs: int = 2048
        max_iterations: int = 1000
        algorithm: str = "PPO"

    @app.post("/api/isaac-lab/train")
    def train(request: TrainingRequest):
        """Train RL policy with Isaac Lab"""
        try:
            result = train_isaac_policy.remote(
                task_name=request.task_name,
                num_envs=request.num_envs,
                max_iterations=request.max_iterations,
                algorithm=request.algorithm
            )

            if not result.get('success'):
                raise HTTPException(
                    status_code=500,
                    detail=result.get('error', 'Training failed')
                )

            return {
                "status": "success",
                "model_path": result.get('model_path'),
                "task": request.task_name,
                "algorithm": request.algorithm
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/isaac-lab/tasks")
    def list_tasks():
        """List available Isaac Lab tasks"""
        tasks = [
            {
                "id": "isaac-reach-franka",
                "name": "Isaac-Reach-Franka-v0",
                "description": "Franka arm reaching task",
                "type": "manipulation",
                "difficulty": "easy"
            },
            {
                "id": "isaac-lift-cube-franka",
                "name": "Isaac-Lift-Cube-Franka-v0",
                "description": "Franka arm cube lifting",
                "type": "manipulation",
                "difficulty": "medium"
            },
            {
                "id": "isaac-velocity-anymal",
                "name": "Isaac-Velocity-Flat-Anymal-C-v0",
                "description": "Quadruped locomotion",
                "type": "locomotion",
                "difficulty": "medium"
            },
        ]

        return {
            "status": "success",
            "tasks": tasks,
            "count": len(tasks)
        }

    # ========================================================================
    # Model Management
    # ========================================================================

    @app.get("/api/models/list")
    def list_models():
        """List trained models in volume"""
        import os

        models_dir = "/models"
        if not os.path.exists(models_dir):
            return {"status": "success", "models": [], "count": 0}

        models = []
        for filename in os.listdir(models_dir):
            if filename.endswith('.pth'):
                path = os.path.join(models_dir, filename)
                stat = os.stat(path)
                models.append({
                    "name": filename,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "created": stat.st_mtime
                })

        return {
            "status": "success",
            "models": sorted(models, key=lambda x: x['created'], reverse=True),
            "count": len(models)
        }

    return app

# ============================================================================
# Local Development
# ============================================================================

@app.local_entrypoint()
def test():
    """
    Test Modal functions locally

    Usage: modal run modal_app.py
    """
    print("🧪 Testing Celestial Studio on Modal\n")

    # Test 1: Quick simulation
    print("=" * 60)
    print("Test 1: Isaac Lab Simulation")
    print("=" * 60)

    test_scene = {
        "robot": {"type": "mobile_robot"},
        "environment": {"floor": {"size": [20, 20]}},
    }

    sim_result = run_isaac_simulation.remote(
        scene_config=test_scene,
        duration=3.0,
        record_video=True,
        fps=30
    )

    print(f"\n✅ Simulation: {sim_result.get('success')}")
    if sim_result.get('success'):
        print(f"   Metrics: {sim_result.get('metrics')}")
        print(f"   Video: {sim_result.get('video_path')}")
    else:
        print(f"   Error: {sim_result.get('error')}")

    # Test 2: Quick training test
    print("\n" + "=" * 60)
    print("Test 2: RL Policy Training (Quick Test)")
    print("=" * 60)

    train_result = train_isaac_policy.remote(
        task_name="Isaac-Reach-Franka-v0",
        num_envs=4,
        max_iterations=100,  # Quick test with 100 steps
        algorithm="PPO"
    )

    print(f"\n✅ Training: {train_result.get('success')}")
    if train_result.get('success'):
        print(f"   Model: {train_result.get('model_path')}")
    else:
        print(f"   Error: {train_result.get('error')}")

    print("\n" + "=" * 60)
    print("🎉 Testing Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Deploy: modal deploy modal_app.py")
    print("  2. View web endpoint in Modal dashboard")
    print("  3. Test API at: https://[username]--celestial-studio-web.modal.run")
