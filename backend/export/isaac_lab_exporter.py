"""
Isaac Lab Package Exporter

Exports simulation packages configured for Isaac Lab on Modal.
Includes scene configuration, training scripts, and deployment instructions.
"""

import os
import json
from typing import Dict, List, Any
from pathlib import Path


class IsaacLabExporter:
    """Export simulations for Isaac Lab on Modal"""

    def export(
        self,
        output_dir: str,
        scene_config: Dict[str, Any],
        algorithms: List[Dict[str, Any]],
        robots: List[Dict[str, Any]],
        project_name: str
    ):
        """
        Export Isaac Lab package

        Args:
            output_dir: Output directory path
            scene_config: Scene configuration
            algorithms: Algorithm list
            robots: Robot configurations
            project_name: Project name
        """
        print(f"Exporting Isaac Lab package to: {output_dir}")

        # Create directory structure
        self._create_directory_structure(output_dir, project_name)

        # Export scene configuration
        self._export_scene_config(output_dir, project_name, scene_config)

        # Export training scripts
        self._export_training_scripts(output_dir, project_name)

        # Export Modal configuration
        self._export_modal_config(output_dir, project_name)

        # Export README
        self._export_readme(output_dir, project_name, scene_config)

        # Export requirements
        self._export_requirements(output_dir)

        print(f"Isaac Lab package exported successfully")

    def _create_directory_structure(self, output_dir: str, project_name: str):
        """Create package directory structure"""
        dirs = [
            f"{project_name}",
            f"{project_name}/scenes",
            f"{project_name}/training",
            f"{project_name}/models",
            f"{project_name}/logs"
        ]

        for dir_path in dirs:
            os.makedirs(os.path.join(output_dir, dir_path), exist_ok=True)

    def _export_scene_config(self, output_dir: str, project_name: str, scene_config: Dict[str, Any]):
        """Export scene configuration in Isaac Lab format"""
        from isaac_lab.scene_converter import convert_scene_to_isaac

        # Convert to Isaac Lab format
        isaac_scene = convert_scene_to_isaac(scene_config)

        # Save scene configuration
        scene_path = os.path.join(output_dir, project_name, "scenes", "scene_config.json")
        with open(scene_path, 'w') as f:
            json.dump(isaac_scene, f, indent=2)

        print(f"Scene configuration exported: {scene_path}")

    def _export_training_scripts(self, output_dir: str, project_name: str):
        """Export training scripts"""

        # Train script
        train_script = '''"""
Training Script for Isaac Lab on Modal

Run this script to train your robot policy using reinforcement learning.
"""

import modal
from typing import Dict, Any

# Import from parent directory
import sys
sys.path.append('..')

# Create Modal stub
stub = modal.Stub(f"{project_name}-training")

# Isaac Lab container image
isaac_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/isaac-sim:2024.1.0",
        add_python="3.10"
    )
    .apt_install([
        "git", "wget", "build-essential", "libgl1-mesa-glx",
        "xvfb", "ffmpeg"
    ])
    .run_commands([
        "cd /tmp && git clone https://github.com/isaac-sim/IsaacLab.git",
        "cd /tmp/IsaacLab && git checkout v1.0.0",
        "cd /tmp/IsaacLab && pip install -e .",
        "pip install stable-baselines3==2.2.1",
        "pip install rl-games==1.6.1",
        "pip install rsl-rl==2.0.0",
    ])
    .env({"ACCEPT_EULA": "Y", "ISAAC_LAB_PATH": "/tmp/IsaacLab"})
)

# Modal volume for model storage
volume = modal.Volume.from_name(f"{project_name}-models", create_if_missing=True)

@stub.function(
    image=isaac_image,
    gpu=modal.gpu.A10G(),
    memory=65536,
    timeout=7200,
    volumes={"/models": volume}
)
def train_policy(
    task_name: str = "Isaac-Reach-Franka-v0",
    num_envs: int = 2048,
    max_iterations: int = 1000,
    algorithm: str = "PPO"
):
    """Train robot policy"""
    import subprocess
    import sys

    # Map algorithm to workflow
    workflow_map = {
        "PPO": "rl_games",
        "SAC": "sb3",
        "RSL": "rsl_rl"
    }
    workflow = workflow_map.get(algorithm, "rl_games")

    # Training command
    train_script = f"/tmp/IsaacLab/source/standalone/workflows/{workflow}/train.py"

    cmd = [
        "python", train_script,
        "--task", task_name,
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations),
        "--headless"
    ]

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Training completed successfully!")
        print(result.stdout)
    else:
        print("Training failed!")
        print(result.stderr)

    return {
        "success": result.returncode == 0,
        "logs": result.stdout
    }

@stub.local_entrypoint()
def main(
    task: str = "Isaac-Reach-Franka-v0",
    algorithm: str = "PPO",
    iterations: int = 1000
):
    """Local entrypoint for training"""
    print(f"Starting training: {task} with {algorithm}")
    result = train_policy.remote(
        task_name=task,
        algorithm=algorithm,
        max_iterations=iterations
    )
    print("Training result:", result)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
'''

        train_path = os.path.join(output_dir, project_name, "training", "train.py")
        with open(train_path, 'w') as f:
            f.write(train_script)

        print(f"Training script exported: {train_path}")

    def _export_modal_config(self, output_dir: str, project_name: str):
        """Export Modal configuration"""

        modal_config = '''"""
Modal Configuration

Configure your Modal account and deploy the training app.
"""

# To deploy this app to Modal:
# 1. Install Modal: pip install modal
# 2. Set up your Modal account: modal setup
# 3. Deploy the app: modal deploy training/train.py

# The app will be available at:
# https://modal.com/apps/your-username/{project_name}-training

# To run training from the command line:
# modal run training/train.py --task Isaac-Reach-Franka-v0 --algorithm PPO --iterations 1000

# To monitor training:
# modal logs {project_name}-training

# To download trained models:
# modal volume get {project_name}-models /models/model.pth ./model.pth
'''

        config_path = os.path.join(output_dir, project_name, "MODAL_SETUP.md")
        with open(config_path, 'w') as f:
            f.write(modal_config)

        print(f"Modal configuration exported: {config_path}")

    def _export_readme(self, output_dir: str, project_name: str, scene_config: Dict[str, Any]):
        """Export README with setup instructions"""

        robot_type = scene_config.get('robot', {}).get('type', 'robot')

        readme = f'''# {project_name} - Isaac Lab Training Package

Exported from Celestial Studio on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

This package contains everything needed to train your **{robot_type}** robot using NVIDIA Isaac Lab on Modal's cloud GPUs.

## What's Included

- `scenes/` - Scene configuration in Isaac Lab format
- `training/` - Training scripts for Modal
- `models/` - Directory for trained models
- `logs/` - Training logs

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Modal

```bash
# Install Modal CLI
pip install modal

# Set up your Modal account (one-time)
modal setup

# This will open a browser to authenticate with Modal
```

### 3. Deploy Training App

```bash
# Deploy the training app to Modal
modal deploy training/train.py
```

### 4. Run Training

```bash
# Start training on cloud GPUs
modal run training/train.py --task Isaac-Reach-Franka-v0 --algorithm PPO --iterations 1000
```

### 5. Monitor Training

```bash
# View real-time logs
modal logs {project_name}-training
```

### 6. Download Trained Model

```bash
# Download the trained model from Modal volume
modal volume get {project_name}-models /models/model.pth ./trained_model.pth
```

## Available Tasks

- **Isaac-Reach-Franka-v0** - Franka arm reaching (easy)
- **Isaac-Lift-Cube-Franka-v0** - Pick and place (medium)
- **Isaac-Velocity-Flat-Anymal-C-v0** - Quadruped navigation (medium)
- **Isaac-Velocity-Rough-Humanoid-v0** - Humanoid locomotion (hard)

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | Isaac-Reach-Franka-v0 | Isaac Lab task name |
| `--algorithm` | PPO | RL algorithm (PPO, SAC, RSL) |
| `--iterations` | 1000 | Number of training iterations |
| `--num_envs` | 2048 | Parallel environments |

## Cost Estimates

| Training Duration | Iterations | Est. Cost |
|-------------------|------------|-----------|
| 30 minutes | 1000 | $5-$10 |
| 1 hour | 2000 | $10-$20 |
| 2 hours | 5000 | $25-$50 |

*Based on Modal A10G GPU pricing*

## Troubleshooting

### Modal authentication issues
```bash
modal setup --force
```

### Training timeout
Increase timeout in `train.py`:
```python
@stub.function(..., timeout=14400)  # 4 hours
```

### Out of memory
Reduce number of parallel environments:
```bash
modal run training/train.py --num_envs 1024
```

## Support

- Modal Docs: https://modal.com/docs
- Isaac Lab Docs: https://isaac-sim.github.io/IsaacLab/
- Celestial Studio: Created this package

## Next Steps

1. Deploy the training app to Modal
2. Run a quick training test (100 iterations)
3. Monitor training progress
4. Download and test the trained model
5. Scale up to full training run

Good luck with your robot training!
'''

        readme_path = os.path.join(output_dir, project_name, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme)

        print(f"README exported: {readme_path}")

    def _export_requirements(self, output_dir: str):
        """Export requirements.txt"""

        requirements = '''# Isaac Lab Training Requirements

modal==0.63.0
fire==0.6.0
numpy==1.26.3
'''

        req_path = os.path.join(output_dir, "requirements.txt")
        with open(req_path, 'w') as f:
            f.write(requirements)

        print(f"Requirements exported: {req_path}")
