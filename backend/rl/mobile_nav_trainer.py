"""
Mobile Robot Navigation Trainer

Trains a mobile robot to navigate to goal positions using PPO.
Adapted from Genesis locomotion/go2_train.py for 2D navigation tasks.

Usage:
    python mobile_nav_trainer.py --num_envs 256 --max_iterations 1000
"""

import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from envs.mobile_robot_env import MobileRobotEnv


def get_train_cfg(exp_name, max_iterations):
    """
    PPO training configuration.

    Based on Genesis go2 locomotion config but adapted for navigation.
    """
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64],  # Smaller network for simpler task
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    """
    Environment, observation, reward, and command configurations.

    Simplified from go2 for mobile robot navigation:
    - 3 actions: [vx, vy, omega]
    - 9 observations: velocity (3) + goal relative (2) + distance (1) + actions (3)
    - Navigation rewards: goal tracking, smoothness, energy
    - Goal commands: [goal_x, goal_y]
    """
    env_cfg = {
        "num_actions": 3,  # vx, vy, omega
        # base pose
        "base_init_pos": [0.0, 0.0, 0.1],  # Start at origin, 10cm above ground
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],  # No rotation
        "episode_length_s": 20.0,
        "resampling_time_s": 10.0,  # Resample goal every 10s
        "action_scale_linear": 1.0,  # Max 1 m/s linear velocity
        "action_scale_angular": 2.0,  # Max 2 rad/s angular velocity
        "clip_actions": 1.0,
        "goal_threshold": 0.3,  # Goal reached if within 30cm
        "min_height": 0.05,  # Collision if below 5cm
        "init_pos_noise": 0.5,  # Randomize start position ±0.5m
    }

    obs_cfg = {
        "num_obs": 9,  # velocity(3) + goal_rel(2) + distance(1) + actions(3)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "goal_pos": 0.5,
            "goal_dist": 0.2,
        },
    }

    reward_cfg = {
        "goal_tracking_sigma": 1.0,  # Exponential decay distance
        "reward_scales": {
            "goal_tracking": 1.0,  # Main reward: move towards goal
            "goal_reached": 10.0,  # Bonus for reaching goal
            "action_smoothness": -0.01,  # Penalize jerky motion
            "linear_velocity": -0.001,  # Small penalty for high speed
            "angular_velocity": -0.005,  # Penalize spinning
            "action_magnitude": -0.001,  # Energy efficiency
            "orientation": 0.1,  # Bonus for facing goal
        },
    }

    command_cfg = {
        "num_commands": 2,  # goal_x, goal_y
        "goal_x_range": [-5.0, 5.0],  # Goals within ±5m in x
        "goal_y_range": [-5.0, 5.0],  # Goals within ±5m in y
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="mobile-nav")
    parser.add_argument("-B", "--num_envs", type=int, default=256)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--show_viewer", action="store_true", help="Show Genesis viewer")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Clear old logs
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Save configs
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    print("=" * 80)
    print(f"🤖 Mobile Robot Navigation Training")
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print(f"Num Envs: {args.num_envs}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Log Dir: {log_dir}")
    print("=" * 80)

    # Create environment
    env = MobileRobotEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
    )

    print(f"✅ Environment created:")
    print(f"   - Observations: {env.num_obs}")
    print(f"   - Actions: {env.num_actions}")
    print(f"   - Parallel envs: {env.num_envs}")
    print("=" * 80)

    # Create PPO runner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    print(f"🚀 Starting training...")
    print(f"   - Monitor with: tensorboard --logdir {log_dir}")
    print("=" * 80)

    # Train
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print("=" * 80)
    print(f"✅ Training complete!")
    print(f"   - Logs saved to: {log_dir}")
    print(f"   - Policy saved to: {log_dir}/model_*.pt")
    print("=" * 80)


if __name__ == "__main__":
    main()


"""
# Quick start - Small test run
python mobile_nav_trainer.py --num_envs 64 --max_iterations 100

# Full training run
python mobile_nav_trainer.py --num_envs 1024 --max_iterations 1000

# With visualization (slower)
python mobile_nav_trainer.py --num_envs 64 --max_iterations 100 --show_viewer

# Custom experiment name
python mobile_nav_trainer.py -e my-experiment --num_envs 256 --max_iterations 500
"""
