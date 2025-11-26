"""
Quick test script to verify environment setup.

Tests that the mobile robot environment can be created and stepped through.
"""

import torch
import genesis as gs
from envs.mobile_robot_env import MobileRobotEnv


def get_test_cfgs():
    """Minimal test configs"""
    env_cfg = {
        "num_actions": 3,
        "base_init_pos": [0.0, 0.0, 0.1],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 5.0,
        "resampling_time_s": 10.0,
        "action_scale_linear": 1.0,
        "action_scale_angular": 2.0,
        "clip_actions": 1.0,
        "goal_threshold": 0.3,
        "min_height": 0.05,
        "init_pos_noise": 0.5,
    }

    obs_cfg = {
        "num_obs": 9,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "goal_pos": 0.5,
            "goal_dist": 0.2,
        },
    }

    reward_cfg = {
        "goal_tracking_sigma": 1.0,
        "reward_scales": {
            "goal_tracking": 1.0,
            "goal_reached": 10.0,
            "action_smoothness": -0.01,
            "linear_velocity": -0.001,
            "angular_velocity": -0.005,
            "action_magnitude": -0.001,
            "orientation": 0.1,
        },
    }

    command_cfg = {
        "num_commands": 2,
        "goal_x_range": [-5.0, 5.0],
        "goal_y_range": [-5.0, 5.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    print("=" * 80)
    print("🧪 Testing Mobile Robot Environment")
    print("=" * 80)

    # Initialize Genesis
    gs.init(logging_level="warning")
    print("✅ Genesis initialized")

    # Get configs
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_test_cfgs()
    print("✅ Configs loaded")

    # Create environment with small number of envs for testing
    num_envs = 4
    print(f"\n📦 Creating environment with {num_envs} parallel envs...")

    env = MobileRobotEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )
    print(f"✅ Environment created")
    print(f"   - Observations: {env.num_obs}")
    print(f"   - Actions: {env.num_actions}")
    print(f"   - Device: {env.device}")

    # Reset environment
    print("\n🔄 Resetting environment...")
    obs, _ = env.reset()
    print(f"✅ Reset successful")
    print(f"   - Obs shape: {obs.shape}")
    print(f"   - Obs dtype: {obs.dtype}")

    # Run a few steps
    print("\n🏃 Running 10 test steps...")
    for i in range(10):
        # Random actions
        actions = torch.randn((num_envs, env.num_actions), device=env.device)
        obs, rewards, dones, extras = env.step(actions)

        print(f"   Step {i+1}:")
        print(f"      Obs: {obs[0, :3].cpu().numpy()}")  # First 3 obs values
        print(f"      Reward: {rewards[0].item():.4f}")
        print(f"      Done: {dones[0].item()}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("   1. Install rsl-rl-lib: pip3 install rsl-rl-lib==2.2.4")
    print("   2. Run training: python mobile_nav_trainer.py --num_envs 64 --max_iterations 100")
    print("=" * 80)


if __name__ == "__main__":
    main()
