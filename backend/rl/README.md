# 🤖 Genesis RL Training System

Reinforcement learning training infrastructure for mobile robot navigation using Genesis physics engine and PPO algorithm.

---

## 📁 Directory Structure

```
backend/rl/
├── envs/
│   ├── __init__.py
│   └── mobile_robot_env.py      # Mobile robot navigation environment
├── policies/                     # (Future) Custom policy networks
├── utils/                        # (Future) Trajectory loading, rewards
├── mobile_nav_trainer.py        # Main training script
├── test_env.py                  # Environment validation script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/samarth/Desktop/idea/backend/rl

# Install RL training dependencies
pip3 install -r requirements.txt
```

**Important:** Make sure to install `rsl-rl-lib==2.2.4`, **NOT** `rsl-rl` (they are different packages)

### 2. Verify Installation

```bash
# Quick import test (no scene building)
python3 -c "from envs.mobile_robot_env import MobileRobotEnv; print('✅ Environment ready')"
```

### 3. Run Training

```bash
# Small test run (64 envs, 100 iterations)
python3 mobile_nav_trainer.py --num_envs 64 --max_iterations 100

# Full training run (1024 envs, 1000 iterations)
python3 mobile_nav_trainer.py --num_envs 1024 --max_iterations 1000

# With visualization (slower)
python3 mobile_nav_trainer.py --num_envs 64 --max_iterations 100 --show_viewer
```

### 4. Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir logs/

# View at http://localhost:6006
```

---

## 🎯 Mobile Robot Environment

### Observation Space (9 dimensions)
- **Base velocity** (3): linear_x, linear_y, angular_z
- **Goal position** (2): relative goal position in x, y
- **Distance to goal** (1): Euclidean distance
- **Last actions** (3): previous action for smoothness

### Action Space (3 dimensions)
- **vx**: Linear velocity in x (forward/backward)
- **vy**: Linear velocity in y (left/right)
- **omega**: Angular velocity (rotation)

### Reward Function
```python
reward = (
    1.0 * goal_tracking +          # Exponential reward towards goal
    10.0 * goal_reached +           # Bonus for reaching goal
    -0.01 * action_smoothness +     # Penalize jerky motion
    -0.001 * linear_velocity +      # Penalize high speed
    -0.005 * angular_velocity +     # Penalize spinning
    -0.001 * action_magnitude +     # Energy efficiency
    0.1 * orientation               # Bonus for facing goal
)
```

### Episode Termination
- **Success**: Robot reaches within 0.3m of goal
- **Timeout**: 20 seconds elapsed
- **Collision**: Robot height below 5cm

---

## 📊 Training Configuration

### PPO Hyperparameters
```python
{
    "clip_param": 0.2,
    "desired_kl": 0.01,
    "entropy_coef": 0.01,
    "gamma": 0.99,
    "lam": 0.95,
    "learning_rate": 0.001,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
}
```

### Policy Network
```python
{
    "actor_hidden_dims": [256, 128, 64],
    "critic_hidden_dims": [256, 128, 64],
    "activation": "elu",
}
```

---

## 🎓 Architecture Details

### Based on Genesis go2_env.py
Our `MobileRobotEnv` is a simplified adaptation of Genesis quadruped locomotion environment:

**Simplifications:**
- 12 DOF → 3 DOF (mobile base instead of legs)
- Joint positions → Direct velocity control
- Complex rewards → Navigation-specific rewards
- 45 observations → 9 observations

**Preserved:**
- Parallel environment structure (GPU-accelerated)
- Reward composition pattern
- Reset logic and termination handling
- Genesis scene building approach

---

## 📈 Performance Expectations

| Metric | Value |
|--------|-------|
| Training Speed | ~10K steps/sec (1024 envs) |
| Sample Efficiency | ~5-10M steps to convergence |
| Success Rate | >80% on navigation task |
| Policy Inference | <5ms real-time control |

---

## 🔧 Customization

### Modify Reward Weights
Edit `get_cfgs()` in `mobile_nav_trainer.py`:

```python
reward_cfg = {
    "reward_scales": {
        "goal_tracking": 2.0,     # Increase goal attraction
        "goal_reached": 20.0,     # Bigger bonus
        # ... other rewards
    },
}
```

### Change Goal Distribution
```python
command_cfg = {
    "goal_x_range": [-10.0, 10.0],  # Wider range
    "goal_y_range": [-10.0, 10.0],
}
```

### Adjust Action Limits
```python
env_cfg = {
    "action_scale_linear": 2.0,   # Max 2 m/s
    "action_scale_angular": 3.0,  # Max 3 rad/s
}
```

---

## 🔗 Integration with Teleoperation

### Load Trained Policy
```python
import torch
from envs.mobile_robot_env import MobileRobotEnv

# Load checkpoint
policy = torch.load("logs/mobile-nav/model_1000.pt")

# Run inference
obs = env.reset()
action = policy(obs)
```

### Connect to Command Executor
```python
from chat.command_executor import CommandExecutor

# Set trained policy
executor.set_policy(policy)

# Execute autonomous navigation
executor.execute_autonomous(task="navigate_to_goal", goal=[2.0, 3.0])
```

---

## 📚 Next Steps

### Phase 1: Basic Training (Current)
- [x] Mobile robot navigation environment
- [x] PPO training script
- [ ] Run first training (requires rsl-rl-lib)
- [ ] Validate convergence

### Phase 2: Trajectory Integration
- [ ] Create `utils/trajectory_loader.py`
- [ ] Load CSV recordings from teleoperation
- [ ] Implement behavior cloning
- [ ] Hybrid training (BC + RL)

### Phase 3: Advanced Features
- [ ] Robotic arm manipulation environment
- [ ] Drone flight environment
- [ ] Multi-robot coordination
- [ ] Vision-based observations

---

## 🐛 Troubleshooting

### Import Error: No module named 'rsl_rl'
**Solution:** Install `rsl-rl-lib==2.2.4`:
```bash
pip3 uninstall rsl-rl  # Remove if installed
pip3 install rsl-rl-lib==2.2.4
```

### Genesis scene building hangs
**Solution:** Genesis initialization can take 30-60 seconds on first run. Be patient!

### CUDA out of memory
**Solution:** Reduce number of parallel environments:
```bash
python3 mobile_nav_trainer.py --num_envs 128  # Instead of 1024
```

---

## 📖 References

- **Genesis Documentation:** https://genesis-world.readthedocs.io/
- **Genesis Locomotion Example:** `Genesis/examples/locomotion/go2_train.py`
- **rsl-rl-lib:** https://github.com/leggedrobotics/rsl_rl
- **PPO Paper:** https://arxiv.org/abs/1707.06347

---

## 🏆 Key Features

✅ **GPU-Accelerated Physics** - Genesis 43M FPS simulation
✅ **Parallel Environments** - Train on 1024 envs simultaneously
✅ **Vectorized Operations** - All operations batched on GPU
✅ **Modular Design** - Easy to extend to new robot types
✅ **Compatible with Teleoperation** - Integrates with existing system
✅ **Production-Ready** - Type hints, docstrings, error handling

---

*Created: November 26, 2025*
*Status: Ready for Training*
