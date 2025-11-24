# 💬 Chat Interface Usage Examples

The Celestial Studio chat interface allows you to create simulations and generate algorithms using natural language. Here are practical examples:

---

## 🤖 Example 1: Creating a Simple Robot Simulation

### User Input:
```
Create a simulation with a Franka Panda robot arm that needs to pick up a red cube from position (1, 1, 0.5)
```

### System Response:
The AI will:
1. Parse your request and identify:
   - Robot type: Franka Panda (manipulator)
   - Task: Pick and place
   - Object: Red cube at (1, 1, 0.5)

2. Create the scene configuration automatically
3. Generate a pick-and-place algorithm using inverse kinematics
4. Set up the Genesis simulation with your robot

### What You'll See:
- Genesis viewer showing Franka Panda arm
- Red cube object in the scene
- Real-time simulation running at 60 FPS
- Algorithm executing the pick-and-place task

---

## 🏃 Example 2: Locomotion Task

### User Input:
```
I want a Go2 quadruped robot to walk to position (5, 0, 0) while avoiding obstacles
```

### System Response:
The AI will:
1. Initialize Genesis with Go2 URDF model
2. Generate a path planning algorithm (A* or RRT)
3. Create a locomotion controller for the quadruped
4. Add obstacle avoidance logic
5. Run the simulation in Genesis

### Generated Algorithm (Python):
```python
import numpy as np
from typing import List, Dict

# Configuration
MAX_SPEED = 1.5  # m/s
GOAL_POSITION = np.array([5.0, 0.0, 0.0])
OBSTACLE_RADIUS = 0.5

def find_path(start: np.ndarray, goal: np.ndarray,
              obstacles: List[Dict]) -> List[np.ndarray]:
    """
    A* path planning for quadruped locomotion
    """
    # Implementation by Qwen 2.5 Coder
    path = a_star_search(start, goal, obstacles)
    return smooth_path(path)

def locomotion_controller(robot_state: Dict,
                         target_waypoint: np.ndarray) -> Dict:
    """
    Generate leg joint commands to reach target waypoint
    """
    current_pos = robot_state['position']
    velocity = calculate_velocity(current_pos, target_waypoint)

    return {
        'joint_commands': inverse_kinematics(velocity),
        'target_speed': MAX_SPEED
    }
```

---

## 🚁 Example 3: Multi-Drone Coordination

### User Input:
```
Create a swarm of 5 Crazyflie drones that form a circle pattern in the air
```

### System Response:
The AI will:
1. Add 5 Crazyflie drone URDFs to the scene
2. Generate formation control algorithm
3. Implement PID controllers for position control
4. Run parallel simulation (Genesis can handle 1000+ robots!)

### What Happens:
- 5 drones spawn at different positions
- Each drone flies to its assigned position in the circle
- Formation is maintained using distributed control
- Real-time video stream shows the swarm behavior

---

## 🧩 Example 4: Custom Algorithm Generation

### User Input:
```
Generate a PID controller algorithm for a mobile robot to follow a sine wave trajectory
```

### System Response:
```python
import numpy as np
from typing import Dict

# PID Gains (tunable via UI)
KP = 2.0
KI = 0.1
KD = 0.5

class PIDController:
    def __init__(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.time = 0.0

    def compute(self, current_pos: np.ndarray,
                target_pos: np.ndarray, dt: float) -> np.ndarray:
        """
        PID control for trajectory tracking
        """
        error = target_pos - current_pos

        # Proportional
        p_term = KP * error

        # Integral
        self.integral += error * dt
        i_term = KI * self.integral

        # Derivative
        d_term = KD * (error - self.prev_error) / dt
        self.prev_error = error

        # Control output
        control = p_term + i_term + d_term
        return control

def sine_trajectory(t: float, amplitude: float = 2.0,
                   frequency: float = 0.5) -> np.ndarray:
    """Generate sine wave trajectory"""
    x = t * 0.5
    y = amplitude * np.sin(2 * np.pi * frequency * t)
    return np.array([x, y, 0.2])

# Main control loop
controller = PIDController()

def control_step(robot_state: Dict, dt: float = 0.016) -> Dict:
    """Execute one control step"""
    t = robot_state['time']
    current_pos = robot_state['position']

    # Get target position on trajectory
    target_pos = sine_trajectory(t)

    # Compute control
    velocity = controller.compute(current_pos, target_pos, dt)

    return {
        'velocity': velocity,
        'target_position': target_pos
    }
```

---

## 🎮 Example 5: Interactive Parameter Tuning

### User Input:
```
Adjust the PID gains for better tracking performance. Current tracking error is too high.
```

### System Response:
The AI will:
1. Analyze current algorithm performance
2. Suggest parameter adjustments:
   ```
   Suggested changes:
   - Increase KP: 2.0 → 3.5 (improve responsiveness)
   - Reduce KD: 0.5 → 0.3 (reduce oscillation)
   - Keep KI: 0.1 (integral wind-up is minimal)
   ```
3. Hot-reload the algorithm with new parameters
4. Continue simulation without restart

You can see the improvement immediately in the Genesis viewer!

---

## 📊 Example 6: Multi-Robot Task Allocation

### User Input:
```
I have 3 mobile robots and 5 target locations. Assign tasks optimally to minimize total travel time.
```

### System Response:
```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def assign_tasks(robot_positions: List[np.ndarray],
                target_positions: List[np.ndarray]) -> Dict[int, int]:
    """
    Hungarian algorithm for optimal task assignment
    """
    n_robots = len(robot_positions)
    n_targets = len(target_positions)

    # Cost matrix (distance)
    cost = np.zeros((n_robots, n_targets))
    for i, robot_pos in enumerate(robot_positions):
        for j, target_pos in enumerate(target_positions):
            cost[i, j] = np.linalg.norm(robot_pos - target_pos)

    # Solve assignment problem
    robot_indices, target_indices = linear_sum_assignment(cost)

    assignments = {
        robot_idx: target_idx
        for robot_idx, target_idx in zip(robot_indices, target_indices)
    }

    return assignments

# Usage in simulation
assignments = assign_tasks(
    robot_positions=[[0, 0, 0], [1, 1, 0], [2, 0, 0]],
    target_positions=[[5, 5, 0], [3, 2, 0], [8, 1, 0],
                     [2, 7, 0], [6, 3, 0]]
)

# Output: {0: 1, 1: 0, 2: 2}
# Robot 0 goes to target 1
# Robot 1 goes to target 0
# Robot 2 goes to target 2
```

---

## 🔧 Example 7: Debugging and Visualization

### User Input:
```
Show me the robot's joint positions over time. I think joint 3 is exceeding limits.
```

### System Response:
The AI will:
1. Access Genesis simulation state
2. Extract joint position data
3. Generate visualization code:

```python
import matplotlib.pyplot as plt

def plot_joint_trajectories(robot_state_history: List[Dict]):
    """Plot joint positions over time"""
    n_joints = len(robot_state_history[0]['joint_positions'])

    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 8))
    time = [state['time'] for state in robot_state_history]

    for i in range(n_joints):
        joint_positions = [
            state['joint_positions'][i]
            for state in robot_state_history
        ]

        axes[i].plot(time, joint_positions, label=f'Joint {i}')
        axes[i].axhline(y=JOINT_LIMITS[i]['max'],
                       color='r', linestyle='--', label='Max')
        axes[i].axhline(y=JOINT_LIMITS[i]['min'],
                       color='r', linestyle='--', label='Min')
        axes[i].legend()
        axes[i].set_ylabel(f'Joint {i} (rad)')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
```

---

## 🎯 Example 8: Complex Manipulation Task

### User Input:
```
Franka robot needs to:
1. Pick up a blue cube from the table
2. Stack it on top of a red cube
3. Return to home position

Use visual feedback from the camera.
```

### System Response:
The AI generates a complete manipulation pipeline:

```python
import numpy as np
from typing import Tuple

# Task stages
class ManipulationTask:
    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        self.stage = 0
        self.stages = [
            'approach_blue_cube',
            'grasp_blue_cube',
            'lift_blue_cube',
            'approach_red_cube',
            'place_on_red_cube',
            'release',
            'return_home'
        ]

    def get_target_pose(self, stage: str,
                       vision_data: Dict) -> np.ndarray:
        """Compute target end-effector pose for each stage"""
        if stage == 'approach_blue_cube':
            blue_pos = vision_data['blue_cube_position']
            return blue_pos + np.array([0, 0, 0.1])  # Approach from above

        elif stage == 'grasp_blue_cube':
            blue_pos = vision_data['blue_cube_position']
            return blue_pos

        elif stage == 'lift_blue_cube':
            return vision_data['blue_cube_position'] + np.array([0, 0, 0.3])

        # ... (more stages)

    def check_stage_complete(self, robot_state: Dict,
                            target: np.ndarray) -> bool:
        """Check if current stage is complete"""
        ee_pos = robot_state['end_effector_position']
        distance = np.linalg.norm(ee_pos - target)
        return distance < 0.01  # 1cm threshold

def manipulation_controller(robot_state: Dict,
                           vision_data: Dict,
                           task: ManipulationTask) -> Dict:
    """Main manipulation control loop"""
    current_stage = task.stages[task.stage]
    target_pose = task.get_target_pose(current_stage, vision_data)

    # Compute inverse kinematics
    joint_targets = inverse_kinematics(
        target_pose,
        robot_state['joint_positions']
    )

    # Check if stage complete
    if task.check_stage_complete(robot_state, target_pose):
        task.stage += 1
        print(f"✅ Completed: {current_stage}")

    return {
        'joint_targets': joint_targets,
        'gripper_state': 'closed' if 'grasp' in current_stage else 'open'
    }
```

---

## 🎨 Tips for Effective Chat Interaction

### Good Prompts:
✅ "Create a Go2 robot that follows a square path"
✅ "Generate A* path planning for mobile robot with 5 obstacles"
✅ "Tune the drone's altitude controller - it's oscillating too much"
✅ "Add 3 more robots to the scene at random positions"

### What to Specify:
1. **Robot type**: Franka, Go2, Drone, Mobile
2. **Task description**: Pick, place, navigate, follow
3. **Constraints**: Speed limits, obstacles, joint limits
4. **Performance goals**: Fast, smooth, energy-efficient

### Interactive Features:
- 🔄 **Real-time updates**: Changes apply immediately
- 📊 **Performance metrics**: FPS, tracking error, success rate
- 🎥 **Video streaming**: Watch your simulation live
- 💾 **Save/Load**: Reuse successful configurations

---

## 🚀 Quick Start Commands

Try these in the chat interface:

```
1. "Show me available robot models"
   → Lists all URDF/MJCF models

2. "Create a simple test scene with Franka"
   → Quick setup with default robot

3. "Generate a circular trajectory following algorithm"
   → AI creates the algorithm code

4. "Increase robot speed by 50%"
   → Modifies algorithm parameters

5. "Reset simulation and start over"
   → Clears scene and reinitializes

6. "Export this algorithm as Python file"
   → Downloads generated code
```

---

## 📚 Advanced: WebSocket Real-Time Control

You can also control simulations programmatically via WebSocket:

```javascript
// Frontend example
const ws = new WebSocket('ws://localhost:8000/api/genesis/ws');

ws.onopen = () => {
  // Request initial state
  ws.send(JSON.stringify({ type: 'get_state' }));

  // Send robot command
  ws.send(JSON.stringify({
    type: 'robot_command',
    robot_id: 'franka1',
    command: {
      joint_positions: [0, 0, 0, -1.5, 0, 1.5, 0]
    }
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'state_update') {
    // Update UI with robot state
    console.log('Robot state:', message.data.robots);
  }

  if (message.type === 'frame') {
    // Display video frame
    displayFrame(message.data);
  }
};
```

---

**Ready to start?** Open http://localhost:5173 and try these examples in the chat! 🚀
