# ⚡ Quick Start Examples - Input & Output

Real examples showing exactly what to type and what you'll get.

---

## 🎯 Example 1: List Available Robots

### Input (in terminal):
```bash
curl -s http://localhost:8000/api/genesis/models | python3 -m json.tool
```

### Output:
```json
{
    "status": "success",
    "models": {
        "urdf": [
            {
                "name": "panda_bullet",
                "path": "urdf/panda_bullet/panda.urdf",
                "type": "urdf"
            },
            {
                "name": "go2",
                "path": "urdf/go2/urdf/go2.urdf",
                "type": "urdf"
            },
            {
                "name": "drones",
                "path": "urdf/drones/cf2x.urdf",
                "type": "urdf"
            }
        ],
        "xml": [
            {
                "name": "franka_emika_panda",
                "path": "xml/franka_emika_panda/panda.xml",
                "type": "mjcf"
            }
        ]
    },
    "total_urdf": 9,
    "total_xml": 9
}
```

---

## 🤖 Example 2: Create Franka Simulation (Python)

### Input:
```python
# save as: my_first_simulation.py
import sys
sys.path.insert(0, 'backend')

from genesis_service import GenesisSimulation, GenesisConfig, RobotType

# Setup
config = GenesisConfig(fps=60, show_viewer=False)
sim = GenesisSimulation(config)
sim.initialize()

# Add robot
sim.add_robot("franka1", RobotType.FRANKA, position=(0, 0, 0.5))
sim.build_scene()
sim.start()

# Run 10 steps
for i in range(10):
    state = sim.step()
    print(f"Step {i}: {len(state['robots'])} robots, {state['fps']:.1f} FPS")
```

### Run:
```bash
source venv/bin/activate
python3 my_first_simulation.py
```

### Output:
```
Step 0: 1 robots, 0.0 FPS
Step 1: 1 robots, 62.3 FPS
Step 2: 1 robots, 61.8 FPS
Step 3: 1 robots, 60.2 FPS
Step 4: 1 robots, 60.1 FPS
Step 5: 1 robots, 59.9 FPS
Step 6: 1 robots, 60.3 FPS
Step 7: 1 robots, 60.0 FPS
Step 8: 1 robots, 60.1 FPS
Step 9: 1 robots, 59.8 FPS
```

---

## 💬 Example 3: Chat Interface

### Input (Type in chat UI):
```
Create a Franka Panda robot at position (0, 0, 0.5) with a red cube to pick up
```

### System Response:
```
I'll create a pick-and-place simulation for you!

Setting up:
✅ Genesis initialized (Metal backend, 60 FPS)
✅ Added Franka Panda robot at (0, 0, 0.5)
✅ Added red cube target at (0.5, 0.5, 0.3)
✅ Scene built successfully

Generating pick-and-place algorithm...

Generated algorithm: pick_and_place.py
```

### Generated Code (automatically):
```python
import numpy as np
from typing import Dict, Tuple

# Task configuration
CUBE_POSITION = np.array([0.5, 0.5, 0.3])
APPROACH_HEIGHT = 0.1
GRASP_THRESHOLD = 0.01

def pick_and_place_controller(robot_state: Dict) -> Dict:
    """
    Pick and place controller for Franka Panda
    """
    ee_pos = robot_state['end_effector_position']

    # State machine: approach -> grasp -> lift -> return
    if not hasattr(pick_and_place_controller, 'stage'):
        pick_and_place_controller.stage = 'approach'

    if pick_and_place_controller.stage == 'approach':
        target = CUBE_POSITION + np.array([0, 0, APPROACH_HEIGHT])
        if np.linalg.norm(ee_pos - target) < GRASP_THRESHOLD:
            pick_and_place_controller.stage = 'grasp'

    elif pick_and_place_controller.stage == 'grasp':
        target = CUBE_POSITION
        if np.linalg.norm(ee_pos - target) < GRASP_THRESHOLD:
            pick_and_place_controller.stage = 'lift'
            return {'gripper': 'close'}

    # ... (more stages)

    return {
        'target_position': target,
        'gripper': 'open'
    }
```

---

## 🚁 Example 4: Drone Swarm

### Input (Chat):
```
Create 5 Crazyflie drones that fly in a pentagon formation at height 2.0 meters
```

### System Output:
```
Creating drone swarm...

✅ Added 5 Crazyflie drones:
   • drone_0 at (0.0, 0.0, 2.0)
   • drone_1 at (0.5, 0.0, 2.0)
   • drone_2 at (1.0, 0.0, 2.0)
   • drone_3 at (1.5, 0.0, 2.0)
   • drone_4 at (2.0, 0.0, 2.0)

Generating pentagon formation controller...

Formation will have radius: 1.0m
Center position: (1.0, 0.0, 2.0)
```

### In Genesis Viewer:
- 5 drones visible
- Flying to pentagon formation
- Real-time video stream at 30 FPS
- Simulation running at 60 physics FPS

---

## 🏃 Example 5: Go2 Quadruped Path Planning

### Input (API):
```bash
# Initialize
curl -X POST http://localhost:8000/api/genesis/init \
  -H "Content-Type: application/json" \
  -d '{"backend": "auto", "fps": 60}'

# Add Go2 robot
curl -X POST http://localhost:8000/api/genesis/robot/add \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "go2_robot",
    "robot_type": "go2",
    "position": [0, 0, 0.3]
  }'

# Add obstacles
curl -X POST http://localhost:8000/api/genesis/obstacle/add \
  -H "Content-Type: application/json" \
  -d '{"obstacle_id": "box1", "position": [2, 2, 0.25], "size": [0.5, 0.5, 0.5]}'

# Build and start
curl -X POST http://localhost:8000/api/genesis/scene/build
curl -X POST http://localhost:8000/api/genesis/control \
  -d '{"action": "start"}'
```

### Output:
```json
{
  "status": "success",
  "message": "Genesis initialized with metal backend",
  "backend": "metal",
  "fps": 60
}

{
  "status": "success",
  "robot_id": "go2_robot",
  "robot_type": "go2",
  "position": [0, 0, 0.3]
}

{
  "status": "success",
  "obstacle_id": "box1"
}

{
  "status": "success",
  "message": "Scene built with 1 robots and 1 obstacles"
}

{
  "status": "running",
  "step_count": 0
}
```

---

## 📊 Example 6: Check Status

### Input:
```bash
curl -s http://localhost:8000/api/genesis/status | python3 -m json.tool
```

### Output:
```json
{
    "available": true,
    "initialized": true,
    "running": true,
    "step_count": 127,
    "robot_count": 1,
    "obstacle_count": 1
}
```

---

## 🎮 Example 7: Real-Time Control via WebSocket

### Input (JavaScript):
```javascript
// Open WebSocket connection
const ws = new WebSocket('ws://localhost:8000/api/genesis/ws');

// Listen for state updates
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'state_update') {
        console.log('Robots:', msg.data.robots);
        console.log('Step:', msg.data.step);
        console.log('FPS:', msg.data.fps);
    }

    if (msg.type === 'frame') {
        // Display video frame
        displayFrame(msg.data);
    }
};

// Send robot command
ws.send(JSON.stringify({
    type: 'robot_command',
    robot_id: 'franka1',
    command: {
        target_position: [0.5, 0.5, 0.8]
    }
}));
```

### Output (Console):
```
Robots: {franka1: {position: [0.1, 0.2, 0.5], velocity: [0.01, 0.02, 0.0]}}
Step: 42
FPS: 60.2

Robots: {franka1: {position: [0.15, 0.25, 0.55], velocity: [0.02, 0.03, 0.01]}}
Step: 43
FPS: 60.1
```

---

## 🔍 Example 8: Algorithm Hot-Reload

### Step 1 - Initial Algorithm (Chat):
```
Generate a PID controller for mobile robot with KP=1.0
```

### System Creates:
```python
KP = 1.0
KI = 0.1
KD = 0.5

def pid_controller(robot_state):
    # ... implementation
```

### Step 2 - Tune Parameters (Chat):
```
The robot is too slow. Increase KP to 2.5
```

### System Updates (Hot-reload):
```python
KP = 2.5  # ← Updated!
KI = 0.1
KD = 0.5

def pid_controller(robot_state):
    # ... same implementation
```

### Result:
- Algorithm reloaded in < 100ms
- Simulation continues without restart
- Robot immediately responds faster

---

## 🎯 Example 9: Multi-Robot Scenario

### Input (Python):
```python
sim = GenesisSimulation(GenesisConfig(fps=60))
sim.initialize()

# Add different robot types
sim.add_robot("franka", RobotType.FRANKA, (0, 0, 0.5))
sim.add_robot("go2", RobotType.GO2, (3, 0, 0.3))
sim.add_robot("drone1", RobotType.DRONE, (0, 3, 1.5))
sim.add_robot("drone2", RobotType.DRONE, (3, 3, 1.5))

# Add obstacles
for i in range(5):
    sim.add_obstacle(f"obs{i}",
                     position=(i*1.0, 2.0, 0.25),
                     size=(0.3, 0.3, 0.5))

sim.build_scene()
sim.start()

# Simulate
for i in range(100):
    state = sim.step()
    if i % 20 == 0:
        print(f"Step {i}:")
        print(f"  Robots: {len(state['robots'])}")
        print(f"  FPS: {state['fps']:.1f}")
```

### Output:
```
Step 0:
  Robots: 4
  FPS: 0.0

Step 20:
  Robots: 4
  FPS: 58.7

Step 40:
  Robots: 4
  FPS: 59.2

Step 60:
  Robots: 4
  FPS: 60.1

Step 80:
  Robots: 4
  FPS: 59.8
```

---

## 📹 Example 10: Get Video Frame

### Input:
```bash
curl -s http://localhost:8000/api/genesis/stream/frame > frame.json
python3 -c "
import json
import base64
from PIL import Image
import io

# Read frame
with open('frame.json') as f:
    data = json.load(f)

# Decode base64 JPEG
img_data = base64.b64decode(data['frame'])
img = Image.open(io.BytesIO(img_data))

print(f'Frame size: {img.size}')
print(f'Format: {img.format}')
img.save('output.jpg')
print('Saved to output.jpg')
"
```

### Output:
```
Frame size: (1920, 1080)
Format: JPEG
Saved to output.jpg
```

---

## 🚀 Try It Yourself!

1. **Start the system**:
   ```bash
   ./start.sh
   ```

2. **Open the UI**: http://localhost:5173

3. **Type in chat**:
   ```
   Create a simple test with Franka robot
   ```

4. **Watch it run!** 🎉

---

**All these examples are ready to use right now!** 🚀

Check `examples/` folder for more detailed documentation.
