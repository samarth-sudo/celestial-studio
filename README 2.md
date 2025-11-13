# 🤖 Robot Code Generator - Complete Platform

**Production-ready robot code + Gazebo simulation environments from natural language**

Generate complete robotics systems including hardware control code, URDF models, and professional Gazebo/ROS2 simulation environments.

---

## ✨ What's New in v2.0

🎮 **Gazebo Simulation Support** - Automatic URDF + Gazebo/ROS2 environment generation
🌐 **Web Interface** - Modern React frontend with Monaco editor
🚀 **FastAPI Backend** - REST API with WebSocket support
📦 **Complete Packages** - Hardware + Simulation in one click
🏭 **Professional Simulator** - Industry-standard Gazebo with ROS2 integration

---

## 🚀 Quick Start

### Option 1: Docker Compose (Fastest)

```bash
# Clone and start
git clone <repo>
cd idea
docker-compose up --build

# Access:
Frontend: http://localhost:3000
API Docs: http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Backend
pip install -r requirements.txt -r backend/requirements.txt
cd backend && python main.py

# Frontend (new terminal)
cd frontend
npm install && npm start
```

---

## 💡 Usage

### Via Web Interface
1. Open http://localhost:3000
2. Fill robot specs (name, DOF, task, hardware)
3. Click "Generate Code"
4. Download packages:
   - **Hardware Package**: Production code for real robot
   - **Simulation Package**: Gazebo/ROS2 environment

### Via API
```python
import requests

response = requests.post('http://localhost:8000/api/generate', json={
    "specs": {
        "robot_name": "my_arm",
        "dof": 6,
        "task": "pick red cubes",
        "hardware": {"servo_controller": "lewansoul"}
    },
    "include_simulation": True,
    "include_hardware": True
})
```

---

## 📦 Generated Packages

### Hardware Package (~1500 LOC)
```
my_arm_python/
├── controller.py          # Complete production code
├── hardware/              # LewanSoul/PCA9685/Dynamixel drivers
├── vision/                # Color detection, tracking
├── control/               # PID, trajectory planning
├── tasks/                 # Pick-place state machines
└── requirements.txt
```

### Simulation Package
```
my_arm_gazebo/
├── urdf/                  # Auto-generated URDF
│   └── my_arm.urdf
├── launch/                # ROS2 launch files
│   └── spawn_my_arm.launch.py
├── worlds/                # Gazebo world files
│   └── default.world
└── README.md              # Setup and usage instructions
```

---

## 🎓 Running in Gazebo

```bash
cd my_arm_gazebo

# Source ROS2 (if not already sourced)
source /opt/ros/humble/setup.bash

# Launch Gazebo simulation
ros2 launch launch/spawn_my_arm.launch.py

# Control robot via ROS2 topics
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"
```

---

## 🏗️ Architecture

```
┌─────────────────┐
│ React Frontend  │ (Monaco Editor, Split Pane UI)
│   Port 3000     │
└────────┬────────┘
         │ HTTP/WebSocket
┌────────▼────────┐
│ FastAPI Backend │ (REST API, Session Management)
│   Port 8000     │
└────────┬────────┘
         │
    ┌────▼─────┬──────────┬───────────┐
    │          │          │           │
┌───▼───┐ ┌───▼───┐ ┌────▼────┐ ┌────▼────┐
│ MLX   │ │  RAG  │ │  URDF   │ │ Gazebo  │
│Qwen2.5│ │ (112  │ │Generator│ │ROS2 Pkg │
│ Model │ │papers)│ │         │ │Builder  │
└───────┘ └───────┘ └─────────┘ └─────────┘
```

---

## 📊 Features Comparison

| Feature | v1.0 (Chainlit) | **v2.0 (Web)** |
|---------|-----------------|----------------|
| Code Generation | ✅ | ✅ **Enhanced** |
| Hardware Drivers | ✅ | ✅ **+1000 LOC** |
| Web Interface | ❌ | ✅ **React + Monaco** |
| URDF Generation | ❌ | ✅ **Auto** |
| Gazebo Simulation | ❌ | ✅ **ROS2** |
| 3D Visualization | ❌ | ✅ **Professional** |
| Real-time Updates | ❌ | ✅ **WebSocket** |
| Package Download | ✅ | ✅ **Improved** |

---

## 🔧 Supported Hardware

### Servo Controllers
- **LewanSoul/Hiwonder** (like Dofbot) - Complete protocol implementation
- **PCA9685** (I2C PWM) - Adafruit library
- **Dynamixel** - Robotis smart servos

### Cameras
- USB Webcam (OpenCV)
- CSI Camera (Jetson Nano/RPi with GStreamer)
- Orbbec Depth Camera

### Grippers
- Servo Gripper
- Pneumatic Gripper
- Electric Gripper

### Platforms
- Jetson Nano (tested with Dofbot)
- Raspberry Pi
- Arduino/ESP32
- Linux PC

---

## 🧪 Testing

```bash
# Test hardware code generation
python test_dofbot_generation.py

# Test simulation package generation
python test_simulation_generation.py

# Test API (requires server running)
python backend/test_api.py
```

---

## 📁 Project Structure

```
robot-code-generator/
├── backend/              # FastAPI server
│   ├── main.py
│   └── requirements.txt
├── frontend/             # React app
│   ├── src/
│   ├── package.json
│   └── Dockerfile
├── src/
│   ├── generation/       # Code generator (MLX + Qwen2.5)
│   ├── hardware/         # Device drivers (1000+ LOC)
│   ├── vision/           # Color detection, tracking
│   ├── control/          # PID, trajectory
│   ├── tasks/            # State machines
│   ├── simulation/       # NEW: URDF + MuJoCo
│   └── packaging/        # Package builders
├── templates/            # Jinja2 templates
├── Modelfile             # Optimized robotics LLM
├── docker-compose.yml
└── README.md
```

---

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Hardware Setup**: See `SETUP.md` in generated packages
- **Training Guide**: See `README.md` in simulation packages
- **Examples**: `test_*.py` files

---

## 🚀 Deployment

### Production (Docker)
```bash
docker-compose up -d
docker-compose logs -f
```

### Environment Variables
```bash
# .env
SUPERMEMORY_API_KEY=your_key
REACT_APP_API_URL=http://localhost:8000
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model Loading | ~5-10s |
| Code Generation | ~30-60s |
| URDF Generation | <1s |
| Package Build | ~5s |
| RL Training (1M steps) | 2-4 hours (CPU) |

---

## 🎯 Use Cases

✅ **Rapid Prototyping** - Generate complete robot code in minutes
✅ **Education** - Learn robotics with production-quality code
✅ **Research** - Quick RL environment setup for manipulation tasks
✅ **Sim-to-Real** - Train in MuJoCo, deploy to hardware
✅ **Baseline Code** - Start with working code, customize later

---

## 🤝 Contributing

Contributions welcome! Focus areas:
- Additional hardware support
- More simulation environments
- Improved RL algorithms
- Better visualization
- Documentation

---

## 📝 License

MIT License

---

## 🙏 Credits

- **Qwen2.5-Coder** (Alibaba) - Code generation
- **MLX** (Apple) - GPU acceleration
- **MuJoCo** (DeepMind) - Physics simulation
- **Stable-Baselines3** - RL algorithms
- **Supermemory** - RAG (112 papers)
- **FastAPI** - Backend framework
- **React** - Frontend
- **Monaco Editor** - Code editor

---

## 🎉 Status

### v2.0 MVP Complete ✅

**Day 1** (Complete):
- ✅ Optimized Modelfile
- ✅ URDF Generator
- ✅ MuJoCo Environment Builder
- ✅ Training Script Generator
- ✅ Simulation Package Builder

**Day 2** (Complete):
- ✅ FastAPI Backend (REST + WebSocket)
- ✅ React Frontend (Monaco Editor)
- ✅ Session Management
- ✅ File Download System

**Day 3** (Complete):
- ✅ Docker Compose
- ✅ Integration Testing
- ✅ Documentation
- ✅ End-to-End Workflow

### Next Steps (Optional):
- 🔄 MuJoCo Browser Visualization (MuJoCo-WASM)
- 🔄 Live Training Metrics
- 🔄 Multi-user Support
- 🔄 Cloud Deployment (AWS/GCP)

---

**Ready for use! Generate your first robot in 60 seconds** 🚀
