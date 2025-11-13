# ✨ Celestial Studio

**AI-Powered Robot Code Generation & 3D Simulation with Dynamic Algorithm Generation**

Conversational robotics IDE that lets you describe robot behavior in natural language, generates TypeScript algorithms in real-time using Qwen 2.5 Coder, and hot-swaps them into live simulations - all running locally on your Mac M2 Pro!

## ✨ Features

### Core Capabilities
- **💬 Natural Language Programming** - Describe behavior like "avoid obstacles smoothly" or "pick up the red cube"
- **🧠 Dynamic Algorithm Generation** - Qwen 2.5 Coder generates TypeScript algorithms (A*, DWA, FABRIK, IK solvers, etc.)
- **🔥 Hot Code Swapping** - Modify algorithms in real-time without restarting simulation
- **🎛️ Live Parameter Tuning** - Adjust algorithm parameters with sliders while simulation runs
- **🎮 3D Web Simulation** - Physics-based visualization with React Three Fiber + Rapier
- **🚀 3 Robot Types** - Mobile robots, robotic arms, and drones
- **📹 Computer Vision System** - Picture-in-Picture camera with object detection overlay
- **🖥️ 100% Local** - No cloud API keys needed, runs entirely on your Mac

### Dynamic System Features
- **Real-time Modification** - Say "make it faster" and watch algorithm adapt live
- **Algorithm Library** - Browse and apply pre-built algorithms (A*, DWA, FABRIK, etc.)
- **Research-Grade Algorithms** - Based on 2024 robotics papers from Supermemory
- **Safety Sandbox** - Isolated code execution with performance monitoring
- **Parameter Extraction** - Auto-generates UI controls from algorithm code

## 🎯 Demo Flow - Dynamic Algorithm System (60 seconds)

### Part 1: Robot Generation (0:00-0:20)
1. **[0:00-0:05]** Type: "Build a 4-wheel mobile warehouse robot"
2. **[0:05-0:10]** Code generates → Click "Run Simulation"
3. **[0:10-0:20]** Watch robot navigate in 3D

### Part 2: Dynamic Algorithm Generation (0:20-0:45)
4. **[0:20-0:25]** Type: "Make it avoid obstacles using DWA algorithm"
5. **[0:25-0:30]** Watch Qwen generate obstacle avoidance code
6. **[0:30-0:35]** Click "Apply Algorithm" → Robot instantly uses new behavior
7. **[0:35-0:40]** Add obstacles to scene, watch robot navigate around them
8. **[0:40-0:45]** Adjust "safety margin" slider → See behavior change in real-time

### Part 3: Real-Time Modification (0:45-0:60)
9. **[0:45-0:50]** Type: "Make it move faster"
10. **[0:50-0:55]** Qwen modifies algorithm → Code hot-swaps while running
11. **[0:55-0:60]** Watch robot speed up without restart

## 🚀 Quick Start

```bash
# Make startup script executable
chmod +x start.sh

# Start both servers
./start.sh
```

Then open: **http://localhost:5173**

## 📋 Manual Setup

### Backend (Terminal 1)
```bash
source venv/bin/activate
python backend/main.py
```

### Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```

## 🎬 Demo Examples

### Mobile Robot
```
Build a 4-wheel mobile warehouse robot for autonomous navigation
```
**Shows:** Waypoint navigation, wheel physics, path following

### Robotic Arm
```
Create a 6-DOF robotic arm for pick and place tasks
```
**Shows:** Joint articulation, gripper control, pick-and-place motion

### Drone
```
Make a quadcopter drone for aerial inspection
```
**Shows:** Flight physics, propeller animation, figure-8 pattern

## 🛠️ Tech Stack

**Frontend:**
- React 18 + TypeScript
- React Three Fiber (3D rendering)
- Rapier (physics engine)
- Drei (3D helpers)
- Monaco Editor (code viewer)
- Vite (blazing fast builds)

**Backend:**
- FastAPI (Python web framework)
- Ollama (local AI server)
- Qwen 2.5 Coder 7B (code generation)

## 📦 Project Structure

```
/idea
├── frontend/                      # React app
│   ├── src/
│   │   ├── App.tsx               # Main app component
│   │   ├── components/
│   │   │   ├── ChatPanel.tsx            # Chat + algorithm modification
│   │   │   ├── Simulator.tsx            # 3D canvas
│   │   │   ├── AlgorithmControls.tsx    # Algorithm management UI
│   │   │   ├── ParameterPanel.tsx       # Real-time parameter sliders
│   │   │   ├── AlgorithmLibrary.tsx     # Algorithm marketplace
│   │   │   ├── CameraView.tsx           # PiP computer vision
│   │   │   └── robots/
│   │   │       ├── MobileRobot.tsx      # 4-wheel robot
│   │   │       ├── RoboticArm.tsx       # 6-DOF arm
│   │   │       └── Drone.tsx            # Quadcopter
│   │   ├── services/
│   │   │   └── AlgorithmManager.ts      # Hot-swappable algorithm system
│   │   ├── utils/
│   │   │   ├── pathPlanning.ts          # A* reference implementation
│   │   │   ├── codeCompiler.ts          # TypeScript → JavaScript
│   │   │   ├── sandboxExecutor.ts       # Isolated code execution
│   │   │   └── statePersistence.ts      # Save/restore robot state
│   │   ├── data/
│   │   │   └── algorithmTemplates.ts    # Pre-built algorithm templates
│   │   └── main.tsx
│   └── package.json
│
├── backend/                       # FastAPI server
│   ├── main.py                   # API endpoints
│   ├── algorithm_generator.py    # Core algorithm generation logic
│   ├── algorithm_templates.py    # Algorithm pattern templates
│   ├── code_validator.py         # Safety checks
│   └── requirements.txt
│
├── .env                          # API keys (not needed for Ollama)
├── robot_env/                    # Python virtual environment
└── start.sh                      # Startup script
```

## 🎨 UI Layout

```
┌─────────────────────────────────────────────────┐
│  🤖 Robotics Demo Platform                      │
│  Chat → Generate → Simulate → Download          │
├──────────────────┬──────────────────────────────┤
│                  │                              │
│  Chat Panel      │   3D Simulator              │
│  (40%)           │   (60%)                      │
│                  │                              │
│  • Input box     │   • Physics-based            │
│  • Example       │   • Camera controls          │
│    prompts       │   • Grid floor               │
│  • Monaco code   │   • Realistic lighting       │
│    viewer        │   • Smooth animations        │
│  • Run button    │                              │
│                  │                              │
└──────────────────┴──────────────────────────────┘
```

## 🔑 Prerequisites

**Ollama Setup:**

1. Ollama should be installed (already done on your system)
2. Start Ollama server:
   ```bash
   ollama serve
   ```
3. The qwen2.5-coder:7b model is already downloaded

**No API keys needed** - everything runs locally on your Mac M2 Pro!

## 📝 Recording the Demo

### Recommended Flow:

1. **Open browser** at http://localhost:5173
2. **Start screen recording** (QuickTime, OBS, etc.)
3. **Run through all 3 robots:**
   - Click "Build a 4-wheel mobile warehouse robot"
   - Wait for code generation
   - Click "Run Simulation"
   - Let it navigate for 10 seconds
   - Repeat for arm and drone
4. **Show all together** (optional): Run all 3 at once

### Tips:
- Full screen the browser for clean recording
- Zoom camera to show robots clearly (mouse drag to orbit)
- Let each robot complete at least one cycle
- Narrate what's happening ("Here the AI generates Python code...")

## 🎯 Key Selling Points

1. **100% Local & Private** - All AI processing on your Mac, no cloud dependencies
2. **End-to-End Workflow** - From description to working simulation in seconds
3. **Production Code** - Generate real Python controllers, not toy examples
4. **Visual Feedback** - See exactly how the robot will move before deploying
5. **Fast & Efficient** - Qwen 2.5 Coder optimized for Mac M2 Pro's unified memory
6. **Web-Based UI** - Modern 3D interface runs in browser
7. **Extensible** - Easy to add new robot types or behaviors

## 🚧 Current Status

### ✅ Implemented
- Natural language robot code generation
- 3D physics simulation with Rapier
- Three robot types (mobile, arm, drone)
- Local AI with Qwen 2.5 Coder

### 🚧 In Development (Dynamic Algorithm System)
- **Phase 1:** Algorithm Generator Backend (in progress)
- **Phase 2:** Dynamic Code Injection Frontend
- **Phase 3:** Real-Time Modification System
- **Phase 4:** Algorithm Library System
- **Phase 5:** Safety & Validation
- **Phase 6:** Computer Vision Integration

### 📋 Known Limitations
- Web-based physics (not hardware-accurate)
- Browser performance limits complex simulations
- No actual robot hardware integration yet
- Single-page app (no persistence yet)

## 🔮 Roadmap

### Near-term (1-2 weeks)
- [x] Backend algorithm generator with Qwen
- [ ] Hot code swapping system
- [ ] Real-time parameter tuning UI
- [ ] Algorithm library marketplace
- [ ] Computer vision with PiP camera

### Mid-term (1-2 months)
- [ ] Save/load robot configurations
- [ ] Export to ROS packages
- [ ] Multi-robot coordination algorithms
- [ ] Custom 3D model uploads
- [ ] Collaborative algorithm sharing

### Long-term (3-6 months)
- [ ] Real robot hardware integration
- [ ] Reinforcement learning training
- [ ] Cloud simulation scaling
- [ ] Mobile app for remote monitoring

## 📞 Support

**Celestial Studio** - AI-Powered Robotics Development

**Stack:** React + Three.js + FastAPI + Qwen 2.5 Coder + Ollama
**Platform:** Mac M2 Pro (optimized for Apple Silicon)
**Purpose:** Demo & development platform

---

**🎬 Ready to demo! Open http://localhost:5173 and start building robots!**

**✨ Powered by Celestial Studio**
