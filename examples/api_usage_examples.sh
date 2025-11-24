#!/bin/bash
# Genesis API Usage Examples
# These examples show how to interact with the Genesis backend

BASE_URL="http://localhost:8000"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 Genesis API Usage Examples"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Check Genesis Status
echo "1️⃣  Check Genesis Status"
echo "────────────────────────────────────────"
curl -s ${BASE_URL}/api/genesis/status | python3 -m json.tool
echo ""
echo ""

# 2. List Available Robot Models
echo "2️⃣  List Available Robot Models"
echo "────────────────────────────────────────"
curl -s ${BASE_URL}/api/genesis/models | python3 -m json.tool | head -50
echo "... (truncated, see full output at ${BASE_URL}/api/genesis/models)"
echo ""
echo ""

# 3. Initialize Genesis Simulation
echo "3️⃣  Initialize Genesis Simulation"
echo "────────────────────────────────────────"
curl -X POST ${BASE_URL}/api/genesis/init \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "auto",
    "fps": 60,
    "render_width": 1920,
    "render_height": 1080,
    "stream_quality": "medium"
  }' | python3 -m json.tool
echo ""
echo ""

# 4. Add Franka Panda Robot
echo "4️⃣  Add Franka Panda Robot"
echo "────────────────────────────────────────"
curl -X POST ${BASE_URL}/api/genesis/robot/add \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "franka1",
    "robot_type": "franka",
    "position": [0, 0, 0.5]
  }' | python3 -m json.tool
echo ""
echo ""

# 5. Add Go2 Quadruped
echo "5️⃣  Add Go2 Quadruped Robot"
echo "────────────────────────────────────────"
curl -X POST ${BASE_URL}/api/genesis/robot/add \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "go2_robot",
    "robot_type": "go2",
    "position": [2, 0, 0.5]
  }' | python3 -m json.tool
echo ""
echo ""

# 6. Add Drone
echo "6️⃣  Add Crazyflie Drone"
echo "────────────────────────────────────────"
curl -X POST ${BASE_URL}/api/genesis/robot/add \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "drone1",
    "robot_type": "drone",
    "position": [0, 2, 1.5]
  }' | python3 -m json.tool
echo ""
echo ""

# 7. Add Obstacles
echo "7️⃣  Add Obstacles"
echo "────────────────────────────────────────"
curl -X POST ${BASE_URL}/api/genesis/obstacle/add \
  -H "Content-Type: application/json" \
  -d '{
    "obstacle_id": "box1",
    "position": [1, 1, 0.25],
    "size": [0.5, 0.5, 0.5]
  }' | python3 -m json.tool
echo ""
echo ""

# 8. Build Scene
echo "8️⃣  Build Scene (Required before simulation)"
echo "────────────────────────────────────────"
curl -X POST ${BASE_URL}/api/genesis/scene/build | python3 -m json.tool
echo ""
echo ""

# 9. Start Simulation
echo "9️⃣  Start Simulation"
echo "────────────────────────────────────────"
curl -X POST ${BASE_URL}/api/genesis/control \
  -H "Content-Type: application/json" \
  -d '{
    "action": "start"
  }' | python3 -m json.tool
echo ""
echo ""

# 10. Get Frame
echo "🔟 Get Current Frame (Video Stream)"
echo "────────────────────────────────────────"
echo "Frame available at: ${BASE_URL}/api/genesis/stream/frame"
echo "(Returns base64-encoded JPEG image)"
echo ""
echo ""

# 11. Stop Simulation
echo "⏸️  Stop Simulation"
echo "────────────────────────────────────────"
curl -X POST ${BASE_URL}/api/genesis/control \
  -H "Content-Type: application/json" \
  -d '{
    "action": "stop"
  }' | python3 -m json.tool
echo ""
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Examples Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📚 Additional Resources:"
echo "  • API Docs: ${BASE_URL}/docs"
echo "  • WebSocket: ws://localhost:8000/api/genesis/ws"
echo "  • Stream Stats: ${BASE_URL}/api/genesis/stream/stats"
