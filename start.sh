#!/bin/bash

# Celestial Studio - Local Startup Script
# Launches backend + frontend with Genesis physics engine

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}   🚀 Celestial Studio - Genesis Edition   ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check Python environment
echo -e "${YELLOW}🔍 Checking Python environment...${NC}"
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${RED}⚠️  Virtual environment not found${NC}"
    echo "   Run: python3 -m venv venv && source venv/bin/activate"
fi

# Check Genesis installation
echo ""
echo -e "${YELLOW}🌌 Checking Genesis physics engine...${NC}"
if python3 -c "import genesis" 2>/dev/null; then
    GENESIS_VERSION=$(python3 -c "import genesis; print(genesis.__version__)" 2>/dev/null)
    echo -e "${GREEN}✅ Genesis ${GENESIS_VERSION} installed${NC}"

    # Check backend type (Metal, CUDA, CPU)
    if python3 -c "import platform; import subprocess; exit(0 if 'Apple' in subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout else 1)" 2>/dev/null; then
        echo -e "${GREEN}   🍎 Apple Silicon detected - will use Metal backend${NC}"
    else
        echo -e "${BLUE}   💻 Intel CPU detected - checking for NVIDIA GPU...${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Genesis not installed${NC}"
    echo -e "${CYAN}   Install with: pip install genesis-world${NC}"
    echo -e "${CYAN}   System will fallback to Three.js renderer${NC}"
fi

# Kill existing processes
echo ""
echo -e "${YELLOW}🔄 Cleaning up existing processes...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 2

# Start backend
echo ""
echo -e "${YELLOW}🔧 Starting Backend (FastAPI)...${NC}"
cd backend
python3 main.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

sleep 5

# Check backend
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend running on http://localhost:8000${NC}"
else
    echo "⚠️  Backend may still be starting..."
fi

# Start frontend
echo ""
echo -e "${YELLOW}🎨 Starting Frontend (Vite)...${NC}"
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

sleep 8

# Check frontend
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend running on http://localhost:5173${NC}"
else
    echo "⚠️  Frontend may still be compiling..."
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   ✅ Celestial Studio is Running!   ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}📊 Services:${NC}"
echo -e "  ${GREEN}•${NC} Backend:  http://localhost:8000"
echo -e "  ${GREEN}•${NC} Frontend: http://localhost:5173"
echo -e "  ${GREEN}•${NC} API Docs: http://localhost:8000/docs"
echo -e "  ${GREEN}•${NC} Genesis:  http://localhost:8000/api/genesis/status"
echo ""
echo -e "${CYAN}🎨 Rendering Options:${NC}"
echo -e "  ${GREEN}•${NC} Genesis Mode:  🚀 GPU-accelerated (43M FPS)"
echo -e "  ${GREEN}•${NC} Three.js Mode: 🎨 Browser rendering (60 FPS)"
echo -e "  ${YELLOW}→${NC} Toggle in UI (top-right button)"
echo ""
echo -e "${CYAN}📝 Logs:${NC}"
echo -e "  ${GREEN}•${NC} Backend:  tail -f logs/backend.log"
echo -e "  ${GREEN}•${NC} Frontend: tail -f logs/frontend.log"
echo ""
echo -e "${CYAN}⏹️  Control:${NC}"
echo -e "  ${GREEN}•${NC} Stop all: ./stop.sh"
echo -e "  ${GREEN}•${NC} Or press: Ctrl+C"
echo ""
echo -e "${YELLOW}🎬 Ready to simulate!${NC}"
echo ""

# Save PIDs
mkdir -p logs
echo $BACKEND_PID > logs/backend.pid
echo $FRONTEND_PID > logs/frontend.pid

# Open browser (optional)
echo "Opening browser..."
sleep 2
open http://localhost:5173 2>/dev/null || echo "Please open http://localhost:5173 in your browser"

echo ""
echo "Press Ctrl+C to stop all services..."
trap 'echo ""; echo "Stopping..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait
