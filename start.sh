#!/bin/bash

# Celestial Studio - Local Startup Script
# Launches backend + frontend for local development

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting Celestial Studio Locally${NC}"
echo "========================================"
echo ""

# Kill existing processes
echo "🔄 Cleaning up existing processes..."
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
echo "========================================"
echo -e "${GREEN}✅ Celestial Studio is Running!${NC}"
echo "========================================"
echo ""
echo "📊 Services:"
echo "  • Backend:  http://localhost:8000"
echo "  • Frontend: http://localhost:5173"
echo "  • API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Logs:"
echo "  • Backend:  tail -f logs/backend.log"
echo "  • Frontend: tail -f logs/frontend.log"
echo ""
echo "⏹️  To stop: ./stop.sh"
echo ""
echo "🎬 Ready for demo recording!"
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
