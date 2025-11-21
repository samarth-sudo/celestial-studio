#!/bin/bash

# Stop Celestial Studio

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${RED}🛑 Stopping Celestial Studio${NC}"
echo "================================"
echo ""

# Kill by port
echo "Stopping backend (port 8000)..."
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "✅ Backend stopped" || echo "ℹ️  Backend not running"

echo "Stopping frontend (port 5173)..."
lsof -ti:5173 | xargs kill -9 2>/dev/null && echo "✅ Frontend stopped" || echo "ℹ️  Frontend not running"

# Clean up PID files
rm -f logs/*.pid 2>/dev/null

echo ""
echo -e "${GREEN}✅ All services stopped${NC}"
echo ""
echo "To restart: ./start.sh"
echo ""
