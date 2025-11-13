#!/bin/bash

# Robust Startup Script for Robotics Demo Platform
# ================================================
# Features:
# - Dependency checking (Ollama, Python venv)
# - Process verification
# - Health checks with HTTP requests
# - Error propagation (exits on failure)
# - Color-coded logging
# - Graceful cleanup on exit

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory (works from any location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Log functions
log_info() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

log_success() {
    echo -e "${GREEN}✓${NC}  $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

log_error() {
    echo -e "${RED}✗${NC}  $1"
}

# Cleanup function
cleanup() {
    log_warning "Cleaning up processes..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 1
}

# Set up trap for cleanup on error
trap cleanup ERR INT TERM

echo ""
log_info "🚀 Starting Robotics Demo Platform..."
echo ""

# ========== Step 1: Dependency Checks ==========
log_info "Checking dependencies..."

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    log_warning "Ollama is not running. Starting Ollama..."
    ollama serve > logs_ollama.txt 2>&1 &
    OLLAMA_PID=$!
    sleep 2

    if ! pgrep -x "ollama" > /dev/null; then
        log_error "Failed to start Ollama. Please install: https://ollama.ai"
        exit 1
    fi
    log_success "Ollama started (PID: $OLLAMA_PID)"
else
    log_success "Ollama is running"
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    log_error "Python virtual environment 'venv' not found"
    log_error "Please create it: python3 -m venv venv"
    exit 1
fi
log_success "Virtual environment found"

# Check if frontend node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    log_warning "Frontend dependencies not installed. Running npm install..."
    cd frontend
    npm install
    if [ $? -ne 0 ]; then
        log_error "Failed to install frontend dependencies"
        exit 1
    fi
    cd ..
    log_success "Frontend dependencies installed"
fi

# ========== Step 2: Kill Existing Processes ==========
log_info "Cleaning up existing processes..."

# Kill processes on ports 8000 and 5173
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

sleep 1
log_success "Port cleanup complete"

# ========== Step 3: Start Backend ==========
echo ""
log_info "📡 Starting backend API on port 8000..."

# Activate virtual environment and start backend
source venv/bin/activate

# Start backend in background
python backend/main.py > logs_backend.txt 2>&1 &
BACKEND_PID=$!

# Verify backend process started
sleep 2
if ! ps -p $BACKEND_PID > /dev/null; then
    log_error "Backend process failed to start"
    log_error "Check logs: tail -f logs_backend.txt"
    tail -20 logs_backend.txt
    exit 1
fi

log_success "Backend process started (PID: $BACKEND_PID)"

# Wait for backend to be ready (health check)
log_info "Waiting for backend to respond..."
BACKEND_READY=false
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        BACKEND_READY=true
        break
    fi
    sleep 1
done

if [ "$BACKEND_READY" = false ]; then
    log_error "Backend failed to respond after 30 seconds"
    log_error "Last 30 lines of backend log:"
    tail -30 logs_backend.txt
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

log_success "Backend is ready and responding"

# ========== Step 4: Start Frontend ==========
echo ""
log_info "🎨 Starting frontend on port 5173..."

cd frontend
npm run dev > ../logs_frontend.txt 2>&1 &
FRONTEND_PID=$!
cd ..

# Verify frontend process started
sleep 2
if ! ps -p $FRONTEND_PID > /dev/null; then
    log_error "Frontend process failed to start"
    log_error "Check logs: tail -f logs_frontend.txt"
    tail -20 logs_frontend.txt
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

log_success "Frontend process started (PID: $FRONTEND_PID)"

# Wait for frontend to be ready
log_info "Waiting for frontend to respond..."
FRONTEND_READY=false
for i in {1..30}; do
    if curl -s http://localhost:5173 > /dev/null 2>&1; then
        FRONTEND_READY=true
        break
    fi
    sleep 1
done

if [ "$FRONTEND_READY" = false ]; then
    log_warning "Frontend not responding yet (still building...)"
    log_info "Frontend may take a few more seconds to be ready"
fi

# ========== Step 5: Summary ==========
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_success "✅ Robotics Demo Platform is running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Frontend:  ${GREEN}http://localhost:5173${NC}"
echo "🔧 Backend:   ${GREEN}http://localhost:8000${NC}"
echo ""
echo "📝 Process IDs:"
echo "   Backend:  $BACKEND_PID"
echo "   Frontend: $FRONTEND_PID"
echo ""
echo "📊 Logs:"
echo "   Backend:  ${BLUE}tail -f logs_backend.txt${NC}"
echo "   Frontend: ${BLUE}tail -f logs_frontend.txt${NC}"
echo ""
echo "🛑 To stop all services:"
echo "   ${YELLOW}kill $BACKEND_PID $FRONTEND_PID${NC}"
echo "   or use: ${YELLOW}./stop.sh${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Keep script running and show live backend logs
log_info "Tailing backend logs (Ctrl+C to exit)..."
tail -f logs_backend.txt
