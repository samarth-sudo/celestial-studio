#!/bin/bash

# Stop Script for Robotics Demo Platform
# ========================================
# Gracefully stops all running services

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

echo ""
log_info "🛑 Stopping Robotics Demo Platform..."
echo ""

# Stop processes on backend port (8000)
if lsof -ti:8000 > /dev/null 2>&1; then
    log_info "Stopping backend (port 8000)..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    log_success "Backend stopped"
else
    log_info "No backend process found on port 8000"
fi

# Stop processes on frontend ports (5173, 5174)
for port in 5173 5174; do
    if lsof -ti:$port > /dev/null 2>&1; then
        log_info "Stopping frontend (port $port)..."
        lsof -ti:$port | xargs kill -9 2>/dev/null
        log_success "Frontend stopped (port $port)"
    fi
done

# Stop Ollama if it was started by start.sh
if [ -f ".ollama_pid" ]; then
    OLLAMA_PID=$(cat .ollama_pid)
    if ps -p $OLLAMA_PID > /dev/null 2>&1; then
        log_info "Stopping Ollama (PID: $OLLAMA_PID)..."
        kill $OLLAMA_PID 2>/dev/null
        rm .ollama_pid
        log_success "Ollama stopped"
    else
        rm .ollama_pid
    fi
fi

# Clean up log files (optional - comment out if you want to keep logs)
# log_info "Cleaning up log files..."
# rm -f logs_backend.txt logs_frontend.txt logs_ollama.txt

echo ""
log_success "✅ All services stopped"
echo ""
