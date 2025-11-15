# 🚀 Quick Start Guide

## Starting the Platform

Simply run:

```bash
./start.sh
```

This will automatically:
- ✅ Check and start Ollama if needed
- ✅ Verify Python virtual environment
- ✅ Install frontend dependencies if missing
- ✅ Clean up any existing processes on ports 8000/5173
- ✅ Start backend API (with auto-reload)
- ✅ Start frontend dev server
- ✅ Run health checks to ensure everything is ready

The script will display live backend logs. Press `Ctrl+C` when done.

## Stopping the Platform

```bash
./stop.sh
```

This gracefully stops all services:
- Backend (port 8000)
- Frontend (ports 5173/5174)
- Ollama (if started by start.sh)

## Accessing the Platform

Once started:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Logs

View logs in real-time:

```bash
# Backend logs
tail -f logs_backend.txt

# Frontend logs
tail -f logs_frontend.txt

# Ollama logs (if started by script)
tail -f logs_ollama.txt
```

## Troubleshooting

### Port Already in Use

If you get port conflicts, stop existing processes:

```bash
./stop.sh
```

Then start again:

```bash
./start.sh
```

### Ollama Not Found

Install Ollama:
```bash
brew install ollama
# or download from https://ollama.ai
```

### Backend Won't Start

Check the logs:
```bash
tail -50 logs_backend.txt
```

Make sure virtual environment is activated:
```bash
source venv/bin/activate
cd backend
pip install -r requirements.txt
```

### Frontend Won't Start

Check the logs:
```bash
tail -50 logs_frontend.txt
```

Reinstall dependencies:
```bash
cd frontend
rm -rf node_modules
npm install
```

## Manual Start (if needed)

If you prefer to run services separately:

### Terminal 1: Ollama
```bash
ollama serve
```

### Terminal 2: Backend
```bash
source venv/bin/activate
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 3: Frontend
```bash
cd frontend
npm run dev
```

## That's It! 🎉

The platform should now be running. Open http://localhost:5173 and start creating simulations!
