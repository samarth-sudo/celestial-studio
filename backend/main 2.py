"""
FastAPI Backend - Main Application
Provides REST API for robot code generation and simulation
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid
import asyncio
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.code_generator import RoboticsCodeGenerator
from src.generation.robot_types import initialize_robot_types
from src.packaging.package_builder import PythonPackageBuilder
from backend.api import chat, auth, keys, robot_types
from backend.auth.middleware import get_current_user, get_current_user_optional, get_user_id
from backend.models.user import user_db
from backend.database import get_session_store, SessionStore


# Pydantic models
class RobotSpecs(BaseModel):
    """Robot specifications input"""
    robot_name: str = Field(..., description="Name of the robot")
    robot_type: str = Field(default="arm", description="Type of robot (arm, mobile, etc.)")
    dof: int = Field(default=6, description="Degrees of freedom")
    task: str = Field(..., description="Task description")
    platforms: str = Field(default="python", description="Target platform (python, ros, etc.)")
    sensors: Optional[str] = Field(default=None, description="Sensors/accessories")
    communication: Optional[str] = Field(default="serial", description="Communication protocol")
    joint_limits: Optional[List[List[float]]] = Field(default=None, description="Joint limits")
    velocity_limits: Optional[List[float]] = Field(default=None, description="Velocity limits")
    hardware: Optional[Dict[str, str]] = Field(default=None, description="Hardware configuration")


class GenerationRequest(BaseModel):
    """Code generation request"""
    specs: RobotSpecs


class GenerationResponse(BaseModel):
    """Code generation response"""
    session_id: str
    status: str
    message: str
    preview: Optional[str] = None


class SessionStatus(BaseModel):
    """Session status"""
    session_id: str
    status: str
    progress: int  # 0-100
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    hardware_package: Optional[str] = None
    preview: Optional[str] = None
    error: Optional[str] = None


# FastAPI app
app = FastAPI(
    title="Celestial Studio API",
    description="AI-powered robotics platform with neural simulation and RL training",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(keys.router, prefix="/api", tags=["api-keys"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(robot_types.router, prefix="/api/robot-types", tags=["robot-types"])

# Global state
session_store: Optional[SessionStore] = None  # SQLite session storage
generator: Optional[RoboticsCodeGenerator] = None
websocket_connections: List[WebSocket] = []


@app.on_event("startup")
async def startup_event():
    """Initialize code generator, session store, and robot type registry on startup"""
    global generator, session_store
    print("✨ Starting Celestial Studio API...")

    # Initialize session store (SQLite database)
    print("💾 Initializing session database...")
    try:
        session_store = get_session_store()
        stats = session_store.get_stats()
        print(f"✅ Session store ready ({stats['total_sessions']} sessions in database)")
    except Exception as e:
        print(f"❌ Failed to initialize session store: {e}")
        session_store = None

    # Initialize robot type registry
    print("🤖 Initializing robot type registry...")
    try:
        initialize_robot_types()
    except Exception as e:
        print(f"❌ Failed to initialize robot types: {e}")

    print("📥 Loading MLX model...")
    try:
        generator = RoboticsCodeGenerator()
        print("✅ Code generator ready")
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        generator = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Celestial Studio API",
        "version": "2.0.0",
        "status": "ready" if generator else "initializing",
        "features": ["Code Generation", "RAG Chat", "Monaco IDE"],
        "endpoints": {
            "generate": "/api/generate",
            "status": "/api/status/{session_id}",
            "download": "/api/download/{filename}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    active_sessions = 0
    if session_store:
        try:
            stats = session_store.get_stats()
            active_sessions = stats.get('total_sessions', 0)
        except:
            active_sessions = 0

    return {
        "status": "healthy" if generator else "initializing",
        "generator_loaded": generator is not None,
        "session_store_ready": session_store is not None,
        "active_sessions": active_sessions
    }


@app.post("/api/generate", response_model=GenerationResponse)
async def generate_code(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate robot controller code

    This endpoint initiates code generation and returns a session ID.
    Use the session ID to check status and download generated files.
    """
    if not generator:
        raise HTTPException(status_code=503, detail="Code generator not ready")

    # Create session
    session_id = str(uuid.uuid4())

    # Convert Pydantic model to dict
    specs_dict = request.specs.dict()

    # Set defaults only for arm/humanoid robots (not mobile robots)
    robot_type = specs_dict.get('robot_type', 'arm').lower()

    if robot_type != 'mobile':
        # Only set joint limits for arm/humanoid robots
        if not specs_dict.get('joint_limits'):
            dof = specs_dict['dof']
            specs_dict['joint_limits'] = [[-3.14, 3.14]] * dof

        if not specs_dict.get('velocity_limits'):
            dof = specs_dict['dof']
            specs_dict['velocity_limits'] = [1.0] * dof

    if not specs_dict.get('hardware'):
        specs_dict['hardware'] = {}

    # Create session in database
    if not session_store:
        raise HTTPException(status_code=503, detail="Session store not ready")

    session_data = {
        'session_id': session_id,
        'status': 'pending',
        'progress': 0,
        'message': 'Starting code generation...',
        'robot_specs': specs_dict,
        'created_at': datetime.now().isoformat()
    }
    session_store.create_session(session_data)

    # Start generation in background
    background_tasks.add_task(
        generate_code_task,
        session_id,
        specs_dict
    )

    return GenerationResponse(
        session_id=session_id,
        status="pending",
        message="Code generation started. Use session_id to check status.",
        preview=None
    )


async def generate_code_task(
    session_id: str,
    specs: Dict[str, Any]
):
    """Background task for hardware code generation"""
    try:
        # Update status
        session_store.update_session(session_id, {
            'status': 'generating',
            'progress': 10,
            'message': 'Generating robot controller code...'
        })
        await broadcast_status(session_id)

        # Generate hardware controller code
        platform = specs.get('platforms', 'python')
        result = generator.generate_code(specs, platform)

        session_store.update_session(session_id, {
            'progress': 50,
            'message': 'Building hardware package...'
        })
        await broadcast_status(session_id)

        # Build hardware package
        hardware_builder = PythonPackageBuilder()
        hardware_package_path = hardware_builder.build(result['main_code'], specs)

        # Complete
        session_store.update_session(session_id, {
            'status': 'completed',
            'progress': 100,
            'message': 'Code generation complete!',
            'completed_at': datetime.now().isoformat(),
            'hardware_package': hardware_package_path,
            'preview': result['main_code']
        })

        await broadcast_status(session_id)

    except Exception as e:
        session_store.update_session(session_id, {
            'status': 'failed',
            'message': f'Error: {str(e)}',
            'completed_at': datetime.now().isoformat(),
            'error': str(e)
        })
        await broadcast_status(session_id)


@app.get("/api/status/{session_id}", response_model=SessionStatus)
async def get_status(session_id: str):
    """Get generation status"""
    session_data = session_store.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    # Convert database format to SessionStatus model
    return SessionStatus(
        session_id=session_data['session_id'],
        status=session_data['status'],
        progress=session_data['progress'],
        message=session_data['message'],
        created_at=datetime.fromisoformat(session_data['created_at']),
        completed_at=datetime.fromisoformat(session_data['completed_at']) if session_data.get('completed_at') else None,
        hardware_package=session_data.get('hardware_package'),
        preview=session_data.get('preview'),
        error=session_data.get('error')
    )


@app.get("/api/download/{session_id}")
async def download_package(session_id: str):
    """Download generated hardware package"""
    session_data = session_store.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_data['status'] != "completed":
        raise HTTPException(status_code=400, detail="Generation not complete")

    # Get hardware package path
    file_path = session_data.get('hardware_package')
    if not file_path:
        raise HTTPException(status_code=404, detail="Hardware package not found")

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Package file not found")

    return FileResponse(
        path=file_path,
        filename=Path(file_path).name,
        media_type='application/zip'
    )


@app.get("/api/preview/{session_id}")
async def get_preview(session_id: str):
    """Get code preview"""
    session_data = session_store.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session_data.get('preview'):
        raise HTTPException(status_code=400, detail="Preview not available yet")

    return {"preview": session_data['preview']}


@app.get("/api/files/{session_id}")
async def get_all_files(session_id: str):
    """Get all generated files from hardware package"""
    import zipfile
    import os

    session_data = session_store.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session_data.get('hardware_package'):
        raise HTTPException(status_code=400, detail="Hardware package not available yet")

    package_path = session_data['hardware_package']

    if not os.path.exists(package_path):
        raise HTTPException(status_code=404, detail="Package file not found")

    files = []

    try:
        with zipfile.ZipFile(package_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                # Skip directories
                if file_info.is_dir():
                    continue

                # Read file content
                with zip_ref.open(file_info.filename) as f:
                    try:
                        content = f.read().decode('utf-8')
                        files.append({
                            'path': file_info.filename,
                            'content': content,
                            'size': file_info.file_size
                        })
                    except UnicodeDecodeError:
                        # Binary file, skip or mark as binary
                        files.append({
                            'path': file_info.filename,
                            'content': '[Binary file]',
                            'size': file_info.file_size,
                            'is_binary': True
                        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading package: {str(e)}")

    # Generate deployment instructions
    robot_specs = session_data.get('robot_specs', {})
    robot_type = robot_specs.get('robot_type', 'robot')
    task = robot_specs.get('task', 'control task')

    instructions = f"""# Hardware Deployment Instructions

## System Requirements
- Python 3.8 or higher
- pip package manager
- Serial/USB connection to robot hardware

## Installation Steps

### 1. Extract the Package
```bash
unzip {os.path.basename(package_path)}
cd {os.path.basename(package_path).replace('.zip', '')}
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Hardware Connection
Edit `config.yaml` to set your robot's serial port:
```yaml
serial_port: /dev/ttyUSB0  # Linux
# or COM3 for Windows
baud_rate: 115200
```

### 4. Run the Controller
```bash
python main.py
```

## Code Structure

```
.
├── main.py              # Main entry point - starts the robot controller
├── robot_controller.py  # Core controller logic for {robot_type}
├── config.yaml          # Hardware configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## What the Code Does

**Task**: {task}

**Robot Type**: {robot_type}

The generated code provides:
- Hardware interface for robot communication
- Control algorithms for {robot_type}
- Task-specific logic for: {task}
- Safety checks and error handling
- Sensor integration and processing

## Testing

1. **Simulation First** (Recommended):
   - Use the simulation package to test before deploying to hardware
   - Verify all behaviors work as expected

2. **Hardware Test**:
   - Start with robot in safe position
   - Test each function individually
   - Monitor for unexpected behavior
   - Keep emergency stop accessible

## Troubleshooting

**Serial Connection Issues**:
- Check cable connection
- Verify correct port in config.yaml
- Ensure proper permissions: `sudo chmod 666 /dev/ttyUSB0`

**Import Errors**:
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

**Robot Not Responding**:
- Check power supply
- Verify baud rate matches robot firmware
- Test with simple movement commands first

## Customization

You can modify the code by:
1. Editing `robot_controller.py` for control logic
2. Adjusting parameters in `config.yaml`
3. Adding custom functions for specific behaviors

For help, use the AI chat assistant in the web interface!
"""

    return {
        "files": files,
        "instructions": instructions,
        "total_files": len(files),
        "package_name": os.path.basename(package_path)
    }


@app.put("/api/files/{session_id}")
async def save_files(session_id: str, files: Dict[str, Any]):
    """
    Save edited code files back to the package

    Request body:
    {
        "files": [
            {"path": "controller.py", "content": "..."},
            {"path": "control/pid.py", "content": "..."}
        ]
    }
    """
    import zipfile
    import os
    import tempfile
    import shutil

    session_data = session_store.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session_data.get('hardware_package'):
        raise HTTPException(status_code=400, detail="Hardware package not available")

    package_path = session_data['hardware_package']

    if not os.path.exists(package_path):
        raise HTTPException(status_code=404, detail="Package file not found")

    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Extract existing package
        with zipfile.ZipFile(package_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Update modified files
        files_to_update = files.get('files', [])
        updated_count = 0

        for file_update in files_to_update:
            file_path = file_update.get('path')
            file_content = file_update.get('content')

            if not file_path or file_content is None:
                continue

            # Full path to file in extracted directory
            full_path = os.path.join(temp_dir, file_path)

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Write updated content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(file_content)

            updated_count += 1

        # Create new ZIP with updated files
        backup_path = package_path + '.backup'
        shutil.move(package_path, backup_path)

        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for root, dirs, filenames in os.walk(temp_dir):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zip_ref.write(file_path, arcname)

        # Remove backup after successful save
        os.remove(backup_path)

        # Clean up temp directory
        shutil.rmtree(temp_dir)

        # Update session timestamp
        session_store.update_session(session_id, {
            'last_updated': datetime.now().isoformat()
        })

        return {
            "status": "success",
            "saved": updated_count,
            "package_path": package_path,
            "message": f"Successfully saved {updated_count} files"
        }

    except Exception as e:
        # Restore backup if it exists
        backup_path = package_path + '.backup'
        if os.path.exists(backup_path):
            shutil.move(backup_path, package_path)

        raise HTTPException(status_code=500, detail=f"Error saving files: {str(e)}")


@app.post("/api/explain-code")
async def explain_code(request: Dict[str, Any]):
    """
    Use AI to explain selected code snippet

    Request body:
    {
        "code": "def move_joint(...)...",
        "file_path": "controller.py",
        "line_range": [45, 52],
        "session_id": "..." (optional, for context)
    }
    """
    if not generator:
        raise HTTPException(status_code=503, detail="Code generator not ready")

    code = request.get('code')
    file_path = request.get('file_path', 'unknown')
    line_range = request.get('line_range', [])

    if not code:
        raise HTTPException(status_code=400, detail="Code snippet required")

    # Build context-aware prompt
    line_info = f" (lines {line_range[0]}-{line_range[1]})" if len(line_range) == 2 else ""
    prompt = f"""Explain this code snippet from {file_path}{line_info}:

```python
{code}
```

Provide a clear, concise explanation covering:
1. What this code does
2. Key components/functions used
3. Any important implementation details
4. Potential use cases or purpose"""

    try:
        # Use the generator's LLM to explain the code
        explanation = generator.model_interface.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )

        return {
            "status": "success",
            "explanation": explanation,
            "file_path": file_path,
            "line_range": line_range
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error explaining code: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time status updates"""
    await websocket.accept()
    websocket_connections.append(websocket)

    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()

            # Echo back for now
            await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        websocket_connections.remove(websocket)


async def broadcast_status(session_id: str):
    """Broadcast status update to all connected WebSocket clients"""
    session_data = session_store.get_session(session_id)
    if not session_data:
        return

    message = {
        "type": "status_update",
        "session_id": session_id,
        "status": session_data['status'],
        "progress": session_data['progress'],
        "message": session_data['message']
    }

    # Remove disconnected clients
    disconnected = []
    for ws in websocket_connections:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        websocket_connections.remove(ws)


# Mount generated packages directory for static file serving
packages_dir = Path(__file__).parent.parent / "generated_packages"
packages_dir.mkdir(exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    pass  # No cleanup needed

if __name__ == "__main__":
    print("✨ Starting Celestial Studio API server...")
    print("📚 Access API docs at: http://localhost:8000/docs")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled to prevent killing render threads
        log_level="info"
    )
