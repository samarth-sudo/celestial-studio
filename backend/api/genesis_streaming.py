"""
Genesis Video Streaming API
WebSocket endpoint for real-time Genesis simulation video streaming
"""

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from genesis_service import get_simulation, GenesisSimulation

logger = logging.getLogger(__name__)

router = APIRouter()


class CameraSwitchRequest(BaseModel):
    """Request to switch camera"""
    camera_name: str


class StreamConfig(BaseModel):
    """Video stream configuration"""
    fps: int = 30
    quality: int = 85
    include_depth: bool = False
    include_segmentation: bool = False


@router.websocket("/ws/genesis/video")
async def genesis_video_stream(websocket: WebSocket):
    """
    WebSocket endpoint for Genesis video streaming

    Sends video frames + simulation state at configured FPS

    Message format (from server):
    {
        "type": "frame",
        "image": "<base64-encoded JPEG>",
        "timestamp": 1234567890.123,
        "camera": "main",
        "state": {
            "fps": 60.5,
            "step": 1234,
            "robots": {...}
        }
    }

    Message format (from client):
    {
        "type": "switch_camera",
        "camera": "fpv"
    }
    or
    {
        "type": "config",
        "fps": 30,
        "quality": 85
    }
    """
    await websocket.accept()
    logger.info("🎥 Genesis video stream client connected")

    # Get simulation instance
    simulation: GenesisSimulation = get_simulation()

    # Stream configuration
    stream_fps = 30
    stream_quality = 85
    frame_interval = 1.0 / stream_fps

    try:
        # Start async tasks
        send_task = asyncio.create_task(
            _send_frames(websocket, simulation, stream_fps, stream_quality)
        )
        receive_task = asyncio.create_task(
            _receive_commands(websocket, simulation)
        )

        # Wait for either task to complete (or fail)
        done, pending = await asyncio.wait(
            [send_task, receive_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info("🎥 Video stream client disconnected")
    except Exception as e:
        logger.error(f"Video streaming error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


async def _send_frames(
    websocket: WebSocket,
    simulation: GenesisSimulation,
    fps: int,
    quality: int
):
    """Send video frames to client at specified FPS"""
    frame_interval = 1.0 / fps
    last_send_time = 0.0

    while True:
        try:
            current_time = time.time()

            # Rate limiting
            elapsed = current_time - last_send_time
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)
                continue

            # Get latest frame
            frame_b64 = simulation.get_frame_base64()

            if frame_b64 is None:
                # No frame available yet, wait
                await asyncio.sleep(0.01)
                continue

            # Get simulation state
            state = simulation._get_state()

            # Send frame + state
            message = {
                "type": "frame",
                "image": frame_b64,
                "timestamp": current_time,
                "camera": simulation.active_camera,
                "state": {
                    "fps": state.get("fps", 0),
                    "step": state.get("step", 0),
                    "robots": state.get("robots", {}),
                }
            }

            await websocket.send_json(message)
            last_send_time = current_time

        except Exception as e:
            logger.error(f"Frame send error: {e}")
            break


async def _receive_commands(
    websocket: WebSocket,
    simulation: GenesisSimulation
):
    """Receive commands from client (camera switching, config changes)"""
    while True:
        try:
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get("type")

            if msg_type == "switch_camera":
                camera_name = message.get("camera")
                if camera_name:
                    success = simulation.set_active_camera(camera_name)
                    await websocket.send_json({
                        "type": "camera_switched",
                        "camera": camera_name,
                        "success": success
                    })

            elif msg_type == "get_cameras":
                cameras = simulation.get_available_cameras()
                await websocket.send_json({
                    "type": "cameras_list",
                    "cameras": cameras,
                    "active": simulation.active_camera
                })

            elif msg_type == "attach_fpv":
                robot_id = message.get("robot_id")
                if robot_id:
                    success = simulation.attach_fpv_camera(robot_id)
                    await websocket.send_json({
                        "type": "fpv_attached",
                        "robot_id": robot_id,
                        "success": success
                    })

            else:
                logger.warning(f"Unknown message type: {msg_type}")

        except WebSocketDisconnect:
            break
        except Exception as e:
            logger.error(f"Command receive error: {e}")
            break


@router.get("/api/genesis/cameras")
async def get_cameras():
    """Get list of available cameras"""
    simulation = get_simulation()
    return {
        "cameras": simulation.get_available_cameras(),
        "active": simulation.active_camera
    }


@router.post("/api/genesis/camera/switch")
async def switch_camera(request: CameraSwitchRequest):
    """Switch active camera"""
    simulation = get_simulation()
    success = simulation.set_active_camera(request.camera_name)
    return {
        "success": success,
        "camera": request.camera_name,
        "active": simulation.active_camera
    }
