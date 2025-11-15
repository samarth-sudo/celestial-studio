"""
WebRTC Streaming Server

Streams video from Isaac Lab simulations to browser clients using WebRTC.
Uses aiortc for Python WebRTC implementation.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
import av
import numpy as np


@dataclass
class StreamSession:
    """Represents an active WebRTC streaming session"""
    session_id: str
    peer_connection: RTCPeerConnection
    video_track: Optional[VideoStreamTrack] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class IsaacLabVideoTrack(VideoStreamTrack):
    """
    Video track that streams frames from Isaac Lab simulation
    """

    def __init__(self, video_path: Optional[str] = None):
        super().__init__()
        self.video_path = video_path
        self.frame_queue = asyncio.Queue(maxsize=30)
        self._timestamp = 0
        self._start_time = None

    async def recv(self):
        """
        Receive the next video frame

        Returns:
            av.VideoFrame: Next frame to stream
        """
        if self._start_time is None:
            self._start_time = datetime.now()

        # Get frame from queue (blocks if empty)
        frame_data = await self.frame_queue.get()

        if isinstance(frame_data, np.ndarray):
            # Convert numpy array to av.VideoFrame
            frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
        elif isinstance(frame_data, av.VideoFrame):
            frame = frame_data
        else:
            raise ValueError(f"Unsupported frame type: {type(frame_data)}")

        # Set presentation timestamp
        frame.pts = self._timestamp
        frame.time_base = av.Rational(1, 30)  # 30 FPS
        self._timestamp += 1

        return frame

    async def add_frame(self, frame: np.ndarray):
        """
        Add a frame to the streaming queue

        Args:
            frame: Numpy array (H, W, 3) in RGB format
        """
        try:
            # Non-blocking put with timeout
            await asyncio.wait_for(
                self.frame_queue.put(frame),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            # Drop frame if queue is full (prevents blocking simulation)
            pass


class WebRTCStreamingServer:
    """
    Manages WebRTC streaming sessions for Isaac Lab simulations
    """

    def __init__(self):
        self.sessions: Dict[str, StreamSession] = {}
        self.relay = MediaRelay()

    async def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new streaming session

        Args:
            metadata: Optional metadata about the simulation

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())

        peer_connection = RTCPeerConnection()

        session = StreamSession(
            session_id=session_id,
            peer_connection=peer_connection,
            metadata=metadata or {}
        )

        self.sessions[session_id] = session

        # Set up connection state handlers
        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state: {peer_connection.connectionState}")
            if peer_connection.connectionState == "failed":
                await self.close_session(session_id)

        @peer_connection.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(f"ICE connection state: {peer_connection.iceConnectionState}")

        print(f"Created streaming session: {session_id}")
        return session_id

    async def handle_offer(
        self,
        session_id: str,
        offer_sdp: Dict[str, str],
        video_source: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Handle WebRTC offer from client

        Args:
            session_id: Session identifier
            offer_sdp: SDP offer from client
            video_source: Optional video file path or stream source

        Returns:
            SDP answer for client
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        pc = session.peer_connection

        # Create offer object
        offer = RTCSessionDescription(
            sdp=offer_sdp["sdp"],
            type=offer_sdp["type"]
        )

        # Set remote description (client's offer)
        await pc.setRemoteDescription(offer)

        # Create video track
        if video_source:
            # Stream from file (for recorded simulations)
            player = MediaPlayer(video_source)
            video_track = player.video
        else:
            # Create live streaming track
            video_track = IsaacLabVideoTrack()

        session.video_track = video_track

        # Add video track to peer connection
        pc.addTrack(self.relay.subscribe(video_track))

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Return answer SDP
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }

    async def stream_frame(self, session_id: str, frame: np.ndarray):
        """
        Stream a single frame to the client

        Args:
            session_id: Session identifier
            frame: Frame data as numpy array (H, W, 3)
        """
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]

        if isinstance(session.video_track, IsaacLabVideoTrack):
            await session.video_track.add_frame(frame)

    async def close_session(self, session_id: str):
        """Close a streaming session"""
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]

        # Close peer connection
        await session.peer_connection.close()

        # Remove from active sessions
        del self.sessions[session_id]

        print(f"Closed streaming session: {session_id}")

    async def close_all_sessions(self):
        """Close all active sessions"""
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)

    def get_session(self, session_id: str) -> Optional[StreamSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[str]:
        """List all active session IDs"""
        return list(self.sessions.keys())


# Global streaming server instance
_streaming_server: Optional[WebRTCStreamingServer] = None


def get_streaming_server() -> WebRTCStreamingServer:
    """Get singleton streaming server instance"""
    global _streaming_server
    if _streaming_server is None:
        _streaming_server = WebRTCStreamingServer()
    return _streaming_server


# FastAPI integration helpers

async def create_webrtc_session(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Create WebRTC session (FastAPI endpoint helper)

    Returns:
        Dict with session_id
    """
    server = get_streaming_server()
    session_id = await server.create_session(metadata)

    return {
        "session_id": session_id,
        "status": "created"
    }


async def handle_webrtc_offer(
    session_id: str,
    offer: Dict[str, str],
    video_source: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle WebRTC offer (FastAPI endpoint helper)

    Args:
        session_id: Session identifier
        offer: Client's SDP offer
        video_source: Optional video file path

    Returns:
        SDP answer for client
    """
    server = get_streaming_server()

    try:
        answer = await server.handle_offer(session_id, offer, video_source)
        return {
            "success": True,
            "answer": answer
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def close_webrtc_session(session_id: str) -> Dict[str, str]:
    """
    Close WebRTC session (FastAPI endpoint helper)

    Returns:
        Status dict
    """
    server = get_streaming_server()
    await server.close_session(session_id)

    return {
        "status": "closed",
        "session_id": session_id
    }


# Modal integration for streaming from Isaac Lab

async def stream_isaac_simulation(
    session_id: str,
    simulation_result: Dict[str, Any]
):
    """
    Stream recorded Isaac Lab simulation

    Args:
        session_id: WebRTC session ID
        simulation_result: Result from Modal's run_isaac_simulation
    """
    server = get_streaming_server()

    video_path = simulation_result.get('video_path')
    if not video_path:
        print("No video path in simulation result")
        return

    # For recorded videos, we'd need to:
    # 1. Download video from Modal volume
    # 2. Stream it via WebRTC
    # This is implemented when handling offers with video_source parameter

    print(f"Streaming video from: {video_path}")


async def stream_live_isaac_frames(
    session_id: str,
    frame_generator
):
    """
    Stream live frames from Isaac Lab simulation

    Args:
        session_id: WebRTC session ID
        frame_generator: Async generator yielding frames
    """
    server = get_streaming_server()

    async for frame in frame_generator:
        await server.stream_frame(session_id, frame)
