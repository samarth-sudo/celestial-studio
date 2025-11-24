"""
Genesis Renderer and Video Streaming
Handles frame capture, encoding, compression, and streaming to frontend.
"""

import asyncio
import base64
import io
import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VideoCodec(Enum):
    """Supported video codecs"""
    MJPEG = "mjpeg"  # Motion JPEG (simple, compatible)
    H264 = "h264"    # H.264 (efficient, widely supported)
    VP9 = "vp9"      # VP9 (open source, efficient)
    RAW = "raw"      # Raw frames (testing only)


class StreamQuality(Enum):
    """Video quality presets"""
    DRAFT = "draft"      # 720p, 30fps, low quality
    MEDIUM = "medium"    # 1080p, 30fps, medium quality
    HIGH = "high"        # 1080p, 60fps, high quality
    ULTRA = "ultra"      # 4K, 60fps, ultra quality


@dataclass
class StreamConfig:
    """Streaming configuration"""
    codec: VideoCodec = VideoCodec.MJPEG
    quality: StreamQuality = StreamQuality.MEDIUM

    # Resolution
    width: int = 1920
    height: int = 1080

    # Frame rate
    fps: int = 30

    # JPEG quality (for MJPEG)
    jpeg_quality: int = 85

    # Buffer settings
    max_buffer_size: int = 30  # frames

    @classmethod
    def from_quality(cls, quality: StreamQuality) -> "StreamConfig":
        """Create config from quality preset"""
        presets = {
            StreamQuality.DRAFT: cls(
                width=1280,
                height=720,
                fps=30,
                jpeg_quality=70,
            ),
            StreamQuality.MEDIUM: cls(
                width=1920,
                height=1080,
                fps=30,
                jpeg_quality=85,
            ),
            StreamQuality.HIGH: cls(
                width=1920,
                height=1080,
                fps=60,
                jpeg_quality=90,
            ),
            StreamQuality.ULTRA: cls(
                width=3840,
                height=2160,
                fps=60,
                jpeg_quality=95,
            ),
        }
        return presets.get(quality, cls())


class FrameBuffer:
    """
    Thread-safe frame buffer for video streaming
    Handles frame queuing, buffering, and retrieval
    """

    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.buffer = queue.Queue(maxsize=max_size)
        self.frame_count = 0
        self.dropped_frames = 0
        self.lock = threading.Lock()

        # Statistics
        self.total_bytes = 0
        self.encoding_times = []

    def add_frame(self, frame: np.ndarray, timestamp: float):
        """Add frame to buffer (non-blocking)"""
        try:
            self.buffer.put_nowait((frame, timestamp))
            self.frame_count += 1
        except queue.Full:
            # Drop oldest frame if buffer is full
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait((frame, timestamp))
                self.dropped_frames += 1
            except:
                pass

    def get_frame(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """Get next frame from buffer (blocking with timeout)"""
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get latest frame, discarding intermediate frames"""
        frame = None

        # Drain buffer to get latest
        while not self.buffer.empty():
            try:
                frame = self.buffer.get_nowait()
            except queue.Empty:
                break

        return frame

    def clear(self):
        """Clear all frames from buffer"""
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    break

    def get_stats(self) -> dict:
        """Get buffer statistics"""
        return {
            'buffer_size': self.buffer.qsize(),
            'max_size': self.max_size,
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'drop_rate': self.dropped_frames / max(1, self.frame_count),
        }


class VideoEncoder:
    """
    Video encoder for Genesis frames
    Supports multiple codecs and quality settings
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self.frame_count = 0
        self.total_encode_time = 0.0

    def encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode frame to bytes"""
        start_time = time.time()

        encoded = None

        if self.config.codec == VideoCodec.MJPEG:
            encoded = self._encode_jpeg(frame)
        elif self.config.codec == VideoCodec.RAW:
            encoded = frame.tobytes()
        else:
            logger.warning(f"Codec {self.config.codec} not yet implemented, falling back to JPEG")
            encoded = self._encode_jpeg(frame)

        encode_time = time.time() - start_time
        self.total_encode_time += encode_time
        self.frame_count += 1

        return encoded

    def encode_frame_base64(self, frame: np.ndarray) -> Optional[str]:
        """Encode frame to base64 string"""
        encoded = self.encode_frame(frame)
        if encoded is None:
            return None

        return base64.b64encode(encoded).decode('utf-8')

    def _encode_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode frame as JPEG"""
        try:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                # Assume float in range [0, 1]
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

            # Resize if needed
            if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
                image = Image.fromarray(frame)
                image = image.resize((self.config.width, self.config.height), Image.Resampling.LANCZOS)
            else:
                image = Image.fromarray(frame)

            # Encode to JPEG
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.config.jpeg_quality, optimize=True)
            buffer.seek(0)

            return buffer.read()

        except Exception as e:
            logger.error(f"JPEG encoding failed: {e}")
            return None

    def get_stats(self) -> dict:
        """Get encoder statistics"""
        avg_encode_time = self.total_encode_time / max(1, self.frame_count)

        return {
            'codec': self.config.codec.value,
            'frame_count': self.frame_count,
            'avg_encode_time_ms': avg_encode_time * 1000,
            'theoretical_fps': 1.0 / max(0.001, avg_encode_time),
        }


class VideoStreamer:
    """
    Video streaming manager
    Handles frame capture, encoding, buffering, and delivery
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self.encoder = VideoEncoder(config)
        self.buffer = FrameBuffer(max_size=config.max_buffer_size)

        self.is_running = False
        self.encoding_thread: Optional[threading.Thread] = None

        # Statistics
        self.stream_start_time = 0.0
        self.frames_delivered = 0
        self.bytes_delivered = 0

    def start(self):
        """Start the video streamer"""
        if self.is_running:
            logger.warning("Streamer already running")
            return

        self.is_running = True
        self.stream_start_time = time.time()

        # Start encoding thread
        self.encoding_thread = threading.Thread(target=self._encoding_loop, daemon=True)
        self.encoding_thread.start()

        logger.info(f"🎥 Video streamer started ({self.config.codec.value}, {self.config.width}x{self.config.height} @ {self.config.fps}fps)")

    def stop(self):
        """Stop the video streamer"""
        self.is_running = False

        if self.encoding_thread is not None:
            self.encoding_thread.join(timeout=2.0)
            self.encoding_thread = None

        logger.info("⏹️  Video streamer stopped")

    def add_frame(self, frame: np.ndarray):
        """Add frame to encoding queue"""
        timestamp = time.time()
        self.buffer.add_frame(frame, timestamp)

    def _encoding_loop(self):
        """Background encoding loop (runs in thread)"""
        logger.info("Encoding thread started")

        while self.is_running:
            # Get frame from buffer
            result = self.buffer.get_frame(timeout=0.1)

            if result is None:
                continue

            frame, timestamp = result

            # Encode frame
            encoded = self.encoder.encode_frame(frame)

            if encoded is not None:
                self.bytes_delivered += len(encoded)

        logger.info("Encoding thread stopped")

    def get_latest_frame_jpeg(self) -> Optional[bytes]:
        """Get latest frame as JPEG bytes"""
        result = self.buffer.get_latest_frame()

        if result is None:
            return None

        frame, timestamp = result

        # Encode immediately
        encoded = self.encoder.encode_frame(frame)

        if encoded is not None:
            self.frames_delivered += 1
            self.bytes_delivered += len(encoded)

        return encoded

    def get_latest_frame_base64(self) -> Optional[str]:
        """Get latest frame as base64 string"""
        jpeg_bytes = self.get_latest_frame_jpeg()

        if jpeg_bytes is None:
            return None

        return base64.b64encode(jpeg_bytes).decode('utf-8')

    def get_stats(self) -> dict:
        """Get streaming statistics"""
        runtime = time.time() - self.stream_start_time

        buffer_stats = self.buffer.get_stats()
        encoder_stats = self.encoder.get_stats()

        return {
            'runtime_seconds': runtime,
            'frames_delivered': self.frames_delivered,
            'bytes_delivered': self.bytes_delivered,
            'avg_fps': self.frames_delivered / max(0.1, runtime),
            'avg_bitrate_mbps': (self.bytes_delivered * 8) / max(0.1, runtime) / 1_000_000,
            'buffer': buffer_stats,
            'encoder': encoder_stats,
        }


class AdaptiveStreamer(VideoStreamer):
    """
    Adaptive quality video streamer
    Automatically adjusts quality based on latency and bandwidth
    """

    def __init__(self, initial_quality: StreamQuality = StreamQuality.MEDIUM):
        config = StreamConfig.from_quality(initial_quality)
        super().__init__(config)

        self.current_quality = initial_quality
        self.target_latency_ms = 100  # Target 100ms latency

        # Adaptation state
        self.recent_latencies = []
        self.max_latency_samples = 30

    def measure_latency(self, frame_timestamp: float) -> float:
        """Measure latency from frame capture to delivery"""
        latency = (time.time() - frame_timestamp) * 1000  # ms
        self.recent_latencies.append(latency)

        # Keep only recent samples
        if len(self.recent_latencies) > self.max_latency_samples:
            self.recent_latencies.pop(0)

        return latency

    def adapt_quality(self):
        """Adapt streaming quality based on latency"""
        if len(self.recent_latencies) < 10:
            return  # Not enough data

        avg_latency = np.mean(self.recent_latencies)

        # If latency too high, reduce quality
        if avg_latency > self.target_latency_ms * 1.5:
            if self.current_quality == StreamQuality.ULTRA:
                self._switch_quality(StreamQuality.HIGH)
            elif self.current_quality == StreamQuality.HIGH:
                self._switch_quality(StreamQuality.MEDIUM)
            elif self.current_quality == StreamQuality.MEDIUM:
                self._switch_quality(StreamQuality.DRAFT)

        # If latency low, increase quality
        elif avg_latency < self.target_latency_ms * 0.5:
            if self.current_quality == StreamQuality.DRAFT:
                self._switch_quality(StreamQuality.MEDIUM)
            elif self.current_quality == StreamQuality.MEDIUM:
                self._switch_quality(StreamQuality.HIGH)
            elif self.current_quality == StreamQuality.HIGH:
                self._switch_quality(StreamQuality.ULTRA)

    def _switch_quality(self, new_quality: StreamQuality):
        """Switch to new quality preset"""
        if new_quality == self.current_quality:
            return

        logger.info(f"📊 Adapting quality: {self.current_quality.value} → {new_quality.value}")

        # Update config
        new_config = StreamConfig.from_quality(new_quality)
        self.config = new_config
        self.encoder.config = new_config
        self.current_quality = new_quality

        # Clear recent latencies to avoid oscillation
        self.recent_latencies.clear()
