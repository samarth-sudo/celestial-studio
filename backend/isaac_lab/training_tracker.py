"""
Training Progress Tracker

Tracks RL training progress and streams updates via Server-Sent Events (SSE).
"""

import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class TrainingProgress:
    """Training progress data"""
    training_id: str
    status: str  # 'starting', 'running', 'completed', 'failed'
    current_iteration: int = 0
    total_iterations: int = 0
    progress_percent: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    elapsed_time: float = 0.0
    estimated_time_remaining: float = 0.0
    message: str = ""
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'training_id': self.training_id,
            'status': self.status,
            'current_iteration': self.current_iteration,
            'total_iterations': self.total_iterations,
            'progress_percent': self.progress_percent,
            'metrics': self.metrics,
            'elapsed_time': self.elapsed_time,
            'estimated_time_remaining': self.estimated_time_remaining,
            'message': self.message,
            'error': self.error,
            'timestamp': self.timestamp
        }


class TrainingProgressTracker:
    """Tracks training progress for multiple concurrent training jobs"""

    def __init__(self):
        self.training_sessions: Dict[str, TrainingProgress] = {}
        self.update_queues: Dict[str, asyncio.Queue] = {}

    def create_session(
        self,
        task_name: str,
        total_iterations: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new training session

        Args:
            task_name: Name of the training task
            total_iterations: Total number of training iterations

        Returns:
            Training session ID
        """
        training_id = str(uuid.uuid4())

        progress = TrainingProgress(
            training_id=training_id,
            status='starting',
            total_iterations=total_iterations,
            message=f"Starting training for {task_name}..."
        )

        self.training_sessions[training_id] = progress
        self.update_queues[training_id] = asyncio.Queue()

        print(f"Created training session: {training_id} for {task_name}")

        return training_id

    async def update_progress(
        self,
        training_id: str,
        current_iteration: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        message: Optional[str] = None,
        elapsed_time: Optional[float] = None
    ):
        """
        Update training progress

        Args:
            training_id: Training session ID
            current_iteration: Current iteration number
            metrics: Training metrics (loss, reward, etc.)
            status: Training status
            message: Status message
            elapsed_time: Elapsed time in seconds
        """
        if training_id not in self.training_sessions:
            return

        progress = self.training_sessions[training_id]

        if current_iteration is not None:
            progress.current_iteration = current_iteration
            progress.progress_percent = (current_iteration / progress.total_iterations) * 100

        if metrics is not None:
            progress.metrics.update(metrics)

        if status is not None:
            progress.status = status

        if message is not None:
            progress.message = message

        if elapsed_time is not None:
            progress.elapsed_time = elapsed_time

            # Estimate time remaining
            if current_iteration and current_iteration > 0:
                time_per_iteration = elapsed_time / current_iteration
                remaining_iterations = progress.total_iterations - current_iteration
                progress.estimated_time_remaining = time_per_iteration * remaining_iterations

        progress.timestamp = datetime.utcnow().isoformat()

        # Add update to queue
        if training_id in self.update_queues:
            await self.update_queues[training_id].put(progress.to_dict())

    async def set_completed(
        self,
        training_id: str,
        model_path: Optional[str] = None,
        final_metrics: Optional[Dict[str, Any]] = None
    ):
        """Mark training as completed"""
        await self.update_progress(
            training_id,
            status='completed',
            message="Training completed successfully!",
            metrics=final_metrics or {}
        )

        if model_path:
            self.training_sessions[training_id].metrics['model_path'] = model_path

    async def set_failed(self, training_id: str, error: str):
        """Mark training as failed"""
        if training_id in self.training_sessions:
            progress = self.training_sessions[training_id]
            progress.status = 'failed'
            progress.error = error
            progress.message = f"Training failed: {error}"
            progress.timestamp = datetime.utcnow().isoformat()

            if training_id in self.update_queues:
                await self.update_queues[training_id].put(progress.to_dict())

    async def stream_progress(self, training_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream training progress updates via async generator (for SSE)

        Args:
            training_id: Training session ID

        Yields:
            Progress updates as dictionaries
        """
        if training_id not in self.training_sessions:
            yield {'error': 'Training session not found'}
            return

        if training_id not in self.update_queues:
            self.update_queues[training_id] = asyncio.Queue()

        queue = self.update_queues[training_id]

        # Send initial state
        yield self.training_sessions[training_id].to_dict()

        # Stream updates until training completes or fails
        while True:
            try:
                # Wait for next update with timeout
                update = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield update

                # Stop streaming if training completed or failed
                if update.get('status') in ['completed', 'failed']:
                    break

            except asyncio.TimeoutError:
                # Send keepalive
                yield {'type': 'keepalive', 'timestamp': datetime.utcnow().isoformat()}

    def get_progress(self, training_id: str) -> Optional[TrainingProgress]:
        """Get current training progress"""
        return self.training_sessions.get(training_id)

    def close_session(self, training_id: str):
        """Close training session and cleanup"""
        if training_id in self.training_sessions:
            del self.training_sessions[training_id]

        if training_id in self.update_queues:
            del self.update_queues[training_id]

        print(f"Closed training session: {training_id}")


# Global tracker instance
_training_tracker: Optional[TrainingProgressTracker] = None


def get_training_tracker() -> TrainingProgressTracker:
    """Get singleton training tracker instance"""
    global _training_tracker
    if _training_tracker is None:
        _training_tracker = TrainingProgressTracker()
    return _training_tracker
