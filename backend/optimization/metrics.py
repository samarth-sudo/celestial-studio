"""
Algorithm Performance Metrics

Defines metrics for evaluating and comparing algorithm performance
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class BenchmarkResult:
    """Results from running a single benchmark"""
    algorithm_id: str
    algorithm_type: str
    algorithm_name: str

    # Performance metrics
    execution_time_ms: float
    success_rate: float  # 0.0 to 1.0

    # Task-specific metrics
    path_length: Optional[float] = None  # For path planning
    path_smoothness: Optional[float] = None  # For path planning (lower is better)
    collision_count: int = 0  # For obstacle avoidance
    goal_reached: bool = False

    # Quality metrics
    optimality_score: Optional[float] = None  # 0.0 to 1.0 (1.0 = optimal)
    robustness_score: Optional[float] = None  # 0.0 to 1.0

    # Resource usage
    memory_usage_mb: Optional[float] = None
    iterations_count: Optional[int] = None

    # Test details
    test_scenario: str = "default"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None


@dataclass
class AlgorithmMetrics:
    """Aggregated metrics for an algorithm across multiple benchmarks"""
    algorithm_id: str
    algorithm_type: str
    algorithm_name: str
    complexity: str

    # Aggregated statistics
    runs_count: int = 0
    avg_execution_time_ms: float = 0.0
    avg_success_rate: float = 0.0

    # Path planning specific
    avg_path_length: Optional[float] = None
    avg_path_smoothness: Optional[float] = None

    # Obstacle avoidance specific
    total_collisions: int = 0
    avg_collisions_per_run: float = 0.0

    # Quality scores
    avg_optimality_score: float = 0.0
    avg_robustness_score: float = 0.0

    # Variability
    std_dev_execution_time: float = 0.0

    # Overall score (0-100)
    overall_score: float = 0.0

    # Rankings
    rank: int = 0

    # Individual benchmark results
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)


class MetricsCalculator:
    """Calculates and aggregates algorithm metrics"""

    @staticmethod
    def calculate_path_smoothness(path: List[tuple]) -> float:
        """
        Calculate path smoothness (lower is smoother)
        Measures sum of turning angles
        """
        if len(path) < 3:
            return 0.0

        import math

        total_angle_change = 0.0
        for i in range(1, len(path) - 1):
            p1, p2, p3 = path[i-1], path[i], path[i+1]

            # Vector from p1 to p2
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            # Vector from p2 to p3
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            # Calculate angle between vectors
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            angle = abs(math.atan2(det, dot))

            total_angle_change += angle

        # Normalize by path length
        return total_angle_change / (len(path) - 2)

    @staticmethod
    def calculate_path_length(path: List[tuple]) -> float:
        """Calculate total path length"""
        if len(path) < 2:
            return 0.0

        import math

        total_length = 0.0
        for i in range(1, len(path)):
            p1, p2 = path[i-1], path[i]
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += distance

        return total_length

    @staticmethod
    def calculate_optimality_score(
        actual_path_length: float,
        optimal_path_length: float
    ) -> float:
        """
        Calculate how close the path is to optimal
        Returns score from 0.0 to 1.0 (1.0 = optimal)
        """
        if optimal_path_length == 0:
            return 1.0

        ratio = optimal_path_length / max(actual_path_length, 0.001)
        return min(ratio, 1.0)

    @staticmethod
    def aggregate_metrics(
        algorithm_id: str,
        algorithm_type: str,
        algorithm_name: str,
        complexity: str,
        results: List[BenchmarkResult]
    ) -> AlgorithmMetrics:
        """Aggregate multiple benchmark results into overall metrics"""

        if not results:
            return AlgorithmMetrics(
                algorithm_id=algorithm_id,
                algorithm_type=algorithm_type,
                algorithm_name=algorithm_name,
                complexity=complexity
            )

        # Calculate averages
        execution_times = [r.execution_time_ms for r in results]
        success_rates = [r.success_rate for r in results]

        metrics = AlgorithmMetrics(
            algorithm_id=algorithm_id,
            algorithm_type=algorithm_type,
            algorithm_name=algorithm_name,
            complexity=complexity,
            runs_count=len(results),
            avg_execution_time_ms=statistics.mean(execution_times),
            avg_success_rate=statistics.mean(success_rates),
            std_dev_execution_time=statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
        )

        # Path planning metrics
        path_lengths = [r.path_length for r in results if r.path_length is not None]
        if path_lengths:
            metrics.avg_path_length = statistics.mean(path_lengths)

        path_smoothness = [r.path_smoothness for r in results if r.path_smoothness is not None]
        if path_smoothness:
            metrics.avg_path_smoothness = statistics.mean(path_smoothness)

        # Collision metrics
        metrics.total_collisions = sum(r.collision_count for r in results)
        metrics.avg_collisions_per_run = metrics.total_collisions / len(results)

        # Quality scores
        optimality_scores = [r.optimality_score for r in results if r.optimality_score is not None]
        if optimality_scores:
            metrics.avg_optimality_score = statistics.mean(optimality_scores)

        robustness_scores = [r.robustness_score for r in results if r.robustness_score is not None]
        if robustness_scores:
            metrics.avg_robustness_score = statistics.mean(robustness_scores)

        # Calculate overall score (weighted average)
        metrics.overall_score = MetricsCalculator.calculate_overall_score(metrics)

        # Store individual results
        metrics.benchmark_results = results

        return metrics

    @staticmethod
    def calculate_overall_score(metrics: AlgorithmMetrics) -> float:
        """
        Calculate overall algorithm score (0-100)
        Weighted combination of all metrics
        """
        score = 0.0
        weights_sum = 0.0

        # Success rate (30% weight)
        score += metrics.avg_success_rate * 30
        weights_sum += 30

        # Speed (20% weight) - faster is better
        # Normalize execution time (assume 1000ms is baseline)
        if metrics.avg_execution_time_ms > 0:
            speed_score = max(0, 100 - (metrics.avg_execution_time_ms / 10))
            score += (speed_score / 100) * 20
            weights_sum += 20

        # Optimality (25% weight)
        if metrics.avg_optimality_score > 0:
            score += metrics.avg_optimality_score * 25
            weights_sum += 25

        # Robustness (15% weight)
        if metrics.avg_robustness_score > 0:
            score += metrics.avg_robustness_score * 15
            weights_sum += 15

        # Collision avoidance (10% weight) - fewer collisions is better
        collision_score = max(0, 100 - (metrics.avg_collisions_per_run * 10))
        score += (collision_score / 100) * 10
        weights_sum += 10

        # Normalize to 0-100 scale
        if weights_sum > 0:
            return (score / weights_sum) * 100

        return 0.0
