"""
Algorithm Comparator

Compares multiple algorithms and selects the best one based on metrics
"""

from typing import List, Dict, Any, Tuple
from .metrics import AlgorithmMetrics, BenchmarkResult, MetricsCalculator
from .benchmark import AlgorithmBenchmark


class AlgorithmComparator:
    """Compare algorithms and select the best one"""

    def __init__(self):
        self.benchmark = AlgorithmBenchmark()

    def compare_algorithms(
        self,
        algorithms: List[Dict[str, Any]],
        scenario: str = "all",
        runs_per_scenario: int = 3
    ) -> Dict[str, Any]:
        """
        Compare multiple algorithms and rank them

        Args:
            algorithms: List of algorithm objects
            scenario: Test scenario to use
            runs_per_scenario: Number of runs per scenario

        Returns:
            Comparison results with rankings and recommendations
        """
        if not algorithms:
            raise ValueError("No algorithms provided for comparison")

        print(f"📊 Comparing {len(algorithms)} algorithms...")

        # Benchmark all algorithms
        all_metrics = []
        for algo in algorithms:
            results = self.benchmark.benchmark_algorithm(
                algorithm=algo,
                scenario_name=scenario,
                runs_per_scenario=runs_per_scenario
            )

            metrics = MetricsCalculator.aggregate_metrics(
                algorithm_id=algo.get('id', 'unknown'),
                algorithm_type=algo.get('type', 'unknown'),
                algorithm_name=algo.get('name', 'Unknown'),
                complexity=algo.get('complexity', 'Unknown'),
                results=results
            )

            all_metrics.append(metrics)

        # Rank algorithms by overall score
        all_metrics.sort(key=lambda m: m.overall_score, reverse=True)

        # Assign ranks
        for i, metrics in enumerate(all_metrics):
            metrics.rank = i + 1

        # Select best algorithm
        best = all_metrics[0] if all_metrics else None

        # Generate comparison report
        comparison = {
            "best_algorithm": self._format_algorithm_summary(best) if best else None,
            "rankings": [self._format_algorithm_summary(m) for m in all_metrics],
            "comparison_matrix": self._create_comparison_matrix(all_metrics),
            "recommendation": self._generate_recommendation(all_metrics),
            "test_info": {
                "scenario": scenario,
                "runs_per_scenario": runs_per_scenario,
                "total_tests": len(algorithms) * runs_per_scenario * (
                    len(self.benchmark.scenarios) if scenario == "all" else 1
                )
            }
        }

        print(f"✅ Comparison complete. Best: {best.algorithm_name if best else 'None'}")

        return comparison

    def select_best_algorithm(
        self,
        algorithms: List[Dict[str, Any]],
        criteria: str = "overall"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Select the best algorithm based on specific criteria

        Args:
            algorithms: List of algorithm objects
            criteria: "overall", "speed", "accuracy", "robustness"

        Returns:
            (best_algorithm, comparison_data)
        """
        comparison = self.compare_algorithms(algorithms)

        if criteria == "overall":
            best = comparison["best_algorithm"]
        elif criteria == "speed":
            # Find fastest algorithm
            rankings = comparison["rankings"]
            best = min(rankings, key=lambda a: a["avg_execution_time_ms"])
        elif criteria == "accuracy":
            # Find most accurate
            rankings = comparison["rankings"]
            best = max(rankings, key=lambda a: a["avg_success_rate"])
        elif criteria == "robustness":
            # Find most robust
            rankings = comparison["rankings"]
            best = max(rankings, key=lambda a: a.get("avg_robustness_score", 0))
        else:
            raise ValueError(f"Unknown criteria: {criteria}")

        # Find the full algorithm object
        best_algo = next(
            (a for a in algorithms if a.get('id') == best["algorithm_id"]),
            None
        )

        return best_algo, comparison

    def _format_algorithm_summary(self, metrics: AlgorithmMetrics) -> Dict[str, Any]:
        """Format metrics as summary dict"""
        return {
            "algorithm_id": metrics.algorithm_id,
            "algorithm_name": metrics.algorithm_name,
            "algorithm_type": metrics.algorithm_type,
            "complexity": metrics.complexity,
            "rank": metrics.rank,
            "overall_score": round(metrics.overall_score, 2),
            "avg_execution_time_ms": round(metrics.avg_execution_time_ms, 2),
            "avg_success_rate": round(metrics.avg_success_rate, 3),
            "avg_path_length": round(metrics.avg_path_length, 2) if metrics.avg_path_length else None,
            "avg_path_smoothness": round(metrics.avg_path_smoothness, 3) if metrics.avg_path_smoothness else None,
            "avg_collisions_per_run": round(metrics.avg_collisions_per_run, 2),
            "avg_optimality_score": round(metrics.avg_optimality_score, 3),
            "runs_count": metrics.runs_count
        }

    def _create_comparison_matrix(
        self,
        all_metrics: List[AlgorithmMetrics]
    ) -> Dict[str, List[Any]]:
        """Create a comparison matrix for visualization"""
        matrix = {
            "algorithms": [m.algorithm_name for m in all_metrics],
            "overall_scores": [round(m.overall_score, 1) for m in all_metrics],
            "execution_times": [round(m.avg_execution_time_ms, 1) for m in all_metrics],
            "success_rates": [round(m.avg_success_rate * 100, 1) for m in all_metrics],
            "collision_rates": [round(m.avg_collisions_per_run, 1) for m in all_metrics]
        }

        # Add path-specific metrics if available
        path_lengths = [
            round(m.avg_path_length, 1) if m.avg_path_length else None
            for m in all_metrics
        ]
        if any(pl is not None for pl in path_lengths):
            matrix["path_lengths"] = path_lengths

        return matrix

    def _generate_recommendation(
        self,
        all_metrics: List[AlgorithmMetrics]
    ) -> Dict[str, Any]:
        """Generate recommendation based on comparison"""
        if not all_metrics:
            return {
                "message": "No algorithms to compare",
                "confidence": 0.0
            }

        best = all_metrics[0]
        second_best = all_metrics[1] if len(all_metrics) > 1 else None

        # Calculate confidence
        if second_best:
            score_diff = best.overall_score - second_best.overall_score
            confidence = min(1.0, score_diff / 50)  # Normalize to 0-1
        else:
            confidence = 1.0

        # Generate reasons
        reasons = []

        if best.avg_success_rate > 0.9:
            reasons.append(f"High success rate ({best.avg_success_rate * 100:.0f}%)")

        if best.avg_execution_time_ms < 50:
            reasons.append(f"Fast execution ({best.avg_execution_time_ms:.1f}ms)")

        if best.avg_collisions_per_run < 1:
            reasons.append("Low collision rate")

        if best.avg_optimality_score > 0.8:
            reasons.append("Near-optimal path quality")

        if not reasons:
            reasons.append("Best overall performance across metrics")

        recommendation = {
            "algorithm_name": best.algorithm_name,
            "algorithm_id": best.algorithm_id,
            "confidence": round(confidence, 2),
            "reasons": reasons,
            "message": self._format_recommendation_message(best, second_best, reasons),
            "alternative": second_best.algorithm_name if second_best else None
        }

        return recommendation

    def _format_recommendation_message(
        self,
        best: AlgorithmMetrics,
        second_best: AlgorithmMetrics | None,
        reasons: List[str]
    ) -> str:
        """Format recommendation as human-readable message"""
        message = f"**Recommended:** {best.algorithm_name}\n\n"
        message += "**Why this algorithm:**\n"
        for reason in reasons:
            message += f"• {reason}\n"

        if second_best:
            message += f"\n**Alternative:** {second_best.algorithm_name} "
            message += f"(score: {second_best.overall_score:.1f} vs {best.overall_score:.1f})"

        return message

    def compare_by_type(
        self,
        algorithms: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group algorithms by type and find best in each category

        Returns:
            Dict of {algorithm_type: comparison_data}
        """
        # Group by type
        by_type = {}
        for algo in algorithms:
            algo_type = algo.get('type', 'unknown')
            if algo_type not in by_type:
                by_type[algo_type] = []
            by_type[algo_type].append(algo)

        # Compare within each type
        results = {}
        for algo_type, algos in by_type.items():
            if len(algos) > 1:
                comparison = self.compare_algorithms(algos)
                results[algo_type] = comparison
            elif len(algos) == 1:
                # Only one algorithm of this type
                results[algo_type] = {
                    "best_algorithm": {
                        "algorithm_id": algos[0].get('id'),
                        "algorithm_name": algos[0].get('name'),
                        "algorithm_type": algo_type,
                        "message": "Only algorithm of this type"
                    },
                    "rankings": [],
                    "comparison_matrix": {},
                    "recommendation": {
                        "message": f"Using {algos[0].get('name')} as the only {algo_type} algorithm",
                        "confidence": 1.0
                    }
                }

        return results
