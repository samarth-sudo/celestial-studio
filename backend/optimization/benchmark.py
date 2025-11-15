"""
Algorithm Benchmarking Engine

Runs performance tests on algorithms and collects metrics
"""

import time
import random
from typing import Dict, List, Any, Optional, Tuple
from .metrics import BenchmarkResult, MetricsCalculator


class AlgorithmBenchmark:
    """Benchmark algorithms across different scenarios"""

    def __init__(self):
        self.scenarios = self._create_test_scenarios()

    def _create_test_scenarios(self) -> List[Dict[str, Any]]:
        """Create standard test scenarios for benchmarking"""
        return [
            {
                "name": "simple",
                "description": "Simple environment with few obstacles",
                "start": (0, 0),
                "goal": (10, 10),
                "obstacles": [
                    {"position": (5, 5), "radius": 1}
                ],
                "world_bounds": {"minX": -5, "maxX": 15, "minZ": -5, "maxZ": 15}
            },
            {
                "name": "medium",
                "description": "Medium complexity with multiple obstacles",
                "start": (0, 0),
                "goal": (15, 15),
                "obstacles": [
                    {"position": (5, 5), "radius": 1.5},
                    {"position": (10, 8), "radius": 1.2},
                    {"position": (7, 12), "radius": 1.0}
                ],
                "world_bounds": {"minX": -5, "maxX": 20, "minZ": -5, "maxZ": 20}
            },
            {
                "name": "complex",
                "description": "Complex maze-like environment",
                "start": (0, 0),
                "goal": (20, 20),
                "obstacles": [
                    {"position": (5, 5), "radius": 2},
                    {"position": (10, 5), "radius": 1.5},
                    {"position": (15, 5), "radius": 2},
                    {"position": (5, 10), "radius": 1.5},
                    {"position": (15, 10), "radius": 1.5},
                    {"position": (5, 15), "radius": 2},
                    {"position": (10, 15), "radius": 1.5},
                    {"position": (15, 15), "radius": 2}
                ],
                "world_bounds": {"minX": -5, "maxX": 25, "minZ": -5, "maxZ": 25}
            },
            {
                "name": "narrow_passages",
                "description": "Environment with narrow passages",
                "start": (0, 5),
                "goal": (20, 5),
                "obstacles": [
                    {"position": (5, 3), "radius": 2.5},
                    {"position": (5, 7), "radius": 2.5},
                    {"position": (15, 3), "radius": 2.5},
                    {"position": (15, 7), "radius": 2.5}
                ],
                "world_bounds": {"minX": -5, "maxX": 25, "minZ": 0, "maxZ": 10}
            }
        ]

    def benchmark_algorithm(
        self,
        algorithm: Dict[str, Any],
        scenario_name: str = "all",
        runs_per_scenario: int = 3
    ) -> List[BenchmarkResult]:
        """
        Benchmark an algorithm on test scenarios

        Args:
            algorithm: Algorithm object with code, type, etc.
            scenario_name: "all" or specific scenario name
            runs_per_scenario: Number of times to run each scenario

        Returns:
            List of BenchmarkResult objects
        """
        results = []

        # Select scenarios
        scenarios = self.scenarios
        if scenario_name != "all":
            scenarios = [s for s in self.scenarios if s["name"] == scenario_name]

        if not scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        algorithm_type = algorithm.get('type', 'unknown')
        algorithm_id = algorithm.get('id', 'unknown')
        algorithm_name = algorithm.get('name', 'Unknown')

        print(f"🔬 Benchmarking {algorithm_name} ({algorithm_type})")
        print(f"   Scenarios: {len(scenarios)} x {runs_per_scenario} runs")

        for scenario in scenarios:
            for run in range(runs_per_scenario):
                result = self._run_single_benchmark(
                    algorithm=algorithm,
                    scenario=scenario,
                    run_number=run + 1
                )
                results.append(result)

        print(f"✅ Completed {len(results)} benchmark runs")
        return results

    def _run_single_benchmark(
        self,
        algorithm: Dict[str, Any],
        scenario: Dict[str, Any],
        run_number: int
    ) -> BenchmarkResult:
        """Run a single benchmark test"""

        algorithm_type = algorithm.get('type', 'unknown')
        algorithm_id = algorithm.get('id', 'unknown')
        algorithm_name = algorithm.get('name', 'Unknown')

        print(f"   Running: {scenario['name']} (run {run_number})")

        # Initialize result
        result = BenchmarkResult(
            algorithm_id=algorithm_id,
            algorithm_type=algorithm_type,
            algorithm_name=algorithm_name,
            execution_time_ms=0.0,
            success_rate=0.0,
            test_scenario=scenario['name']
        )

        try:
            # Run the algorithm based on type
            if algorithm_type == 'path_planning':
                path, exec_time = self._benchmark_path_planning(algorithm, scenario)
                result.execution_time_ms = exec_time

                if path and len(path) >= 2:
                    result.success_rate = 1.0
                    result.goal_reached = True
                    result.path_length = MetricsCalculator.calculate_path_length(path)
                    result.path_smoothness = MetricsCalculator.calculate_path_smoothness(path)

                    # Calculate optimality (compare to straight line distance)
                    optimal_length = self._calculate_straight_line_distance(
                        scenario['start'],
                        scenario['goal']
                    )
                    result.optimality_score = MetricsCalculator.calculate_optimality_score(
                        result.path_length,
                        optimal_length
                    )

                    # Check for collisions
                    result.collision_count = self._count_path_collisions(path, scenario['obstacles'])

                else:
                    result.success_rate = 0.0
                    result.error_message = "No path found"

            elif algorithm_type == 'obstacle_avoidance':
                collisions, exec_time = self._benchmark_obstacle_avoidance(algorithm, scenario)
                result.execution_time_ms = exec_time
                result.collision_count = collisions
                result.success_rate = 1.0 if collisions == 0 else 0.5
                result.goal_reached = collisions == 0

            elif algorithm_type == 'inverse_kinematics':
                success, exec_time = self._benchmark_inverse_kinematics(algorithm, scenario)
                result.execution_time_ms = exec_time
                result.success_rate = 1.0 if success else 0.0
                result.goal_reached = success

            elif algorithm_type == 'computer_vision':
                detections, exec_time = self._benchmark_computer_vision(algorithm, scenario)
                result.execution_time_ms = exec_time
                result.success_rate = min(1.0, detections / max(1, len(scenario.get('obstacles', []))))
                result.goal_reached = detections > 0

            else:
                # Generic benchmark
                exec_time = self._benchmark_generic(algorithm, scenario)
                result.execution_time_ms = exec_time
                result.success_rate = 1.0
                result.goal_reached = True

            # Calculate robustness (consistent performance across runs)
            result.robustness_score = 0.9  # Placeholder - would need multiple runs to calculate

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            result.error_message = str(e)
            result.success_rate = 0.0

        return result

    def _benchmark_path_planning(
        self,
        algorithm: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> Tuple[Optional[List[tuple]], float]:
        """
        Benchmark path planning algorithm
        Returns: (path, execution_time_ms)
        """
        start_time = time.time()

        try:
            # Simulate path planning execution
            # In a real implementation, this would execute the actual algorithm code
            # For now, we'll simulate based on algorithm complexity

            complexity = algorithm.get('complexity', 'Unknown')
            obstacles = scenario['obstacles']

            # Simulate execution time based on complexity
            if 'O(n²)' in complexity or 'O(n^2)' in complexity:
                time.sleep(0.05 + len(obstacles) * 0.01)  # Slower for quadratic
            elif 'O(n log n)' in complexity:
                time.sleep(0.03 + len(obstacles) * 0.005)  # Medium speed
            else:
                time.sleep(0.02 + len(obstacles) * 0.002)  # Fast

            # Generate simulated path (in real impl, this would come from algorithm)
            path = self._generate_simulated_path(
                scenario['start'],
                scenario['goal'],
                obstacles,
                algorithm_name=algorithm.get('name', '')
            )

            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            return path, execution_time

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            print(f"Path planning error: {e}")
            return None, execution_time

    def _benchmark_obstacle_avoidance(
        self,
        algorithm: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> Tuple[int, float]:
        """
        Benchmark obstacle avoidance algorithm
        Returns: (collision_count, execution_time_ms)
        """
        start_time = time.time()

        try:
            # Simulate obstacle avoidance
            obstacles = scenario['obstacles']
            complexity = algorithm.get('complexity', 'Unknown')

            if 'O(n²)' in complexity or 'O(n^2)' in complexity:
                time.sleep(0.03 + len(obstacles) * 0.008)
            else:
                time.sleep(0.02 + len(obstacles) * 0.003)

            # Simulate collision detection
            # Better algorithms have fewer collisions
            if 'dynamic' in algorithm.get('name', '').lower():
                collisions = random.randint(0, 1)
            else:
                collisions = random.randint(0, 2)

            execution_time = (time.time() - start_time) * 1000
            return collisions, execution_time

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return 999, execution_time

    def _benchmark_inverse_kinematics(
        self,
        algorithm: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Benchmark inverse kinematics algorithm"""
        start_time = time.time()

        try:
            complexity = algorithm.get('complexity', 'Unknown')

            if 'iterative' in complexity.lower():
                time.sleep(0.04)
            else:
                time.sleep(0.02)

            # Simulate success rate (analytical methods more accurate)
            success = random.random() > 0.1  # 90% success rate

            execution_time = (time.time() - start_time) * 1000
            return success, execution_time

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return False, execution_time

    def _benchmark_computer_vision(
        self,
        algorithm: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> Tuple[int, float]:
        """Benchmark computer vision algorithm"""
        start_time = time.time()

        try:
            obstacles = scenario.get('obstacles', [])
            time.sleep(0.05)  # CV algorithms typically slower

            # Simulate detection rate
            detections = int(len(obstacles) * 0.85)  # 85% detection rate

            execution_time = (time.time() - start_time) * 1000
            return detections, execution_time

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return 0, execution_time

    def _benchmark_generic(
        self,
        algorithm: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> float:
        """Generic benchmark for unknown algorithm types"""
        start_time = time.time()
        time.sleep(0.03)  # Simulate execution
        return (time.time() - start_time) * 1000

    def _generate_simulated_path(
        self,
        start: tuple,
        goal: tuple,
        obstacles: List[Dict],
        algorithm_name: str
    ) -> List[tuple]:
        """Generate a simulated path for testing"""
        import math

        # Simple path generation for simulation
        # A* would generate more optimal paths than Dijkstra
        if 'A*' in algorithm_name or 'astar' in algorithm_name.lower():
            # More direct path
            steps = 10
        else:
            # Less optimal path
            steps = 15

        path = []
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + (goal[0] - start[0]) * t
            z = start[1] + (goal[1] - start[1]) * t

            # Add some variation to avoid obstacles
            if obstacles:
                for obs in obstacles:
                    obs_pos = obs['position']
                    dist = math.sqrt((x - obs_pos[0])**2 + (z - obs_pos[1])**2)
                    if dist < obs['radius'] + 1:
                        # Deviate slightly
                        x += random.uniform(-0.5, 0.5)
                        z += random.uniform(-0.5, 0.5)

            path.append((x, z))

        return path

    def _calculate_straight_line_distance(self, start: tuple, goal: tuple) -> float:
        """Calculate Euclidean distance between two points"""
        import math
        return math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)

    def _count_path_collisions(
        self,
        path: List[tuple],
        obstacles: List[Dict]
    ) -> int:
        """Count how many times path intersects with obstacles"""
        import math

        collisions = 0
        for point in path:
            for obs in obstacles:
                obs_pos = obs['position']
                obs_radius = obs['radius']
                dist = math.sqrt((point[0] - obs_pos[0])**2 + (point[1] - obs_pos[1])**2)
                if dist < obs_radius:
                    collisions += 1
                    break

        return collisions
