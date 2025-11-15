"""
Algorithm Optimization Module

Provides benchmarking, comparison, and automatic selection of best algorithms
"""

from .metrics import AlgorithmMetrics, BenchmarkResult
from .benchmark import AlgorithmBenchmark
from .comparator import AlgorithmComparator

__all__ = [
    'AlgorithmMetrics',
    'BenchmarkResult',
    'AlgorithmBenchmark',
    'AlgorithmComparator'
]
