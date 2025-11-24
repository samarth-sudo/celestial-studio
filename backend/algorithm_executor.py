"""
Algorithm Executor for Genesis Simulation
Handles dynamic loading, execution, and hot-swapping of Python algorithms
"""

import importlib
import importlib.util
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmMetadata:
    """Metadata about a loaded algorithm"""
    algorithm_id: str
    algorithm_type: str
    function_name: str
    code: str
    parameters: Dict[str, Any]
    load_time: float
    execution_count: int = 0
    total_execution_time: float = 0.0
    last_error: Optional[str] = None


class AlgorithmExecutor:
    """
    Executes dynamically loaded Python algorithms
    Supports hot-swapping and parameter updates
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize algorithm executor

        Args:
            cache_dir: Directory to store algorithm modules (temp dir if None)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "genesis_algorithms"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Algorithm storage
        self.algorithms: Dict[str, AlgorithmMetadata] = {}
        self.modules: Dict[str, any] = {}
        self.functions: Dict[str, Callable] = {}

        logger.info(f"Algorithm executor initialized with cache dir: {self.cache_dir}")

    def load_algorithm(
        self,
        algorithm_id: str,
        code: str,
        function_name: str,
        algorithm_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Load algorithm from Python code

        Args:
            algorithm_id: Unique identifier for this algorithm
            code: Python source code
            function_name: Name of the main function to execute
            algorithm_type: Type of algorithm (path_planning, obstacle_avoidance, etc.)
            parameters: Configurable parameters

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading algorithm: {algorithm_id} ({algorithm_type})")

            # Write code to temporary module file
            module_name = f"algorithm_{algorithm_id}".replace('-', '_')
            module_path = self.cache_dir / f"{module_name}.py"

            with open(module_path, 'w') as f:
                f.write(code)

            # Load module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise Exception(f"Failed to create module spec for {module_name}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Get function from module
            if not hasattr(module, function_name):
                raise Exception(f"Function '{function_name}' not found in algorithm code")

            function = getattr(module, function_name)

            if not callable(function):
                raise Exception(f"'{function_name}' is not a callable function")

            # Store algorithm
            self.algorithms[algorithm_id] = AlgorithmMetadata(
                algorithm_id=algorithm_id,
                algorithm_type=algorithm_type,
                function_name=function_name,
                code=code,
                parameters=parameters or {},
                load_time=time.time(),
            )

            self.modules[algorithm_id] = module
            self.functions[algorithm_id] = function

            logger.info(f"✅ Algorithm '{algorithm_id}' loaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load algorithm '{algorithm_id}': {e}")
            if algorithm_id in self.algorithms:
                self.algorithms[algorithm_id].last_error = str(e)
            return False

    def reload_algorithm(self, algorithm_id: str, code: str) -> bool:
        """
        Hot-reload algorithm with new code

        Args:
            algorithm_id: ID of algorithm to reload
            code: New Python source code

        Returns:
            True if reloaded successfully
        """
        if algorithm_id not in self.algorithms:
            logger.error(f"Algorithm '{algorithm_id}' not found")
            return False

        try:
            metadata = self.algorithms[algorithm_id]

            logger.info(f"Reloading algorithm: {algorithm_id}")

            # Update code file
            module_name = f"algorithm_{algorithm_id}".replace('-', '_')
            module_path = self.cache_dir / f"{module_name}.py"

            with open(module_path, 'w') as f:
                f.write(code)

            # Reload module
            if algorithm_id in self.modules:
                importlib.reload(self.modules[algorithm_id])
            else:
                # Module not in cache, load fresh
                return self.load_algorithm(
                    algorithm_id,
                    code,
                    metadata.function_name,
                    metadata.algorithm_type,
                    metadata.parameters
                )

            # Update function reference
            module = self.modules[algorithm_id]
            function = getattr(module, metadata.function_name)
            self.functions[algorithm_id] = function

            # Update metadata
            metadata.code = code
            metadata.load_time = time.time()
            metadata.last_error = None

            logger.info(f"✅ Algorithm '{algorithm_id}' reloaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to reload algorithm '{algorithm_id}': {e}")
            if algorithm_id in self.algorithms:
                self.algorithms[algorithm_id].last_error = str(e)
            return False

    def execute(
        self,
        algorithm_id: str,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Execute algorithm with given arguments

        Args:
            algorithm_id: ID of algorithm to execute
            *args: Positional arguments to pass to algorithm function
            **kwargs: Keyword arguments to pass to algorithm function

        Returns:
            Result from algorithm execution, or None if error
        """
        if algorithm_id not in self.functions:
            logger.error(f"Algorithm '{algorithm_id}' not loaded")
            return None

        metadata = self.algorithms[algorithm_id]

        try:
            start_time = time.time()

            # Execute function
            result = self.functions[algorithm_id](*args, **kwargs)

            # Update statistics
            execution_time = time.time() - start_time
            metadata.execution_count += 1
            metadata.total_execution_time += execution_time
            metadata.last_error = None

            return result

        except Exception as e:
            logger.error(f"❌ Algorithm '{algorithm_id}' execution failed: {e}")
            metadata.last_error = str(e)
            return None

    def update_parameters(self, algorithm_id: str, parameters: Dict[str, Any]) -> bool:
        """
        Update algorithm parameters

        Args:
            algorithm_id: ID of algorithm
            parameters: New parameter values

        Returns:
            True if updated successfully
        """
        if algorithm_id not in self.algorithms:
            logger.error(f"Algorithm '{algorithm_id}' not found")
            return False

        try:
            metadata = self.algorithms[algorithm_id]
            module = self.modules[algorithm_id]

            # Update module-level variables
            for name, value in parameters.items():
                if hasattr(module, name):
                    setattr(module, name, value)
                    metadata.parameters[name] = value
                else:
                    logger.warning(f"Parameter '{name}' not found in algorithm '{algorithm_id}'")

            logger.info(f"Updated parameters for algorithm '{algorithm_id}': {parameters}")
            return True

        except Exception as e:
            logger.error(f"Failed to update parameters for '{algorithm_id}': {e}")
            return False

    def get_parameters(self, algorithm_id: str) -> Optional[Dict[str, Any]]:
        """Get current parameter values for algorithm"""
        if algorithm_id not in self.algorithms:
            return None

        metadata = self.algorithms[algorithm_id]
        module = self.modules[algorithm_id]

        current_values = {}
        for name in metadata.parameters.keys():
            if hasattr(module, name):
                current_values[name] = getattr(module, name)

        return current_values

    def get_statistics(self, algorithm_id: str) -> Optional[Dict[str, Any]]:
        """Get execution statistics for algorithm"""
        if algorithm_id not in self.algorithms:
            return None

        metadata = self.algorithms[algorithm_id]

        avg_execution_time = (
            metadata.total_execution_time / max(1, metadata.execution_count)
        )

        return {
            'algorithm_id': metadata.algorithm_id,
            'algorithm_type': metadata.algorithm_type,
            'function_name': metadata.function_name,
            'execution_count': metadata.execution_count,
            'total_execution_time_ms': metadata.total_execution_time * 1000,
            'avg_execution_time_ms': avg_execution_time * 1000,
            'last_error': metadata.last_error,
            'loaded_at': metadata.load_time,
        }

    def unload_algorithm(self, algorithm_id: str) -> bool:
        """
        Unload algorithm and free resources

        Args:
            algorithm_id: ID of algorithm to unload

        Returns:
            True if unloaded successfully
        """
        if algorithm_id not in self.algorithms:
            return False

        try:
            # Remove from dictionaries
            self.algorithms.pop(algorithm_id, None)
            self.modules.pop(algorithm_id, None)
            self.functions.pop(algorithm_id, None)

            # Remove from sys.modules
            module_name = f"algorithm_{algorithm_id}".replace('-', '_')
            sys.modules.pop(module_name, None)

            # Delete module file
            module_path = self.cache_dir / f"{module_name}.py"
            if module_path.exists():
                module_path.unlink()

            logger.info(f"Unloaded algorithm: {algorithm_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload algorithm '{algorithm_id}': {e}")
            return False

    def list_algorithms(self) -> List[Dict[str, Any]]:
        """List all loaded algorithms"""
        return [
            {
                'algorithm_id': meta.algorithm_id,
                'algorithm_type': meta.algorithm_type,
                'function_name': meta.function_name,
                'parameter_count': len(meta.parameters),
                'execution_count': meta.execution_count,
                'last_error': meta.last_error,
            }
            for meta in self.algorithms.values()
        ]

    def cleanup(self):
        """Clean up all loaded algorithms and cache"""
        logger.info("Cleaning up algorithm executor...")

        # Unload all algorithms
        algorithm_ids = list(self.algorithms.keys())
        for algorithm_id in algorithm_ids:
            self.unload_algorithm(algorithm_id)

        # Clean cache directory
        try:
            for file in self.cache_dir.glob("algorithm_*.py"):
                file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean cache directory: {e}")

        logger.info("✅ Algorithm executor cleaned up")


# Global executor instance
_executor: Optional[AlgorithmExecutor] = None


def get_executor(cache_dir: Optional[str] = None) -> AlgorithmExecutor:
    """Get or create global algorithm executor"""
    global _executor

    if _executor is None:
        _executor = AlgorithmExecutor(cache_dir=cache_dir)

    return _executor


def reset_executor():
    """Reset global algorithm executor"""
    global _executor

    if _executor is not None:
        _executor.cleanup()
        _executor = None
