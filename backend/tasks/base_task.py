"""
Base Task Interface

Abstract base class for robot manipulation tasks, inspired by Isaac Lab's task framework.
Defines the standard interface for rewards, observations, resets, and termination conditions.

Each task implements:
    - get_observations(): Return task-relevant state
    - compute_rewards(): Compute dense rewards for RL
    - is_done(): Check success/failure termination
    - reset(): Reset task to initial state

Usage:
    class MyTask(BaseTask):
        def get_observations(self):
            return {"ee_pos": robot.get_state().ee_pos}

        def compute_rewards(self):
            return -distance_to_target  # Negative distance

        def is_done(self):
            return distance_to_target < 0.01  # 1cm threshold

References:
    - Isaac Lab: isaaclab/envs/mdp/
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

from backend.simulation.pybullet_interface import PyBulletRobotInterface, RobotState


@dataclass
class TaskConfig:
    """
    Base configuration for tasks

    Attributes:
        success_threshold: Distance threshold for success (meters)
        timeout_steps: Maximum steps before timeout
        penalty_collision: Penalty for collisions
        penalty_timeout: Penalty for timeouts
        reward_success: Bonus reward for success
    """
    success_threshold: float = 0.01      # 1cm
    timeout_steps: int = 1000
    penalty_collision: float = -10.0
    penalty_timeout: float = -1.0
    reward_success: float = 10.0


class BaseTask(ABC):
    """
    Abstract base class for robot manipulation tasks

    Provides common functionality:
        - Step counting
        - Success/failure tracking
        - Robot state access
        - Observation/reward interface

    Subclasses must implement:
        - get_observations()
        - compute_rewards()
        - is_done()
        - reset()
    """

    def __init__(
        self,
        robot: PyBulletRobotInterface,
        config: Optional[TaskConfig] = None
    ):
        """
        Initialize base task

        Args:
            robot: PyBulletRobotInterface for robot control
            config: TaskConfig with thresholds and penalties
        """
        self.robot = robot
        self.config = config or TaskConfig()

        # Episode tracking
        self.step_count = 0
        self.episode_count = 0
        self.success_count = 0

        # Task state
        self.is_success = False
        self.is_timeout = False
        self.is_collision = False

    # ========================================================================
    # Abstract Methods (must implement in subclass)
    # ========================================================================

    @abstractmethod
    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get task-relevant observations

        Returns:
            Dictionary of observation arrays (e.g., {"ee_pos": [...], "object_pos": [...]})

        Example:
            >>> obs = task.get_observations()
            >>> print(obs["ee_pos"])  # [0.5, 0.0, 0.3]
        """
        pass

    @abstractmethod
    def compute_rewards(self) -> float:
        """
        Compute reward for current state

        Returns:
            Scalar reward (typically negative distance for dense rewards)

        Example:
            >>> reward = task.compute_rewards()
            >>> print(reward)  # -0.15 (negative distance to goal)
        """
        pass

    @abstractmethod
    def is_done(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if episode should terminate

        Returns:
            (done, info) where:
                - done: True if episode finished
                - info: Dictionary with termination reason
                    - "success": True if task succeeded
                    - "timeout": True if exceeded max steps
                    - "collision": True if collision occurred

        Example:
            >>> done, info = task.is_done()
            >>> if done and info["success"]:
            >>>     print("Task completed successfully!")
        """
        pass

    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset task to initial state

        Should:
            - Reset robot to initial pose
            - Randomize object positions (if applicable)
            - Reset internal counters
            - Return initial observations

        Returns:
            Initial observations dictionary

        Example:
            >>> obs = task.reset()
            >>> print(obs["ee_pos"])
        """
        pass

    # ========================================================================
    # Common Functionality
    # ========================================================================

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one task step (RL-style interface)

        Args:
            action: Optional action to execute (joint commands, IK targets, etc.)

        Returns:
            (observations, reward, done, info) - Standard RL tuple

        Example:
            >>> obs, reward, done, info = task.step(action)
            >>> if done:
            >>>     obs = task.reset()
        """
        self.step_count += 1

        # Execute action (if provided)
        if action is not None:
            self._execute_action(action)

        # Get observations
        obs = self.get_observations()

        # Compute reward
        reward = self.compute_rewards()

        # Check termination
        done, info = self.is_done()

        # Add step count to info
        info["step_count"] = self.step_count

        return obs, reward, done, info

    def _execute_action(self, action: np.ndarray):
        """
        Execute action on robot (override for custom action spaces)

        Default: Treat action as joint position targets

        Args:
            action: Action array
        """
        self.robot.set_joint_targets(action)
        self.robot.step_simulation()

    def get_state(self) -> RobotState:
        """
        Convenience method to get robot state

        Returns:
            Current RobotState
        """
        return self.robot.get_state()

    def check_timeout(self) -> bool:
        """
        Check if episode has timed out

        Returns:
            True if exceeded max steps
        """
        return self.step_count >= self.config.timeout_steps

    def get_success_rate(self) -> float:
        """
        Get success rate over all episodes

        Returns:
            Success rate (0.0 to 1.0)
        """
        if self.episode_count == 0:
            return 0.0
        return self.success_count / self.episode_count

    # ========================================================================
    # Utilities
    # ========================================================================

    def print_status(self):
        """Print current task status (for debugging)"""
        print(f"Task Status:")
        print(f"  Episode: {self.episode_count}")
        print(f"  Step: {self.step_count}")
        print(f"  Success Rate: {self.get_success_rate()*100:.1f}%")
        print(f"  Current State:")
        print(f"    Success: {self.is_success}")
        print(f"    Timeout: {self.is_timeout}")
        print(f"    Collision: {self.is_collision}")


# ============================================================================
# Example Task Implementation
# ============================================================================

class ExampleReachTask(BaseTask):
    """
    Simple example: Reach a fixed target position

    Demonstrates how to implement BaseTask interface.
    """

    def __init__(
        self,
        robot: PyBulletRobotInterface,
        target_pos: np.ndarray,
        config: Optional[TaskConfig] = None
    ):
        super().__init__(robot, config)
        self.target_pos = np.array(target_pos)

    def get_observations(self) -> Dict[str, np.ndarray]:
        state = self.robot.get_state()
        return {
            "ee_pos": state.ee_pos,
            "target_pos": self.target_pos,
            "ee_to_target": self.target_pos - state.ee_pos
        }

    def compute_rewards(self) -> float:
        state = self.robot.get_state()
        distance = np.linalg.norm(state.ee_pos - self.target_pos)

        # Dense reward: negative distance
        reward = -distance

        # Add success bonus
        if distance < self.config.success_threshold:
            reward += self.config.reward_success

        return reward

    def is_done(self) -> Tuple[bool, Dict[str, Any]]:
        state = self.robot.get_state()
        distance = np.linalg.norm(state.ee_pos - self.target_pos)

        # Check success
        if distance < self.config.success_threshold:
            self.is_success = True
            return True, {"success": True, "timeout": False, "collision": False}

        # Check timeout
        if self.check_timeout():
            self.is_timeout = True
            return True, {"success": False, "timeout": True, "collision": False}

        return False, {"success": False, "timeout": False, "collision": False}

    def reset(self) -> Dict[str, np.ndarray]:
        # Reset robot
        self.robot.reset_to_initial_pose()

        # Reset counters
        self.step_count = 0
        self.episode_count += 1
        self.is_success = False
        self.is_timeout = False
        self.is_collision = False

        return self.get_observations()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from backend.assets.robot_configs import FRANKA_PANDA_CONFIG
    from backend.simulation.pybullet_interface import PyBulletRobotInterface

    print("Base Task Example")
    print("=" * 50)

    # Create robot
    robot = PyBulletRobotInterface(FRANKA_PANDA_CONFIG, use_gui=True)

    # Create task
    initial_state = robot.get_state()
    target = initial_state.ee_pos + np.array([0.1, 0.0, 0.0])
    task = ExampleReachTask(robot, target)

    # Run episode
    obs = task.reset()
    print(f"Initial EE Position: {obs['ee_pos']}")
    print(f"Target Position: {obs['target_pos']}")
    print(f"Distance: {np.linalg.norm(obs['ee_to_target'])*1000:.1f}mm")

    done = False
    total_reward = 0.0

    while not done:
        # Use IK to reach target (simple controller)
        from backend.controllers.differential_ik import solve_ik_position
        joint_vel = solve_ik_position(robot, target)

        # Step task
        obs, reward, done, info = task.step()

        total_reward += reward

        # Print progress every 100 steps
        if task.step_count % 100 == 0:
            dist = np.linalg.norm(obs['ee_to_target']) * 1000
            print(f"Step {task.step_count}: Distance = {dist:.1f}mm, Reward = {reward:.3f}")

    print(f"\nEpisode finished:")
    print(f"  Success: {info['success']}")
    print(f"  Total Steps: {info['step_count']}")
    print(f"  Total Reward: {total_reward:.2f}")

    input("\nPress Enter to exit...")
    robot.disconnect()
