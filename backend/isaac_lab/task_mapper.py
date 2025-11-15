"""
Task Mapper

Maps user descriptions and scene configurations to Isaac Lab pre-built tasks.
Provides intelligent task recommendations based on robot type and user requirements.
"""

from typing import Dict, Any, List, Optional, Tuple
import re


class IsaacLabTask:
    """Represents an Isaac Lab pre-built task"""

    def __init__(
        self,
        task_id: str,
        task_name: str,
        description: str,
        robot_type: str,
        task_type: str,
        difficulty: str,
        keywords: List[str],
        num_envs_default: int = 2048,
        max_iterations_default: int = 1000,
        algorithm_default: str = "PPO"
    ):
        self.task_id = task_id
        self.task_name = task_name
        self.description = description
        self.robot_type = robot_type  # 'robotic_arm', 'quadruped', 'humanoid', etc.
        self.task_type = task_type  # 'manipulation', 'navigation', 'locomotion'
        self.difficulty = difficulty  # 'easy', 'medium', 'hard'
        self.keywords = keywords
        self.num_envs_default = num_envs_default
        self.max_iterations_default = max_iterations_default
        self.algorithm_default = algorithm_default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'description': self.description,
            'robot_type': self.robot_type,
            'task_type': self.task_type,
            'difficulty': self.difficulty,
            'keywords': self.keywords,
            'defaults': {
                'num_envs': self.num_envs_default,
                'max_iterations': self.max_iterations_default,
                'algorithm': self.algorithm_default
            }
        }


class TaskMapper:
    """Maps user descriptions to Isaac Lab tasks"""

    # Pre-built Isaac Lab tasks catalog
    TASKS = [
        IsaacLabTask(
            task_id='isaac-reach-franka',
            task_name='Isaac-Reach-Franka-v0',
            description='Franka robot arm reaching task - learn to reach target positions',
            robot_type='robotic_arm',
            task_type='manipulation',
            difficulty='easy',
            keywords=['reach', 'reaching', 'arm', 'franka', 'position', 'target', 'point']
        ),
        IsaacLabTask(
            task_id='isaac-lift-cube-franka',
            task_name='Isaac-Lift-Cube-Franka-v0',
            description='Franka robot cube lifting task - pick and place objects',
            robot_type='robotic_arm',
            task_type='manipulation',
            difficulty='medium',
            keywords=['lift', 'pick', 'place', 'grasp', 'cube', 'box', 'object', 'manipulation']
        ),
        IsaacLabTask(
            task_id='isaac-stack-cube-franka',
            task_name='Isaac-Stack-Cube-Franka-v0',
            description='Franka robot stacking cubes task - advanced manipulation',
            robot_type='robotic_arm',
            task_type='manipulation',
            difficulty='hard',
            keywords=['stack', 'stacking', 'assembly', 'blocks', 'tower', 'advanced']
        ),
        IsaacLabTask(
            task_id='isaac-velocity-flat-anymal',
            task_name='Isaac-Velocity-Flat-Anymal-C-v0',
            description='ANYmal quadruped navigation on flat terrain',
            robot_type='quadruped',
            task_type='navigation',
            difficulty='medium',
            keywords=['walk', 'navigate', 'flat', 'terrain', 'quadruped', 'anymal', 'mobile']
        ),
        IsaacLabTask(
            task_id='isaac-velocity-rough-anymal',
            task_name='Isaac-Velocity-Rough-Anymal-C-v0',
            description='ANYmal quadruped navigation on rough terrain',
            robot_type='quadruped',
            task_type='navigation',
            difficulty='hard',
            keywords=['rough', 'terrain', 'obstacles', 'uneven', 'outdoor', 'challenging']
        ),
        IsaacLabTask(
            task_id='isaac-velocity-humanoid',
            task_name='Isaac-Velocity-Rough-Humanoid-v0',
            description='Humanoid robot locomotion and balance',
            robot_type='humanoid',
            task_type='locomotion',
            difficulty='hard',
            keywords=['humanoid', 'walk', 'balance', 'bipedal', 'human', 'locomotion']
        ),
        IsaacLabTask(
            task_id='isaac-cart-pole',
            task_name='Isaac-Cartpole-v0',
            description='Classic cart-pole balancing task',
            robot_type='cart_pole',
            task_type='control',
            difficulty='easy',
            keywords=['cartpole', 'cart', 'pole', 'balance', 'classic', 'simple', 'control']
        ),
        IsaacLabTask(
            task_id='isaac-shadow-hand',
            task_name='Isaac-Shadow-Hand-Over-v0',
            description='Shadow Hand dexterous manipulation task',
            robot_type='dexterous_hand',
            task_type='manipulation',
            difficulty='hard',
            keywords=['hand', 'finger', 'dexterous', 'shadow', 'complex', 'manipulation']
        )
    ]

    def __init__(self):
        self._task_by_id = {task.task_id: task for task in self.TASKS}
        self._task_by_name = {task.task_name: task for task in self.TASKS}

    def find_task_by_keywords(self, description: str, robot_type: Optional[str] = None) -> Optional[IsaacLabTask]:
        """
        Find best matching task based on description keywords

        Args:
            description: User's task description
            robot_type: Optional robot type filter

        Returns:
            Best matching IsaacLabTask or None
        """
        description_lower = description.lower()

        # Score each task based on keyword matches
        task_scores = []

        for task in self.TASKS:
            # Skip if robot type doesn't match (if specified)
            if robot_type and task.robot_type != robot_type:
                continue

            # Count keyword matches
            score = sum(1 for keyword in task.keywords if keyword in description_lower)

            # Bonus for exact robot type match in description
            if task.robot_type.replace('_', ' ') in description_lower:
                score += 2

            # Bonus for task type match
            if task.task_type in description_lower:
                score += 1

            if score > 0:
                task_scores.append((task, score))

        if not task_scores:
            return None

        # Return task with highest score
        task_scores.sort(key=lambda x: x[1], reverse=True)
        return task_scores[0][0]

    def map_scene_to_task(self, scene_config: Dict[str, Any]) -> Optional[IsaacLabTask]:
        """
        Map scene configuration to Isaac Lab task

        Args:
            scene_config: Celestial scene configuration

        Returns:
            Best matching IsaacLabTask or None
        """
        robot_config = scene_config.get('robot', {})
        robot_type = robot_config.get('type', '')
        task_description = scene_config.get('task', {})

        if isinstance(task_description, dict):
            task_text = task_description.get('type', '') + ' ' + task_description.get('description', '')
        else:
            task_text = str(task_description)

        # Map Celestial robot types to Isaac Lab robot types
        robot_type_mapping = {
            'robotic_arm': 'robotic_arm',
            'mobile_robot': 'quadruped',  # Use quadruped as mobile base
            'drone': 'quadruped',
            'humanoid': 'humanoid',
            'urdf_custom': None
        }

        isaac_robot_type = robot_type_mapping.get(robot_type)

        # Try keyword matching
        task = self.find_task_by_keywords(task_text, isaac_robot_type)

        if task:
            return task

        # Fallback: return default task for robot type
        defaults = {
            'robotic_arm': self._task_by_id.get('isaac-reach-franka'),
            'quadruped': self._task_by_id.get('isaac-velocity-flat-anymal'),
            'humanoid': self._task_by_id.get('isaac-velocity-humanoid'),
        }

        return defaults.get(isaac_robot_type)

    def suggest_tasks(
        self,
        robot_type: Optional[str] = None,
        task_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: int = 5
    ) -> List[IsaacLabTask]:
        """
        Get list of suggested tasks based on filters

        Args:
            robot_type: Filter by robot type
            task_type: Filter by task type (manipulation, navigation, etc.)
            difficulty: Filter by difficulty (easy, medium, hard)
            limit: Maximum number of tasks to return

        Returns:
            List of matching tasks
        """
        filtered_tasks = self.TASKS

        if robot_type:
            filtered_tasks = [t for t in filtered_tasks if t.robot_type == robot_type]

        if task_type:
            filtered_tasks = [t for t in filtered_tasks if t.task_type == task_type]

        if difficulty:
            filtered_tasks = [t for t in filtered_tasks if t.difficulty == difficulty]

        return filtered_tasks[:limit]

    def get_task_by_id(self, task_id: str) -> Optional[IsaacLabTask]:
        """Get task by ID"""
        return self._task_by_id.get(task_id)

    def get_task_by_name(self, task_name: str) -> Optional[IsaacLabTask]:
        """Get task by full name"""
        return self._task_by_name.get(task_name)

    def list_all_tasks(self) -> List[IsaacLabTask]:
        """Get all available tasks"""
        return self.TASKS

    def parse_user_confirmation(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Parse user's confirmation message for training parameters

        Extracts algorithm, duration, iterations, etc. from natural language

        Args:
            message: User's confirmation message

        Returns:
            Dict with extracted parameters or None
        """
        message_lower = message.lower()

        params = {}

        # Detect algorithm
        if 'ppo' in message_lower:
            params['algorithm'] = 'PPO'
        elif 'sac' in message_lower:
            params['algorithm'] = 'SAC'
        elif 'rsl' in message_lower:
            params['algorithm'] = 'RSL'
        else:
            params['algorithm'] = 'PPO'  # default

        # Extract duration/seconds for simulation
        duration_match = re.search(r'(\d+)\s*(?:second|sec|s)', message_lower)
        if duration_match:
            params['duration'] = int(duration_match.group(1))

        # Extract iterations for training
        iter_match = re.search(r'(\d+)\s*(?:iteration|iter|step)', message_lower)
        if iter_match:
            params['max_iterations'] = int(iter_match.group(1))

        # Extract number of environments
        env_match = re.search(r'(\d+)\s*(?:environment|env)', message_lower)
        if env_match:
            params['num_envs'] = int(env_match.group(1))

        # Detect confirmation
        confirmation_keywords = ['yes', 'sure', 'proceed', 'go', 'start', 'run', 'train', 'simulate']
        params['confirmed'] = any(keyword in message_lower for keyword in confirmation_keywords)

        # Detect cancellation
        cancel_keywords = ['no', 'cancel', 'stop', 'abort', 'skip', 'never mind']
        params['cancelled'] = any(keyword in message_lower for keyword in cancel_keywords)

        return params if params['confirmed'] or params['cancelled'] else None

    def get_task_info(self, task: IsaacLabTask) -> str:
        """
        Get formatted information about a task

        Args:
            task: IsaacLabTask object

        Returns:
            Formatted string with task details
        """
        return f"""**{task.task_name}**

{task.description}

- **Type:** {task.task_type.capitalize()}
- **Robot:** {task.robot_type.replace('_', ' ').title()}
- **Difficulty:** {task.difficulty.capitalize()}

**Default Training Config:**
- Algorithm: {task.algorithm_default}
- Environments: {task.num_envs_default}
- Iterations: {task.max_iterations_default}

**Keywords:** {', '.join(task.keywords)}"""


# Singleton instance
_task_mapper: Optional[TaskMapper] = None


def get_task_mapper() -> TaskMapper:
    """Get singleton task mapper instance"""
    global _task_mapper
    if _task_mapper is None:
        _task_mapper = TaskMapper()
    return _task_mapper
