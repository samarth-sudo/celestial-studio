"""
Genesis Documentation Retriever (RAG System)
Retrieves relevant Genesis documentation based on user queries.
"""

import logging
from typing import List, Dict
from pathlib import Path
from .doc_indexer import GenesisDocIndexer, get_indexer

logger = logging.getLogger(__name__)


class GenesisDocRetriever:
    """RAG system for retrieving relevant Genesis documentation"""

    def __init__(self, indexer: GenesisDocIndexer = None):
        """
        Initialize retriever with indexed documentation

        Args:
            indexer: Pre-initialized GenesisDocIndexer (or will create one)
        """
        self.indexer = indexer if indexer is not None else get_indexer()

        # Load API reference
        api_ref_path = Path(__file__).parent / "api_reference.md"
        if api_ref_path.exists():
            with open(api_ref_path, 'r') as f:
                self.api_reference = f.read()
        else:
            logger.warning("API reference file not found")
            self.api_reference = ""

        # Task-specific keywords mapping
        self.task_keywords = {
            'mobile_robot': ['mobile', 'robot', 'navigation', 'velocity', 'wheeled', 'set_velocity'],
            'robotic_arm': ['arm', 'manipulator', 'joint', 'dof', 'control_dofs', 'franka', 'panda'],
            'drone': ['drone', 'quadcopter', 'flight', 'thrust', 'altitude'],
            'pick_place': ['pick', 'place', 'grasp', 'gripper', 'manipulate'],
            'path_planning': ['path', 'planning', 'navigation', 'a*', 'waypoint'],
            'obstacle_avoidance': ['obstacle', 'avoidance', 'collision', 'safe'],
            'vision': ['camera', 'render', 'rgb', 'depth', 'vision'],
            'rl_training': ['reinforcement', 'learning', 'parallel', 'environment', 'batch']
        }

    def retrieve_relevant_docs(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant documentation sections for a query

        Args:
            query: User query or task description
            top_k: Number of relevant documents to return

        Returns:
            List of relevant documentation snippets
        """
        # Detect task type
        task_type = self._detect_task_type(query)

        # Get relevant code examples
        code_examples = []

        # 1. Search by task-specific keywords
        if task_type:
            keywords = self.task_keywords.get(task_type, [])
            for keyword in keywords[:3]:  # Top 3 keywords
                examples = self.indexer.search_code_examples(keyword, top_k=2)
                code_examples.extend(examples)

        # 2. Search by query keywords
        query_keywords = self._extract_keywords(query)
        for keyword in query_keywords[:3]:
            examples = self.indexer.search_code_examples(keyword, top_k=2)
            code_examples.extend(examples)

        # Remove duplicates and limit
        unique_examples = list(dict.fromkeys(code_examples))[:top_k]

        return unique_examples

    def get_full_context(self, query: str) -> str:
        """
        Get complete context including API reference + relevant examples

        Args:
            query: User query or task description

        Returns:
            Combined context string ready for LLM prompt
        """
        # Get relevant docs
        relevant_docs = self.retrieve_relevant_docs(query, top_k=5)

        # Build context
        context_parts = []

        # 1. Always include API reference
        context_parts.append("# GENESIS API REFERENCE\n")
        context_parts.append(self.api_reference)

        # 2. Add relevant examples
        if relevant_docs:
            context_parts.append("\n\n# RELEVANT CODE EXAMPLES\n")
            for i, example in enumerate(relevant_docs, 1):
                context_parts.append(f"\n## Example {i}:\n```python\n{example}\n```\n")

        return "\n".join(context_parts)

    def get_pattern_for_task(self, task_type: str) -> str:
        """
        Get common code pattern for a specific task type

        Args:
            task_type: Type of task (mobile_robot, robotic_arm, etc.)

        Returns:
            Code pattern string or empty string
        """
        patterns = {
            'mobile_robot': '''
import genesis as gs
import numpy as np

gs.init(backend=gs.metal)
scene = gs.Scene(show_viewer=False)

# Add entities
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.Box(size=(0.5, 0.3, 0.2), pos=(0, 0, 0.5))
)

scene.build()

# Control loop
for step in range(1000):
    # Set velocity
    vx = 1.0  # m/s
    omega = 0.1  # rad/s
    robot.set_velocity(linear=(vx, 0, 0), angular=(0, 0, omega))

    # Get state
    pos = robot.get_pos()
    vel = robot.get_vel()

    scene.step()
''',
            'robotic_arm': '''
import genesis as gs
import numpy as np

gs.init(backend=gs.metal)
scene = gs.Scene(show_viewer=False)

# Add arm
plane = scene.add_entity(gs.morphs.Plane())
arm = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
)

scene.build()

# Get DOFs
jnt_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
dofs_idx = [arm.get_joint(name).dof_idx_local for name in jnt_names]

# Set gains
arm.set_dofs_kp(kp=np.array([4500]*7), dofs_idx_local=dofs_idx)
arm.set_dofs_kv(kv=np.array([450]*7), dofs_idx_local=dofs_idx)

# Control loop
target_pos = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
for step in range(1000):
    arm.control_dofs_position(target_pos, dofs_idx)
    scene.step()
''',
            'drone': '''
import genesis as gs
import numpy as np

gs.init(backend=gs.metal)
scene = gs.Scene(show_viewer=False)

# Add drone
plane = scene.add_entity(gs.morphs.Plane())
drone = scene.add_entity(
    gs.morphs.Box(size=(0.3, 0.3, 0.1), pos=(0, 0, 1))
)

scene.build()

# Control loop
for step in range(1000):
    # Simple velocity control
    drone.set_velocity(linear=(0, 0, 0.1), angular=(0, 0, 0))
    scene.step()
'''
        }

        return patterns.get(task_type, "")

    def _detect_task_type(self, query: str) -> str:
        """Detect task type from query"""
        query_lower = query.lower()

        # Check each task type
        for task_type, keywords in self.task_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return task_type

        return ""

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                      'to', 'for', 'with', 'by', 'from', 'as', 'is', 'are',
                      'was', 'were', 'be', 'been', 'being', 'have', 'has',
                      'had', 'do', 'does', 'did', 'will', 'would', 'should',
                      'could', 'may', 'might', 'must', 'can', 'that', 'this'}

        # Split and filter
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        return keywords

    def get_api_for_task(self, query: str) -> str:
        """
        Get recommended APIs for a specific task

        Args:
            query: User query

        Returns:
            String with recommended APIs
        """
        task_type = self._detect_task_type(query)

        api_recommendations = {
            'mobile_robot': "set_velocity(), get_pos(), get_vel()",
            'robotic_arm': "control_dofs_position(), get_dofs_position(), set_dofs_kp()",
            'drone': "set_velocity(), control_dofs_force()",
            'vision': "camera.render(rgb=True), scene.add_camera()",
            'rl_training': "Parallel scenes with batch operations"
        }

        return api_recommendations.get(task_type, "")


# Singleton instance
_retriever_instance = None


def get_retriever() -> GenesisDocRetriever:
    """Get or create singleton retriever instance"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = GenesisDocRetriever()
    return _retriever_instance
