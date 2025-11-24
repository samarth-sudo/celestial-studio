"""
Algorithm Generator Module for Celestial Studio - FIXED VERSION

Key improvements:
1. Enforces standardized function names
2. Validates function signatures
3. Returns function metadata
4. Includes interface definitions in prompts
"""

import requests
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from code_validator import validate_code, ValidationResult
except ImportError:
    from backend.code_validator import validate_code, ValidationResult


@dataclass
class AlgorithmRequest:
    """Request for algorithm generation"""
    description: str
    robot_type: str
    algorithm_type: str
    current_code: Optional[str] = None
    modification_request: Optional[str] = None
    use_web_search: bool = False


@dataclass
class AlgorithmResponse:
    """Generated algorithm with metadata"""
    code: str
    parameters: List[Dict[str, any]]
    algorithm_type: str
    description: str
    estimated_complexity: str
    function_name: str  # NEW: Primary function name
    function_signature: Dict[str, any]  # NEW: Function signature info


# Standardized function names by algorithm type (Python signatures for Genesis)
REQUIRED_FUNCTIONS = {
    'path_planning': {
        'name': 'find_path',
        'signature': '(start: np.ndarray, goal: np.ndarray, obstacles: List[Dict], robot_state: Dict) -> List[np.ndarray]'
    },
    'obstacle_avoidance': {
        'name': 'compute_safe_velocity',
        'signature': '(current_pos: np.ndarray, current_vel: np.ndarray, obstacles: List[Dict], goal: np.ndarray, max_speed: float, params: Dict) -> np.ndarray'
    },
    'inverse_kinematics': {
        'name': 'solve_ik',
        'signature': '(target_pos: np.ndarray, current_angles: np.ndarray, link_lengths: np.ndarray, params: Dict) -> np.ndarray'
    },
    'computer_vision': {
        'name': 'process_vision',
        'signature': '(camera_state: Dict, scene_objects: List[Dict], params: Dict) -> List[Dict]'
    }
}


class AlgorithmGenerator:
    """Generates algorithm code using Qwen 2.5 Coder"""

    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "qwen2.5-coder-robotics"

    def __init__(self):
        self.templates = self._load_templates()

    def generate(self, request: AlgorithmRequest) -> AlgorithmResponse:
        """Generate algorithm code from natural language description"""
        try:
            print(f"🔄 Generating {request.algorithm_type} algorithm...")
            print(f"Description: \"{request.description}\"")

            # Get required function info
            required_func = REQUIRED_FUNCTIONS.get(request.algorithm_type)
            if not required_func:
                raise ValueError(f"Unknown algorithm type: {request.algorithm_type}")

            # Build prompt with strict interface requirements
            prompt = self._build_prompt(request, required_func)

            # Generate code using Qwen
            code = self._call_ollama(prompt)

            # Validate that required function exists
            validation = self._validate_function_exists(code, required_func['name'])
            if not validation['valid']:
                print(f"⚠️ Generated code missing required function: {required_func['name']}")
                print(f"🔄 Attempting regeneration with stricter prompt...")
                
                # Try again with even stricter prompt
                strict_prompt = prompt + f"\n\nCRITICAL: You MUST include a function named '{required_func['name']}' with signature: {required_func['signature']}"
                code = self._call_ollama(strict_prompt, temperature=0.1)
                
                # Validate again
                validation = self._validate_function_exists(code, required_func['name'])
                if not validation['valid']:
                    raise Exception(f"Generated code does not contain required function: {required_func['name']}")

            # Extract parameters from generated code
            parameters = self._extract_parameters(code)

            # Estimate complexity
            complexity = self._estimate_complexity(code)

            return AlgorithmResponse(
                code=code,
                parameters=parameters,
                algorithm_type=request.algorithm_type,
                description=request.description,
                estimated_complexity=complexity,
                function_name=required_func['name'],  # NEW
                function_signature=required_func  # NEW
            )

        except Exception as error:
            print(f"❌ Algorithm generation failed: {error}")
            raise

    def _build_prompt(self, request: AlgorithmRequest, required_func: Dict) -> str:
        """Build Qwen prompt with strict interface requirements for Python/Genesis"""

        robot_context = self._get_robot_context(request.robot_type)
        template = self.templates.get(request.algorithm_type, "")

        prompt = f"""You are an expert robotics algorithm engineer. Generate a Python algorithm for {request.algorithm_type} that will run in the Genesis physics engine.

Robot Type: {request.robot_type}
{robot_context}

Task: {request.description}

CRITICAL REQUIREMENTS:
1. You MUST implement a function named: {required_func['name']}
2. Function signature MUST match: {required_func['signature']}
3. Use proper Python type hints (numpy arrays, List, Dict)
4. Include configurable parameters as module-level variables
5. Add clear inline comments
6. Handle edge cases and errors
7. Optimize for real-time performance in Genesis simulation loop
8. Use numpy for all vector/matrix operations
9. The function will be called every simulation step

REQUIRED FUNCTION:
def {required_func['name']}{required_func['signature']}:
    \"\"\"
    {request.description}

    Args:
        [document parameters here]

    Returns:
        [document return value]
    \"\"\"
    # Your implementation here
    pass

REQUIRED IMPORTS:
import numpy as np
from typing import List, Dict, Tuple, Optional

CONFIGURABLE PARAMETERS (at module level):
# Example: SPEED_LIMIT = 2.0  # Maximum speed in m/s
# These will be exposed to the UI for real-time tuning

Template Structure:
{template}

Algorithm Guidelines:
- For path planning: Use A*, RRT, Dijkstra, or PRM based on the task
- For obstacle avoidance: Use DWA, APF, VFH, or potential fields based on the task
- For inverse kinematics: Use FABRIK, CCD, Jacobian, or analytical solutions based on the task
- For computer vision: Choose appropriate CV algorithm for the task

Genesis Integration Notes:
- Robot state includes: position (3D), orientation (quaternion), velocity, joint angles
- Obstacles are dictionaries with 'position' and 'size' keys
- Return values should be numpy arrays for compatibility with Genesis
- Avoid heavy computations; aim for <1ms execution time

Output ONLY the complete Python code. No explanations, no markdown formatting.
The code MUST include the function named '{required_func['name']}'."""

        return prompt

    def _validate_function_exists(self, code: str, function_name: str) -> Dict[str, any]:
        """Validate that required function exists in generated Python code"""

        # Check for Python function declaration
        function_patterns = [
            f"def\\s+{function_name}\\s*\\(",  # def find_path(
            f"async\\s+def\\s+{function_name}\\s*\\(",  # async def find_path(
        ]

        for pattern in function_patterns:
            if re.search(pattern, code):
                return {
                    'valid': True,
                    'function_name': function_name,
                    'pattern_matched': pattern
                }

        return {
            'valid': False,
            'function_name': function_name,
            'error': f"Function '{function_name}' not found in generated code"
        }

    def _call_ollama(self, prompt: str, temperature: float = 0.2) -> str:
        """Call Ollama API to generate code"""
        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 3000,
                        "top_p": 0.9,
                        "top_k": 40,
                        "stop": ["```\n\n", "// Example usage:", "/* Example"]
                    }
                },
                timeout=180
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")

            result = response.json()
            code = result.get("response", "")

            # Clean up the code
            code = self._clean_code(code)

            return code

        except requests.exceptions.Timeout:
            raise Exception("Algorithm generation timed out")
        except Exception as e:
            raise Exception(f"Failed to generate algorithm: {str(e)}")

    def _clean_code(self, code: str) -> str:
        """Clean up generated Python code"""
        # Remove markdown code blocks
        code = re.sub(r'^```python\n', '', code)
        code = re.sub(r'^```py\n', '', code)
        code = re.sub(r'^```\n', '', code)
        code = re.sub(r'\n```$', '', code)

        # Remove example usage sections
        code = re.sub(r'# Example usage:.*$', '', code, flags=re.DOTALL)
        code = re.sub(r'if __name__ == ["\']__main__["\']:.*$', '', code, flags=re.DOTALL)

        return code.strip()

    def _extract_parameters(self, code: str) -> List[Dict[str, any]]:
        """Extract configurable parameters from Python code"""
        parameters = []

        # Match Python module-level constants: NAME = value  # comment
        const_pattern = r'^([A-Z_][A-Z0-9_]*)\s*[:=]\s*([^#\n]+)(?:\s*#\s*(.+))?$'
        matches = re.finditer(const_pattern, code[:1500], re.MULTILINE)

        for match in matches:
            name, value, comment = match.groups()

            value = value.strip()

            # Infer type from value
            param_type = self._infer_python_type(value)

            param = {
                "name": name,
                "type": param_type,
                "value": self._parse_python_value(value, param_type),
                "description": comment.strip() if comment else name.replace('_', ' ').title()
            }

            if param["type"] in ["float", "int"]:
                param["min"] = 0
                param["max"] = max(param["value"] * 3, 10)
                param["step"] = param["value"] * 0.1 if param["type"] == "float" else 1

            parameters.append(param)

        return parameters

    def _parse_value(self, value_str: str, type_hint: str) -> any:
        """Parse string value to proper type (legacy TypeScript support)"""
        try:
            if type_hint.lower() == "number":
                return float(value_str)
            elif type_hint.lower() == "boolean":
                return value_str.lower() == "true"
            elif type_hint.lower() == "string":
                return value_str.strip('"\'')
            else:
                return value_str
        except:
            return value_str

    def _infer_python_type(self, value_str: str) -> str:
        """Infer Python type from value string"""
        value_str = value_str.strip()

        # Check for boolean
        if value_str in ['True', 'False']:
            return 'bool'

        # Check for string
        if value_str.startswith('"') or value_str.startswith("'"):
            return 'str'

        # Check for float
        if '.' in value_str:
            try:
                float(value_str)
                return 'float'
            except:
                pass

        # Check for int
        try:
            int(value_str)
            return 'int'
        except:
            pass

        # Default to string
        return 'str'

    def _parse_python_value(self, value_str: str, param_type: str) -> any:
        """Parse Python value string to proper type"""
        try:
            if param_type == 'int':
                return int(float(value_str))
            elif param_type == 'float':
                return float(value_str)
            elif param_type == 'bool':
                return value_str.strip() == 'True'
            elif param_type == 'str':
                return value_str.strip('"\'')
            else:
                return value_str
        except:
            return value_str

    def _estimate_complexity(self, code: str) -> str:
        """Estimate computational complexity from Python code structure"""
        # Count nested for/while loops
        nested_loops = len(re.findall(r'for\s+\w+\s+in\s+[^:]+:[^f]*for\s+\w+\s+in', code))
        single_loops = len(re.findall(r'(?:for|while)\s+', code)) - nested_loops * 2

        if nested_loops >= 2:
            return "O(n³) - High complexity"
        elif nested_loops >= 1:
            return "O(n²) - Moderate complexity"
        elif single_loops >= 1:
            return "O(n) - Linear complexity"
        else:
            return "O(1) - Constant time"

    def _get_robot_context(self, robot_type: str) -> str:
        """Get context about robot capabilities"""
        contexts = {
            "mobile": """
Robot Capabilities:
- 4-wheel differential drive
- Max speed: 3 m/s
- Can rotate in place
- Ground-based navigation (XZ plane, Y=0.5)
- Collision radius: 0.6m
""",
            "arm": """
Robot Capabilities:
- 6-DOF robotic arm
- 4 revolute joints
- Link lengths: [1.0, 0.8, 0.6, 0.4] meters
- Workspace: ~2.5m radius sphere
- Gripper with 2 fingers
""",
            "drone": """
Robot Capabilities:
- Quadcopter with 4 propellers
- 6-DOF movement (full 3D)
- Max speed: 5 m/s
- Hover altitude: 0.5-5 meters
- Tilt constraints: ±30 degrees
"""
        }
        return contexts.get(robot_type, "")

    def _load_templates(self) -> Dict[str, str]:
        """Load algorithm templates"""
        return {
            "path_planning": """
// Path Planning Algorithm Template
interface PathPoint {
  x: number
  z: number
}

function findPath(
  start: THREE.Vector3,
  goal: THREE.Vector3,
  obstacles: Array<{position: THREE.Vector3, radius: number}>
): THREE.Vector3[] {
  // Algorithm implementation here
  return []
}
""",
            "obstacle_avoidance": """
// Obstacle Avoidance Algorithm Template
interface Vector2D {
  x: number
  z: number
}

function computeSafeVelocity(
  currentPos: Vector2D,
  currentVel: Vector2D,
  obstacles: Array<{position: THREE.Vector3, radius: number}>,
  goal: Vector2D,
  maxSpeed: number
): Vector2D {
  // Algorithm implementation here
  return { x: 0, z: 0 }
}
"""
        }


# Singleton instance
_generator = None

def get_generator() -> AlgorithmGenerator:
    """Get singleton algorithm generator instance"""
    global _generator
    if _generator is None:
        _generator = AlgorithmGenerator()
    return _generator