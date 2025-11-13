"""
Algorithm Generator Module for Celestial Studio

Uses Qwen 2.5 Coder to dynamically generate TypeScript algorithm code
from natural language descriptions. Supports real-time modification and
parameter extraction.
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
    description: str  # Natural language description
    robot_type: str  # 'mobile', 'arm', 'drone'
    algorithm_type: str  # 'path_planning', 'obstacle_avoidance', 'inverse_kinematics', 'computer_vision'
    current_code: Optional[str] = None  # For modifications
    modification_request: Optional[str] = None  # "Make it faster", "Add safety margin", etc.


@dataclass
class AlgorithmResponse:
    """Generated algorithm with metadata"""
    code: str  # TypeScript/JavaScript code
    parameters: List[Dict[str, any]]  # Extractable parameters for UI controls
    algorithm_type: str
    description: str
    estimated_complexity: str  # O(n), O(n log n), etc.


class AlgorithmGenerator:
    """Generates algorithm code using Qwen 2.5 Coder"""

    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "qwen2.5-coder:7b"

    def __init__(self):
        """Initialize the algorithm generator"""
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load algorithm templates for reference"""
        return {
            "path_planning": """
// Path Planning Algorithm Template
// Input: start position, goal position, obstacles
// Output: array of waypoints

interface PathPoint {
  x: number
  z: number
}

function findPath(
  start: PathPoint,
  goal: PathPoint,
  obstacles: Array<{position: PathPoint, radius: number}>
): PathPoint[] {
  // Algorithm implementation here
  // Should return array of waypoints from start to goal
  return []
}
""",
            "obstacle_avoidance": """
// Obstacle Avoidance Algorithm Template
// Input: current position, velocity, obstacles, goal
// Output: safe velocity vector

interface Vector2 {
  x: number
  z: number
}

function calculateSafeVelocity(
  currentPos: Vector2,
  currentVel: Vector2,
  obstacles: Array<{position: Vector2, radius: number}>,
  goal: Vector2,
  maxSpeed: number
): Vector2 {
  // Algorithm implementation here
  // Should return velocity vector that avoids obstacles
  return { x: 0, z: 0 }
}
""",
            "inverse_kinematics": """
// Inverse Kinematics Algorithm Template
// Input: target position, current joint angles
// Output: joint angles to reach target

interface JointAngles {
  angles: number[]  // Radians for each joint
}

function solveIK(
  targetPos: {x: number, y: number, z: number},
  currentAngles: number[],
  linkLengths: number[]
): JointAngles {
  // Algorithm implementation here
  // Should return joint angles to reach target position
  return { angles: currentAngles }
}
""",
            "computer_vision": """
// Computer Vision Algorithm Template
// Input: camera state, scene objects
// Output: visual detections/features/tracking data

import * as THREE from 'three'

interface CameraState {
  position: THREE.Vector3
  direction: THREE.Vector3
  up: THREE.Vector3
}

interface Detection {
  label: string
  confidence: number
  bbox: {x: number, y: number, width: number, height: number}
  position3D: THREE.Vector3
  distance: number
}

function processVision(
  camera: CameraState,
  objects: Array<{position: THREE.Vector3, label: string, radius: number}>,
  params: {threshold: number, range: number}
): Detection[] {
  // Algorithm implementation here
  // Should return visual analysis results
  return []
}
"""
        }

    def generate(self, request: AlgorithmRequest) -> AlgorithmResponse:
        """
        Generate algorithm code from natural language description

        Args:
            request: AlgorithmRequest with description and context

        Returns:
            AlgorithmResponse with generated code and metadata
        """
        # Build prompt for Qwen
        prompt = self._build_prompt(request)

        # Generate code using Qwen
        code = self._call_ollama(prompt)

        # Validate generated code for safety
        validation = validate_code(code, request.algorithm_type)

        if not validation.is_valid:
            # Code has critical errors - try regenerating once with stricter prompt
            print(f"⚠️ Generated code has errors: {validation.errors}")
            print(f"🔄 Attempting regeneration with stricter constraints...")

            stricter_prompt = prompt + "\n\nIMPORTANT: Ensure code has no syntax errors and follows best practices."
            code = self._call_ollama(stricter_prompt, temperature=0.1)

            # Validate again
            validation = validate_code(code, request.algorithm_type)

            if not validation.is_valid:
                raise Exception(f"Generated code validation failed: {', '.join(validation.errors)}")

        # Log warnings if any
        if validation.warnings:
            print(f"⚠️ Code warnings: {validation.warnings[:3]}")  # Show first 3

        # Extract parameters from generated code
        parameters = self._extract_parameters(code)

        # Estimate complexity
        complexity = self._estimate_complexity(code)

        return AlgorithmResponse(
            code=code,
            parameters=parameters,
            algorithm_type=request.algorithm_type,
            description=request.description,
            estimated_complexity=complexity
        )

    def modify(self, current_code: str, modification: str, algorithm_type: str) -> AlgorithmResponse:
        """
        Modify existing algorithm code based on natural language request

        Args:
            current_code: Existing TypeScript algorithm code
            modification: Natural language modification request
            algorithm_type: Type of algorithm

        Returns:
            AlgorithmResponse with modified code
        """
        prompt = f"""You are modifying an existing algorithm. Make ONLY the requested changes.

Current Code:
```typescript
{current_code}
```

Modification Request: {modification}

Generate the COMPLETE modified TypeScript code. Include all necessary imports and type definitions.
Output ONLY the code, no explanations."""

        code = self._call_ollama(prompt)

        # Validate modified code
        validation = validate_code(code, algorithm_type)

        if not validation.is_valid:
            raise Exception(f"Modified code validation failed: {', '.join(validation.errors)}")

        if validation.warnings:
            print(f"⚠️ Modified code warnings: {validation.warnings[:2]}")

        parameters = self._extract_parameters(code)
        complexity = self._estimate_complexity(code)

        return AlgorithmResponse(
            code=code,
            parameters=parameters,
            algorithm_type=algorithm_type,
            description=f"Modified: {modification}",
            estimated_complexity=complexity
        )

    def _build_prompt(self, request: AlgorithmRequest) -> str:
        """Build Qwen prompt for algorithm generation"""

        # Get template for this algorithm type
        template = self.templates.get(request.algorithm_type, "")

        # Build context based on robot type
        robot_context = self._get_robot_context(request.robot_type)

        # If modifying existing code
        if request.current_code and request.modification_request:
            return f"""You are modifying an existing {request.algorithm_type} algorithm.

Current Code:
```typescript
{request.current_code}
```

Modification Request: {request.modification_request}

Generate the COMPLETE modified TypeScript code with all improvements.
Output ONLY the code, no explanations."""

        # New algorithm generation
        prompt = f"""You are an expert robotics algorithm engineer. Generate a TypeScript algorithm for {request.algorithm_type}.

Robot Type: {request.robot_type}
{robot_context}

Task: {request.description}

Template Structure:
{template}

Requirements:
1. Generate PRODUCTION-READY TypeScript code
2. Use proper type annotations
3. Include configurable parameters as constants at the top
4. Add clear inline comments explaining the algorithm
5. Handle edge cases and errors
6. Optimize for real-time performance (this runs in browser at 60 FPS)
7. Use efficient data structures (typed arrays when possible)
8. Follow the template structure

Algorithm Guidelines:
- For path planning: Use A*, RRT, or Dijkstra based on the task
- For obstacle avoidance: Use DWA, APF, or VFH based on the task
- For inverse kinematics: Use FABRIK, CCD, or Jacobian based on the task
- For computer vision: Choose appropriate algorithm:
  * Object detection: YOLO-style detection with bounding boxes
  * Object tracking: Kalman filter or tracking-by-detection
  * Feature detection: Corner detection, SIFT-like features
  * Optical flow: Lucas-Kanade or dense flow estimation
  * Semantic segmentation: Region-based or pixel-wise classification

Output ONLY the complete TypeScript code. No explanations, no markdown formatting."""

        return prompt

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
                        "temperature": temperature,  # Lower for more consistent code
                        "num_predict": 3000,  # More tokens for complex algorithms
                        "top_p": 0.9,
                        "top_k": 40,
                        "stop": ["```\n\n", "// Example usage:", "/* Example"]  # Stop before examples
                    }
                },
                timeout=180  # 3 minutes for complex algorithms
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")

            result = response.json()
            code = result.get("response", "")

            # Clean up the code
            code = self._clean_code(code)

            return code

        except requests.exceptions.Timeout:
            raise Exception("Algorithm generation timed out. Try a simpler description.")
        except Exception as e:
            raise Exception(f"Failed to generate algorithm: {str(e)}")

    def _clean_code(self, code: str) -> str:
        """Clean up generated code"""
        # Remove markdown code blocks
        code = re.sub(r'^```typescript\n', '', code)
        code = re.sub(r'^```javascript\n', '', code)
        code = re.sub(r'^```\n', '', code)
        code = re.sub(r'\n```$', '', code)

        # Remove example usage sections
        code = re.sub(r'// Example usage:.*$', '', code, flags=re.DOTALL)
        code = re.sub(r'/\* Example.*?\*/', '', code, flags=re.DOTALL)

        # Trim whitespace
        code = code.strip()

        return code

    def _extract_parameters(self, code: str) -> List[Dict[str, any]]:
        """
        Extract configurable parameters from code

        Looks for constants/config at the top of the code and creates
        UI-friendly parameter definitions
        """
        parameters = []

        # Find const declarations at top of file
        const_pattern = r'const\s+(\w+):\s*(\w+)\s*=\s*([^;]+);?\s*(?://\s*(.+))?'
        matches = re.finditer(const_pattern, code[:1000])  # Check first 1000 chars

        for match in matches:
            name, type_hint, value, comment = match.groups()

            # Skip non-config constants
            if name.startswith('_') or name[0].isupper():
                continue

            param = {
                "name": name,
                "type": type_hint.lower(),
                "value": self._parse_value(value.strip(), type_hint),
                "description": comment.strip() if comment else name.replace('_', ' ').title()
            }

            # Add min/max for numbers
            if param["type"] == "number":
                param["min"] = 0
                param["max"] = param["value"] * 3  # Default range
                param["step"] = param["value"] * 0.1

            parameters.append(param)

        return parameters

    def _parse_value(self, value_str: str, type_hint: str) -> any:
        """Parse string value to proper type"""
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

    def _estimate_complexity(self, code: str) -> str:
        """Estimate computational complexity from code structure"""
        # Simple heuristic based on loop nesting
        nested_loops = len(re.findall(r'for\s*\([^)]+\)[^{]*{[^}]*for\s*\(', code))
        single_loops = len(re.findall(r'for\s*\([^)]+\)', code)) - nested_loops * 2

        if nested_loops >= 2:
            return "O(n³) - High complexity, may impact performance"
        elif nested_loops >= 1:
            return "O(n²) - Moderate complexity"
        elif single_loops >= 1:
            return "O(n) - Linear complexity"
        else:
            return "O(1) - Constant time"


# Singleton instance
_generator = None

def get_generator() -> AlgorithmGenerator:
    """Get singleton algorithm generator instance"""
    global _generator
    if _generator is None:
        _generator = AlgorithmGenerator()
    return _generator
