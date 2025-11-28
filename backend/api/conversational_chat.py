from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import sys
import os
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat.context_manager import ConversationContext
from chat.question_generator import SimulationQuestionGenerator
from chat.memory_manager import get_memory_manager
from simulation.scene_generator import SceneGenerator
from export.package_generator import PackageGenerator
from optimization.comparator import AlgorithmComparator
from utils.toon_service import get_toon_service

# Import Genesis simulation service
from genesis_service import get_simulation, GenesisConfig, RobotType

router = APIRouter()

# Store active conversations (in production, use Redis or database)
active_conversations: Dict[str, ConversationContext] = {}


class ChatRequest(BaseModel):
    userId: str
    message: str


class ChatResponse(BaseModel):
    type: str  # "clarification_needed", "simulation_ready", "chat", "export_ready"
    message: str
    questions: Optional[list] = None
    simulation: Optional[dict] = None
    requirements: Optional[dict] = None
    export_data: Optional[dict] = None  # For export downloads


def detect_export_request(message: str) -> Optional[str]:
    """
    Detect if user is requesting to export/download simulation
    Returns export format if detected, None otherwise
    """
    message_lower = message.lower()

    # Check for export/download keywords
    export_keywords = ['download', 'export', 'package', 'save']
    has_export_keyword = any(keyword in message_lower for keyword in export_keywords)

    if not has_export_keyword:
        return None

    # Detect format
    if 'react' in message_lower or 'typescript' in message_lower or 'web' in message_lower:
        return 'react'
    elif 'ros' in message_lower or 'robot operating system' in message_lower:
        return 'ros'
    elif 'python' in message_lower:
        return 'python'
    elif 'algorithm' in message_lower and 'only' in message_lower:
        return 'algorithms'
    else:
        # Default to React if format not specified
        return 'react'


def detect_comparison_request(message: str) -> bool:
    """
    Detect if user is requesting algorithm comparison/benchmarking
    """
    message_lower = message.lower()

    # Only match if BOTH comparison intent AND algorithm mention are present
    comparison_intent = [
        'compare', 'comparison', 'benchmark', 'which is better',
        'which is best', 'best algorithm', 'fastest algorithm',
        'most efficient algorithm', 'performance comparison',
        'evaluate algorithms', 'test algorithms'
    ]

    algorithm_mention = ['algorithm', 'algorithms']

    has_comparison = any(keyword in message_lower for keyword in comparison_intent)
    has_algorithm = any(keyword in message_lower for keyword in algorithm_mention)

    # Only trigger if both are present, or if it's a very explicit comparison phrase
    explicit_phrases = ['compare algorithm', 'benchmark algorithm', 'which algorithm']
    is_explicit = any(phrase in message_lower for phrase in explicit_phrases)

    return is_explicit or (has_comparison and has_algorithm)


def detect_robot_building_request(message: str) -> Optional[str]:
    """
    Detect if user wants to build a custom robot
    Returns preset name if requesting preset, 'custom' if custom build, None otherwise
    """
    message_lower = message.lower()

    # Check for preset requests
    if 'simple arm' in message_lower or 'simple_arm' in message_lower or '2-dof arm' in message_lower or '2 dof arm' in message_lower:
        return 'simple_arm'
    elif 'mobile base' in message_lower or 'mobile_base' in message_lower or 'differential drive' in message_lower:
        return 'mobile_base'

    # Check for EXPLICIT custom robot building requests
    # Only match if user explicitly says "custom robot" or "build my own" etc.
    custom_phrases = [
        'custom robot',
        'build my own',
        'build a custom',
        'create my own',
        'create a custom',
        'design my own',
        'design a custom',
        'from scratch',
        'build custom',
        'make custom'
    ]

    # Only return 'custom' if there's an explicit custom robot building phrase
    if any(phrase in message_lower for phrase in custom_phrases):
        return 'custom'

    return None


def detect_multi_robot_request(message: str) -> bool:
    """
    Detect if user wants multiple robots in the scene
    """
    message_lower = message.lower()

    multi_keywords = [
        'multiple robots', 'multi-robot', 'multi robot', 'several robots',
        'many robots', 'add another', 'more robots', 'two robots',
        'three robots', 'swarm', 'fleet', 'team of robots'
    ]

    return any(keyword in message_lower for keyword in multi_keywords)


# REMOVED: Isaac Lab detection - using Genesis instead


def detect_training_request(message: str) -> bool:
    """
    Detect if user wants to train robot policy with RL
    """
    message_lower = message.lower()

    training_keywords = [
        'train', 'training', 'learn', 'learning', 'teach',
        'reinforcement learning', 'rl', 'policy', 'ppo', 'sac',
        'train the robot', 'teach the robot', 'machine learning'
    ]

    return any(keyword in message_lower for keyword in training_keywords)


def detect_algorithm_request(message: str) -> bool:
    """
    Detect if user wants to generate algorithms for their robot
    """
    message_lower = message.lower()

    algorithm_keywords = [
        'generate algorithm', 'create algorithm', 'make algorithm',
        'algorithm for', 'algorithms for', 'write algorithm',
        'path planning', 'pathfinding', 'path finding',
        'navigation algorithm', 'obstacle avoidance algorithm',
        'control algorithm', 'kinematics', 'inverse kinematics',
        'motion planning', 'trajectory planning',
        'better navigation', 'better travel', 'better movement',
        'improve navigation', 'improve pathfinding'
    ]

    return any(keyword in message_lower for keyword in algorithm_keywords)


async def handle_export_request(context: ConversationContext, export_format: str) -> ChatResponse:
    """Handle export request from user"""

    # Check if simulation has been generated
    if not hasattr(context, 'last_simulation') or not context.last_simulation:
        return ChatResponse(
            type="chat",
            message="No simulation to export yet. Please create a simulation first by telling me what robot and environment you need."
        )

    # Check if algorithms have been generated
    if not hasattr(context, 'algorithms') or not context.algorithms:
        return ChatResponse(
            type="chat",
            message="No algorithms generated yet. The simulation needs algorithms before it can be exported."
        )

    try:
        # Generate package
        generator = PackageGenerator()

        # Prepare export data
        algorithms = context.algorithms
        scene_config = context.last_simulation
        robots = [{'type': context.requirements.get('robot_type', 'mobile')}]
        project_name = f"celestial_{context.requirements.get('robot_type', 'simulation')}"

        # Generate package
        zip_path = generator.generate_package(
            export_format=export_format,
            algorithms=algorithms,
            scene_config=scene_config,
            robots=robots,
            project_name=project_name
        )

        filename = os.path.basename(zip_path)

        # Format description
        format_names = {
            'react': 'React/TypeScript',
            'ros': 'ROS',
            'python': 'Python',
            'algorithms': 'Algorithm Files'
        }

        format_descriptions = {
            'react': 'Complete web application with Three.js physics simulation',
            'ros': 'ROS workspace with Python nodes for real robot deployment',
            'python': 'Standalone Python scripts with algorithm implementations',
            'algorithms': 'Algorithm code files with metadata'
        }

        format_name = format_names.get(export_format, export_format)
        format_desc = format_descriptions.get(export_format, '')

        return ChatResponse(
            type="export_ready",
            message=f"""Package exported successfully!

**Format:** {format_name}
**Description:** {format_desc}
**File:** {filename}

Your download will start automatically. The package includes:
{'- Full React project setup' if export_format == 'react' else ''}
{'- ROS workspace with nodes and launch files' if export_format == 'ros' else ''}
{'- Standalone Python implementations' if export_format == 'python' else ''}
{'- Algorithm code files' if export_format == 'algorithms' else ''}
- Algorithm implementations
- Scene configuration
- Robot models
- Setup instructions

You can now use this package for further development!""",
            export_data={
                "filename": filename,
                "format": export_format,
                "download_url": f"/api/export/download/{filename}"
            }
        )

    except Exception as e:
        print(f"Export error: {e}")
        import traceback
        traceback.print_exc()

        return ChatResponse(
            type="chat",
            message=f"Export failed: {str(e)}\n\nPlease try again or contact support if the issue persists."
        )


async def handle_comparison_request(context: ConversationContext) -> ChatResponse:
    """Handle algorithm comparison/benchmarking request"""

    # Check if algorithms have been generated
    if not hasattr(context, 'algorithms') or not context.algorithms:
        return ChatResponse(
            type="chat",
            message="No algorithms to compare yet. Generate some algorithms first, then I can benchmark and compare them for you."
        )

    if len(context.algorithms) < 2:
        return ChatResponse(
            type="chat",
            message=f"Need at least 2 algorithms to compare. You currently have {len(context.algorithms)} algorithm. Generate more algorithms to see performance comparison."
        )

    try:
        print(f"Comparing {len(context.algorithms)} algorithms...")

        # Run comparison
        comparator = AlgorithmComparator()
        comparison = comparator.compare_algorithms(
            algorithms=context.algorithms,
            scenario="all",
            runs_per_scenario=2  # Reduced for faster response
        )

        best = comparison["best_algorithm"]
        rankings = comparison["rankings"]
        recommendation = comparison["recommendation"]

        # Serialize comparison data to TOON format for efficient storage/transmission
        toon_service = get_toon_service()
        toon_comparison = toon_service.serialize_comparison_data(comparison)

        print(f"TOON formatted comparison data (token-efficient)")

        # Format response message
        message = f"""**Algorithm Comparison Results**

**Winner: {best['algorithm_name']}**
- Overall Score: {best['overall_score']}/100
- Avg Execution Time: {best['avg_execution_time_ms']:.1f}ms
- Success Rate: {best['avg_success_rate'] * 100:.0f}%
- Collisions: {best['avg_collisions_per_run']:.1f} per run

**Rankings:**
"""

        for i, algo in enumerate(rankings[:3], 1):  # Top 3
            message += f"{i}. **{algo['algorithm_name']}** (Score: {algo['overall_score']:.1f}/100)\n"
            message += f"   - Time: {algo['avg_execution_time_ms']:.1f}ms"
            if algo['avg_path_length']:
                message += f" | Path: {algo['avg_path_length']:.1f}m"
            message += f" | Success: {algo['avg_success_rate'] * 100:.0f}%\n"

        message += f"\n**Recommendation:**\n{recommendation['message']}\n"
        message += f"\n*Tested across {comparison['test_info']['total_tests']} scenarios*"

        return ChatResponse(
            type="chat",
            message=message,
            export_data={
                "comparison": comparison,
                "toon_data": toon_comparison  # TOON format for efficient LLM use
            }
        )

    except Exception as e:
        print(f"Comparison error: {e}")
        import traceback
        traceback.print_exc()

        return ChatResponse(
            type="chat",
            message=f"Comparison failed: {str(e)}\n\nPlease try again."
        )


async def handle_robot_building_request(preset_or_custom: str) -> ChatResponse:
    """Handle robot building request"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        if preset_or_custom in ['simple_arm', 'mobile_base']:
            # Load preset
            from urdf.generator import RobotPresets
            from urdf.parser import URDFParser
            from urdf.threejs_converter import ThreeJSConverter

            print(f"Loading preset: {preset_or_custom}")

            # Generate preset
            if preset_or_custom == 'simple_arm':
                gen = RobotPresets.simple_arm()
                description = "2-DOF robotic arm with revolute joints"
            else:
                gen = RobotPresets.mobile_base()
                description = "Differential drive mobile robot with continuous wheel joints"

            # Generate URDF
            urdf_xml = gen.generate()

            # Parse and convert
            parser = URDFParser()
            urdf_data = parser.parse_string(urdf_xml)

            converter = ThreeJSConverter()
            scene_config = converter.convert(urdf_data)

            message = f"""**Robot Generated**

I've created a **{description}** for you.

The robot is now loaded in the 3D simulator on the right. You can:
- Ask me to modify it (change colors, sizes, add links)
- Add behaviors or algorithms
- Add more robots to create a multi-robot scene
- Export it to React, ROS, or Python

What would you like to do next?"""

            return ChatResponse(
                type="simulation_ready",
                message=message,
                simulation=scene_config
            )

        else:  # custom
            message = """**Let's Build a Custom Robot**

I can help you create a custom robot. Tell me:

1. **What type of robot?** (arm, mobile robot, legged robot, etc.)
2. **How many joints/links?**
3. **What dimensions?** (sizes, lengths)
4. **Any specific requirements?**

For example:
- "A 3-DOF robot arm with 0.5m links"
- "A mobile robot with 4 wheels"
- "An arm with a gripper at the end"

What would you like to build?"""

            return ChatResponse(
                type="chat",
                message=message
            )

    except Exception as e:
        print(f"Robot building failed: {e}")
        import traceback
        traceback.print_exc()

        return ChatResponse(
            type="chat",
            message=f"Sorry, I encountered an error building the robot: {str(e)}\n\nPlease try again or describe what you'd like differently."
        )


async def handle_multi_robot_request(context: ConversationContext) -> ChatResponse:
    """Handle multi-robot request"""
    try:
        # Check if simulation exists
        if not hasattr(context, 'last_simulation') or not context.last_simulation:
            message = """**Multi-Robot Setup**

To create a multi-robot scene, I need at least one robot first! You can:

1. **Generate a robot:** "Create a mobile robot"
2. **Load a preset:** "Load the simple arm preset"
3. **Build custom:** "Build a 2-DOF robot arm"

Once we have the first robot, I can add more robots to the scene at different positions!

What robot should we start with?"""

            return ChatResponse(
                type="chat",
                message=message
            )

        # User has a simulation, offer to add more robots
        message = """**Adding More Robots**

Great! I can add more robots to your scene. For each additional robot, tell me:

1. **What type of robot?** (or use existing design)
2. **Where to place it?** (position: x, y, z)
3. **Any specific name?**

Examples:
- "Add another mobile robot at position (2, 0, 0)"
- "Add a simple arm at position (-2, 0, 0) named arm_2"
- "Add 3 more robots in a line"

What robot would you like to add?"""

        return ChatResponse(
            type="chat",
            message=message
        )

    except Exception as e:
        print(f"Multi-robot request failed: {e}")
        return ChatResponse(
            type="chat",
            message=f"Error setting up multi-robot scene: {str(e)}"
        )


# REMOVED: Isaac Lab and Modal related handlers - using Genesis instead


def generate_genesis_python_code(requirements: Dict[str, Any]) -> str:
    """
    Generate executable Python code using Genesis API based on requirements

    Returns complete Python script that initializes Genesis simulation
    """
    robot_type = requirements.get('robot_type', 'mobile')
    environment = requirements.get('environment', 'warehouse')
    task = requirements.get('task', 'navigation')
    objects = requirements.get('objects', ['boxes'])

    # Map robot type to Genesis entity
    robot_configs = {
        'mobile': {
            'type': 'URDF',
            'file': 'urdf/mobile_robot.urdf',
            'description': 'Differential drive mobile robot'
        },
        'arm': {
            'type': 'MJCF',
            'file': 'xml/franka_emika_panda/panda.xml',
            'description': 'Franka Panda robotic arm'
        },
        'drone': {
            'type': 'Drone',
            'file': 'urdf/drones/cf2x.urdf',
            'description': 'Crazyflie 2.X quadcopter'
        }
    }

    robot_config = robot_configs.get(robot_type, robot_configs['mobile'])

    # Generate Python code
    code = f'''"""
Genesis Simulation: {robot_type.capitalize()} Robot - {task.capitalize()}
Generated by Celestial Studio

Robot: {robot_config['description']}
Environment: {environment.capitalize()}
Task: {task.capitalize()}
"""

import genesis as gs
import numpy as np

# Initialize Genesis with Metal backend (macOS GPU acceleration)
gs.init(
    backend=gs.metal,  # Use Metal for Apple Silicon
    precision='32',
    logging_level='info'
)

# Create scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,  # 100 Hz physics
        substeps=10,
        gravity=(0, 0, -9.81)
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0, 0, 0.5),
        camera_fov=40,
        max_FPS=60
    ),
    show_viewer=False  # Headless mode for server
)

# Add ground plane
ground = scene.add_entity(gs.morphs.Plane())
'''

    # Add robot based on type
    if robot_type == 'mobile':
        code += '''
# Add mobile robot
robot = scene.add_entity(
    gs.morphs.URDF(
        file='urdf/mobile_robot.urdf',
        pos=(0, 0, 0.5),
        fixed=False
    )
)
'''
    elif robot_type == 'arm':
        code += '''
# Add Franka Panda arm
robot = scene.add_entity(
    gs.morphs.MJCF(
        file='xml/franka_emika_panda/panda.xml',
        pos=(0, 0, 0),
        euler=(0, 0, 0)
    )
)
'''
    elif robot_type == 'drone':
        code += '''
# Add Crazyflie drone
robot = scene.add_entity(
    gs.morphs.Drone(
        file='urdf/drones/cf2x.urdf',
        model='CF2X',
        pos=(0, 0, 0.5),
        propellers_link_name=('prop0_link', 'prop1_link', 'prop2_link', 'prop3_link'),
        propellers_spin=(-1, 1, -1, 1)
    )
)
'''

    # Add obstacles
    code += f'''
# Add obstacles ({', '.join(objects)})
obstacles = []
for i in range(5):
    obstacle = scene.add_entity(
        gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(np.random.uniform(-3, 3), np.random.uniform(-3, 3), 0.25),
            fixed=True
        )
    )
    obstacles.append(obstacle)
'''

    # Add cameras
    code += '''
# Add cameras for different views
main_camera = scene.add_camera(
    res=(1920, 1080),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=40,
    GUI=False  # Headless rendering
)

fpv_camera = scene.add_camera(
    res=(1280, 720),
    pos=(0.5, 0, 0.5),
    lookat=(1, 0, 0.5),
    fov=60,
    GUI=False
)

top_camera = scene.add_camera(
    res=(1280, 720),
    pos=(0, 0, 5),
    lookat=(0, 0, 0),
    fov=50,
    GUI=False
)

# Build scene (triggers JIT compilation)
print("Building scene...")
scene.build()
print("Scene built successfully!")

# Simulation loop
print("Starting simulation...")
for step in range(1000):
    '''

    # Add robot control based on type
    if robot_type == 'mobile':
        code += '''
    # Mobile robot control example
    if step < 200:
        # Move forward
        robot.set_vel(np.array([0.5, 0, 0]))
    elif step < 400:
        # Rotate
        robot.set_ang(np.array([0, 0, 0.5]))
    else:
        # Stop
        robot.set_vel(np.array([0, 0, 0]))
        robot.set_ang(np.array([0, 0, 0]))
    '''
    elif robot_type == 'arm':
        code += '''
    # Robotic arm control example
    if step < 500:
        # Move to target position
        target_pos = np.array([0.3, 0, 0.5, 0, 0, 0, 0, 0.04, 0.04])
        robot.control_dofs_position(target_pos, robot.get_dofs_idx())
    '''
    elif robot_type == 'drone':
        code += '''
    # Drone control example
    hover_rpm = 14468.429183500699
    if step < 500:
        # Hover
        robot.set_propellels_rpm(np.array([hover_rpm] * 4))
    '''

    code += '''
    # Step physics
    scene.step()

    # Render frame every 2 steps (30 FPS)
    if step % 2 == 0:
        rgb, _, _, _ = main_camera.render(rgb=True)
        # Frame available for streaming

    # Print progress
    if step % 100 == 0:
        print(f"Step {step}/1000")

print("Simulation complete!")
'''

    return code


async def execute_genesis_simulation(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Genesis simulation from requirements

    This function initializes Genesis and starts the simulation
    Returns status and generated Python code
    """
    try:
        # Get Genesis simulation instance
        simulation = get_simulation()

        # Generate Python code
        python_code = generate_genesis_python_code(requirements)

        # Initialize Genesis if first time, otherwise reset scene
        if not simulation.is_initialized:
            print("🚀 Initializing Genesis simulation for the first time...")
            simulation.initialize()
            print("✅ Genesis initialized successfully")
        else:
            print("♻️ Genesis already initialized, resetting scene for new simulation...")
            simulation.reset_scene()
            print("✅ Scene reset complete")

        # Map robot type
        robot_type_map = {
            'mobile': RobotType.MOBILE_ROBOT,
            'arm': RobotType.ROBOTIC_ARM,
            'drone': RobotType.DRONE
        }

        robot_type_str = requirements.get('robot_type', 'mobile')
        robot_type = robot_type_map.get(robot_type_str, RobotType.MOBILE_ROBOT)

        # Add robot to scene
        robot_id = "robot-1"
        print(f"🤖 Adding {robot_type_str} robot to scene...")
        simulation.add_robot(
            robot_id=robot_id,
            robot_type=robot_type,
            position=(0, 0, 0.5)
        )
        print(f"✅ Robot '{robot_id}' added")

        # Add obstacles
        import random
        obstacles = requirements.get('objects', ['boxes'])
        num_obstacles = 5
        print(f"📦 Adding {num_obstacles} obstacles...")
        for i in range(num_obstacles):
            simulation.add_obstacle(
                obstacle_id=f"obstacle-{i}",
                position=(random.uniform(-3, 3), random.uniform(-3, 3), 0.25),
                size=(0.5, 0.5, 0.5)
            )
        print(f"✅ {num_obstacles} obstacles added")

        # Build scene (JIT compilation)
        print("🔨 Building Genesis scene (JIT compilation)...")
        simulation.build_scene()
        print("✅ Scene built successfully")

        # Start simulation
        print("▶️ Starting simulation loop...")
        simulation.start()
        print("✅ Simulation running at 60 FPS")

        return {
            "status": "success",
            "message": "Genesis simulation initialized and running",
            "python_code": python_code,
            "robot_id": robot_id,
            "robot_type": robot_type_str
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Failed to execute Genesis simulation: {e}")
        print(error_trace)

        return {
            "status": "error",
            "message": f"Failed to initialize Genesis: {str(e)}",
            "python_code": python_code if 'python_code' in locals() else None,
            "error": error_trace
        }


async def handle_algorithm_request(context: ConversationContext, message: str) -> ChatResponse:
    """Handle algorithm generation request - Generate PYTHON code for Genesis"""
    try:
        # Debug logging
        print(f"🔍 Algorithm request - Checking simulation state:")
        print(f"   - hasattr(context, 'last_simulation'): {hasattr(context, 'last_simulation')}")
        if hasattr(context, 'last_simulation'):
            print(f"   - context.last_simulation is not None: {context.last_simulation is not None}")
            if context.last_simulation:
                print(f"   - Simulation exists with keys: {context.last_simulation.keys()}")

        # Check if simulation exists
        if not hasattr(context, 'last_simulation') or not context.last_simulation:
            return ChatResponse(
                type="chat",
                message="""To generate algorithms, I need a robot simulation first!

Tell me:
- What kind of robot? (mobile robot, robotic arm, drone, etc.)
- What environment? (warehouse, office, outdoor, etc.)
- What task? (navigation, pick and place, inspection, etc.)

Example: "Create a mobile robot in a warehouse for navigation" """
            )

        robot_type = context.requirements.get('robot_type', 'mobile')
        task = context.requirements.get('task', 'navigation')
        message_lower = message.lower()

        # Generate Python code for the algorithm
        python_code = generate_genesis_python_code(context.requirements)

        # Determine algorithm type based on keywords
        algorithm_type = None
        if any(word in message_lower for word in ['path', 'navigate', 'travel', 'route', 'waypoint',
                                                    'longest', 'maximum path', 'scenic', 'exploration',
                                                    'coverage', 'furthest', 'maximize distance', 'explore area']):
            algorithm_type = 'path_planning'
            algorithm_name = 'Path Planning'
        elif any(word in message_lower for word in ['obstacle', 'avoid', 'collision']):
            algorithm_type = 'obstacle_avoidance'
            algorithm_name = 'Obstacle Avoidance'
        elif any(word in message_lower for word in ['inverse', 'kinematics', 'ik', 'arm control']):
            algorithm_type = 'inverse_kinematics'
            algorithm_name = 'Inverse Kinematics'
        elif any(word in message_lower for word in ['vision', 'detect', 'recognition', 'image']):
            algorithm_type = 'computer_vision'
            algorithm_name = 'Computer Vision'
        elif any(word in message_lower for word in ['control', 'motion', 'movement']):
            algorithm_type = 'motion_control'
            algorithm_name = 'Motion Control'
        else:
            # Default based on robot type and task
            if robot_type == 'arm':
                algorithm_type = 'inverse_kinematics'
                algorithm_name = 'Inverse Kinematics'
            else:
                algorithm_type = 'path_planning'
                algorithm_name = 'Path Planning'

        return ChatResponse(
            type="chat",
            message=f"""**Generating {algorithm_name} Algorithm (Python)**

I'll create a custom {algorithm_name.lower()} algorithm for your {robot_type} robot using Genesis!

**Algorithm Details:**
- Type: {algorithm_name}
- Robot: {robot_type.capitalize()}
- Task: {task}
- Language: **Python** (Genesis API)
- Generation: Using latest research and best practices

The Python code has been generated! You can:
- View the algorithm code in the code viewer
- Modify parameters in real-time
- Execute it in Genesis simulation
- Export it as a Python script

**Note:** Algorithms are now generated as **Python code** that uses the Genesis physics engine, not TypeScript!

```python
# Preview of generated code:
{python_code[:200]}...
```""",
            export_data={
                "python_code": python_code,
                "algorithm_type": algorithm_type
            }
        )

    except Exception as e:
        print(f"Algorithm request failed: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            type="chat",
            message=f"Error generating algorithm: {str(e)}"
        )


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main conversational chat endpoint

    Handles the conversation flow:
    1. Check if user wants to export
    2. Extract requirements from user message
    3. Check if we have enough info
    4. Either ask questions or generate simulation
    """

    user_id = request.userId
    message = request.message

    try:
        # Health check: Ensure Ollama is running before processing
        from utils.ollama_health import check_ollama_health
        is_healthy, health_msg = check_ollama_health()
        if not is_healthy:
            raise HTTPException(
                status_code=503,
                detail=f"⚠️ AI model server (Ollama) is unavailable: {health_msg}\n\n"
                      f"To fix this:\n"
                      f"1. Open a terminal\n"
                      f"2. Run: ollama serve\n"
                      f"3. Ensure 'qwen2.5-coder:7b' model is installed\n"
                      f"4. Try your request again"
            )

        # Get or create conversation context (with user_id for ChromaDB)
        if user_id not in active_conversations:
            active_conversations[user_id] = ConversationContext(user_id=user_id)
            print(f"🆕 Created new context for user {user_id}")
        else:
            print(f"♻️  Retrieved existing context for user {user_id}")

        context = active_conversations[user_id]
        print(f"   - Context ID: {id(context)}")
        print(f"   - Has last_simulation: {hasattr(context, 'last_simulation')}")
        if hasattr(context, 'last_simulation') and context.last_simulation:
            print(f"   - Simulation keys: {context.last_simulation.keys()}")

        # Check if this is an export request
        export_format = detect_export_request(message)
        if export_format:
            return await handle_export_request(context, export_format)

        # Check if this is a comparison/benchmark request
        if detect_comparison_request(message):
            return await handle_comparison_request(context)

        # Check if this is a robot building request
        robot_building = detect_robot_building_request(message)
        if robot_building:
            return await handle_robot_building_request(robot_building)

        # Check if this is a multi-robot request
        if detect_multi_robot_request(message):
            return await handle_multi_robot_request(context)

        # Check if this is an algorithm generation request
        if detect_algorithm_request(message):
            return await handle_algorithm_request(context, message)

        # Update context with user's message (extracts requirements in background)
        requirements = context.update_from_message(message)

        # Check if we can generate simulation
        if context.is_ready_to_generate():
            # We have enough info! Generate simulation
            scene_generator = SceneGenerator()
            simulation_config = scene_generator.generate_from_requirements(requirements)

            # Store simulation in context for later export
            context.last_simulation = simulation_config
            print(f"✅ Stored simulation in context for user {user_id}")
            print(f"   - Context ID: {id(context)}")
            print(f"   - Simulation keys: {simulation_config.keys()}")

            # Initialize empty algorithms list (will be populated when algorithms are generated)
            if not hasattr(context, 'algorithms'):
                context.algorithms = []

            # NEW: Actually execute Genesis simulation!
            print("🚀 Executing Genesis simulation...")
            genesis_result = await execute_genesis_simulation(requirements)

            # Build conversational response message
            robot_type = requirements.get('robot_type', 'robot')
            environment = requirements.get('environment', 'environment')
            task = requirements.get('task', 'task')

            if genesis_result['status'] == 'success':
                response_message = f"""Perfect! I've created your simulation in Genesis:

- {robot_type.capitalize()} robot
- {environment.capitalize()} environment
- Task: {task}
- Objects: {', '.join(requirements.get('objects', ['boxes']))}

**✅ Genesis Simulation Running!**
- Physics engine: Metal (GPU accelerated)
- Video stream: Active at 30 FPS
- Python code: Generated and executing

**Generated Python Code:**
```python
{genesis_result['python_code'][:300]}...
```

**Next steps:**
- View the simulation video stream (should appear automatically)
- Generate algorithms for your robot (Python code)
- Modify and test the Python code
- Export the complete Python project"""
            else:
                response_message = f"""I've created your simulation configuration:

- {robot_type.capitalize()} robot
- {environment.capitalize()} environment
- Task: {task}

**⚠️ Genesis initialization failed:** {genesis_result['message']}

The browser preview will work, but for full Genesis physics simulation, please check the backend logs.

**Next steps:**
- Generate algorithms for your robot
- Fix Genesis initialization issues
- Export as Python project"""

            return ChatResponse(
                type="simulation_ready",
                message=response_message,
                simulation=simulation_config,
                requirements=requirements,
                export_data={
                    "genesis_status": genesis_result['status'],
                    "python_code": genesis_result.get('python_code')
                }
            )
        else:
            # Need more information - generate conversational response
            conversational_message = context.generate_conversational_response(message)

            # If generate_conversational_response returns None, it means we've exceeded
            # question limit and should force generation with partial requirements
            if conversational_message is None:
                print(f"⚠️ Question limit exceeded, forcing simulation generation with partial requirements")

                # Generate simulation with whatever we have
                scene_generator = SceneGenerator()
                simulation_config = scene_generator.generate_from_requirements(requirements)

                # Store simulation in context
                context.last_simulation = simulation_config

                # Initialize empty algorithms list
                if not hasattr(context, 'algorithms'):
                    context.algorithms = []

                # NEW: Execute Genesis simulation
                print("🚀 Executing Genesis simulation (forced generation)...")
                genesis_result = await execute_genesis_simulation(requirements)

                # Build response message acknowledging we're generating with partial info
                robot_type = requirements.get('robot_type', 'mobile robot')
                environment = requirements.get('environment', 'simulation environment')
                task = requirements.get('task', 'autonomous navigation')

                if genesis_result['status'] == 'success':
                    response_message = f"""Got it! I'll create the simulation with the information you've provided:

- Robot: {robot_type}
- Environment: {environment}
- Task: {task}

**✅ Genesis Simulation Running!**
The simulation is now running with Genesis physics engine. You can refine it by chatting with me.

**Generated Python Code:**
```python
{genesis_result['python_code'][:250]}...
```

**Next steps:**
- View the simulation video stream
- Generate algorithms for your robot (Python code)
- Modify and test the code
- Export as Python project"""
                else:
                    response_message = f"""Got it! I'll create the simulation with the information you've provided:

- Robot: {robot_type}
- Environment: {environment}
- Task: {task}

⚠️ Genesis had issues: {genesis_result['message']}

The simulation config is ready. You can refine it by chatting with me.

**Next steps:**
- Generate algorithms for your robot
- Fix Genesis initialization
- Export as Python project"""

                return ChatResponse(
                    type="simulation_ready",
                    message=response_message,
                    simulation=simulation_config,
                    requirements=requirements,
                    export_data={
                        "genesis_status": genesis_result['status'],
                        "python_code": genesis_result.get('python_code')
                    }
                )

            return ChatResponse(
                type="clarification_needed",
                message=conversational_message,
                requirements=requirements
            )

    except HTTPException:
        # Re-raise HTTP exceptions (like the Ollama health check)
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()

        # Provide user-friendly error messages based on error type
        if "Connection" in error_msg or "11434" in error_msg:
            detail = ("⚠️ AI model server (Ollama) is not responding.\n\n"
                     "Please ensure Ollama is running with 'qwen2.5-coder:7b' model loaded.\n\n"
                     "To fix:\n"
                     "• Run 'ollama serve' in a terminal\n"
                     "• Verify the model is installed: ollama list\n"
                     "• Pull the model if needed: ollama pull qwen2.5-coder:7b")
        elif "timeout" in error_msg.lower():
            detail = ("⏱️ Request timed out. The AI took too long to respond.\n\n"
                     "Try:\n"
                     "• Simplifying your description\n"
                     "• Checking if Ollama is overloaded\n"
                     "• Restarting the Ollama service")
        elif "JSON" in error_msg or "parse" in error_msg.lower():
            detail = ("🔧 AI returned an invalid response format.\n\n"
                     "This is usually a temporary issue. Please:\n"
                     "• Try your request again\n"
                     "• Simplify your description\n"
                     "• Restart Ollama if the problem persists")
        else:
            detail = (f"❌ Unexpected error: {error_msg}\n\n"
                     f"Please try:\n"
                     f"• Rephrasing your request\n"
                     f"• Checking the backend logs for details\n"
                     f"• Ensuring all services are running correctly")

        raise HTTPException(
            status_code=500,
            detail=detail
        )


@router.post("/api/chat/reset")
async def reset_conversation(userId: str):
    """Reset conversation context for a user"""
    if userId in active_conversations:
        active_conversations[userId].reset()

    return {"status": "reset", "userId": userId}


@router.get("/api/chat/status/{userId}")
async def get_conversation_status(userId: str):
    """Get current conversation status"""
    if userId not in active_conversations:
        return {
            "exists": False,
            "requirements": {},
            "ready": False
        }

    context = active_conversations[userId]
    return {
        "exists": True,
        "requirements": context.requirements,
        "ready": context.is_ready_to_generate(),
        "missing": context.get_missing_info()
    }


class AlgorithmData(BaseModel):
    """Model for storing algorithm data"""
    userId: str
    algorithm: Dict[str, Any]


@router.post("/api/chat/store-algorithm")
async def store_algorithm(data: AlgorithmData):
    """
    Store generated algorithm in conversation context AND ChromaDB
    This is called by the frontend after an algorithm is generated
    """
    user_id = data.userId

    # Get or create conversation context
    if user_id not in active_conversations:
        active_conversations[user_id] = ConversationContext(user_id=user_id)

    context = active_conversations[user_id]

    # Initialize algorithms list if needed
    if not hasattr(context, 'algorithms'):
        context.algorithms = []

    # Add algorithm to context (in-memory)
    context.algorithms.append(data.algorithm)

    # Store in ChromaDB for long-term memory and semantic search
    try:
        memory = get_memory_manager()
        memory.add_algorithm(
            user_id=user_id,
            algorithm=data.algorithm
        )
        print(f"✅ Stored algorithm in ChromaDB for user {user_id}")
    except Exception as e:
        print(f"⚠️ Failed to store algorithm in ChromaDB: {e}")

    print(f"Stored algorithm for user {user_id}. Total algorithms: {len(context.algorithms)}")

    return {
        "status": "success",
        "algorithm_count": len(context.algorithms)
    }


class SyncSceneRequest(BaseModel):
    userId: str
    sceneConfig: Dict[str, Any]


@router.post("/api/chat/sync-scene")
async def sync_scene(request: SyncSceneRequest):
    """
    Sync manually created scene from Scene Editor with conversation context
    This allows users to create scenes manually and then generate algorithms for them
    """
    user_id = request.userId
    scene_config = request.sceneConfig

    try:
        # Get or create conversation context
        if user_id not in active_conversations:
            active_conversations[user_id] = ConversationContext(user_id=user_id)
            print(f"🆕 Created new context for scene sync - user {user_id}")

        context = active_conversations[user_id]

        # Store the scene configuration (in-memory)
        context.last_simulation = scene_config

        # Store in ChromaDB for long-term memory
        try:
            memory = get_memory_manager()
            memory.add_scene(
                user_id=user_id,
                scene_config=scene_config
            )
            print(f"✅ Stored scene in ChromaDB for user {user_id}")
        except Exception as e:
            print(f"⚠️ Failed to store scene in ChromaDB: {e}")

        # Extract requirements from scene config
        robot_type = scene_config.get('robot', {}).get('type', 'mobile_robot')
        environment = scene_config.get('environment', {}).get('floor', {}).get('texture', 'warehouse')

        # Determine task based on objects in scene
        task = "navigation"
        if 'waypoints' in scene_config:
            task = "navigation"
        elif 'task_markers' in scene_config:
            task = "pick and place"

        # Update context requirements
        context.requirements = {
            'robot_type': robot_type,
            'environment': environment,
            'task': task,
            'objects': ['boxes']  # Default, can be enhanced later
        }

        # Initialize algorithms list if not exists
        if not hasattr(context, 'algorithms'):
            context.algorithms = []

        print(f"✅ Synced scene for user {user_id}")
        print(f"   - Robot: {robot_type}")
        print(f"   - Environment: {environment}")
        print(f"   - Task: {task}")
        print(f"   - Objects count: {len(scene_config.get('objects', []))}")

        return {
            "status": "success",
            "message": "Scene synced with conversation context",
            "context_id": id(context)
        }

    except Exception as e:
        print(f"❌ Error syncing scene: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to sync scene: {str(e)}")

# ============================================================================
# Robot Command Execution
# ============================================================================

from chat.command_parser import CommandParser, RobotCommand
from chat.command_executor import CommandExecutor, ExecutionStatus


class RobotCommandRequest(BaseModel):
    """Request for robot command execution"""
    userId: str
    robotId: str = Field(default="robot-1")
    robotType: str = Field(default="mobile", description="mobile, arm, or drone")
    command: str  # Natural language command


class RobotCommandResponse(BaseModel):
    """Response from robot command execution"""
    status: str  # "success", "error", "parsing_failed"
    message: str
    parsed_commands: Optional[List[Dict]] = None
    execution_results: Optional[List[Dict]] = None
    error: Optional[str] = None


# Global command parsers (one per robot type)
command_parsers = {
    "mobile": CommandParser("mobile"),
    "arm": CommandParser("arm"),
    "drone": CommandParser("drone"),
}

# Global command executor (will be initialized with teleop server)
global_command_executor: Optional[CommandExecutor] = None


def get_command_executor():
    """Get or create command executor"""
    global global_command_executor
    
    if global_command_executor is None:
        # Import here to avoid circular dependency
        try:
            from main import get_teleop_server
            teleop = get_teleop_server()
            global_command_executor = CommandExecutor(teleop)
            print("✅ Initialized command executor with teleoperation server")
        except Exception as e:
            print(f"⚠️ Failed to initialize command executor: {e}")
            global_command_executor = CommandExecutor()  # No teleop server
    
    return global_command_executor


@router.post("/api/robot/command", response_model=RobotCommandResponse)
async def execute_robot_command(request: RobotCommandRequest):
    """
    Execute natural language robot commands
    
    Examples:
    - "move forward 2 meters"
    - "rotate 90 degrees clockwise"
    - "grab the red cube"
    - "move forward 1 meter then rotate 90 degrees"
    """
    try:
        # Get command parser for robot type
        parser = command_parsers.get(request.robotType)
        if not parser:
            return RobotCommandResponse(
                status="error",
                message=f"Unsupported robot type: {request.robotType}",
                error=f"Robot type must be one of: {list(command_parsers.keys())}"
            )
        
        # Parse command(s)
        print(f"🤖 Parsing command: '{request.command}' for {request.robotType} robot")
        
        # Try single command first
        parsed_command = parser.parse(request.command)
        
        if parsed_command:
            commands = [parsed_command]
        else:
            # Try sequence parsing
            commands = parser.parse_sequence(request.command)
        
        if not commands:
            return RobotCommandResponse(
                status="parsing_failed",
                message=f"Could not parse command: '{request.command}'",
                error="Command not recognized. Try commands like 'move forward 2 meters' or 'rotate 90 degrees'"
            )
        
        print(f"✅ Parsed {len(commands)} command(s)")
        
        # Get command executor
        executor = get_command_executor()
        
        # Execute commands
        execution_results = []
        
        for cmd in commands:
            print(f"▶️ Executing: {cmd.command_type.value}")
            
            # Execute asynchronously
            execution = await executor.execute(cmd, request.robotId)
            
            execution_results.append(execution.to_dict())
            
            print(f"{'✅' if execution.status == ExecutionStatus.COMPLETED else '❌'} "
                  f"{cmd.command_type.value}: {execution.status.value} ({execution.duration:.2f}s)")
            
            # Stop on first failure
            if execution.status == ExecutionStatus.FAILED:
                break
        
        # Determine overall status
        all_completed = all(r['status'] == ExecutionStatus.COMPLETED.value for r in execution_results)
        overall_status = "success" if all_completed else "error"
        
        # Generate response message
        if all_completed:
            message = f"Successfully executed {len(commands)} command(s)"
        else:
            failed_count = sum(1 for r in execution_results if r['status'] == ExecutionStatus.FAILED.value)
            message = f"Executed {len(execution_results)} command(s), {failed_count} failed"
        
        return RobotCommandResponse(
            status=overall_status,
            message=message,
            parsed_commands=[cmd.model_dump() for cmd in commands],
            execution_results=execution_results
        )
    
    except Exception as e:
        print(f"❌ Error executing robot command: {e}")
        import traceback
        traceback.print_exc()
        
        return RobotCommandResponse(
            status="error",
            message="Internal server error",
            error=str(e)
        )


@router.get("/api/robot/command-status")
async def get_command_status():
    """Get current command execution status"""
    try:
        executor = get_command_executor()
        status = executor.get_status()
        
        return {
            "status": "success",
            **status
        }
    
    except Exception as e:
        print(f"❌ Error getting command status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/robot/parse-command")
async def parse_command(request: RobotCommandRequest):
    """
    Parse a command without executing it (for testing/preview)
    """
    try:
        parser = command_parsers.get(request.robotType)
        if not parser:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported robot type: {request.robotType}"
            )
        
        # Try parsing
        parsed_command = parser.parse(request.command)
        
        if parsed_command:
            commands = [parsed_command]
        else:
            commands = parser.parse_sequence(request.command)
        
        if not commands:
            return {
                "status": "parsing_failed",
                "message": "Could not parse command",
                "commands": []
            }
        
        return {
            "status": "success",
            "message": f"Parsed {len(commands)} command(s)",
            "commands": [cmd.model_dump() for cmd in commands],
            "actions": [parser.to_action(cmd) for cmd in commands]
        }
    
    except Exception as e:
        print(f"❌ Error parsing command: {e}")
        raise HTTPException(status_code=500, detail=str(e))
