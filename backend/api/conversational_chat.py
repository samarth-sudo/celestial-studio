from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
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


def detect_isaac_simulation_request(message: str) -> bool:
    """
    Detect if user wants Isaac Lab GPU simulation (vs Rapier preview)
    """
    message_lower = message.lower()

    isaac_keywords = [
        'isaac lab', 'isaac sim', 'gpu simulation', 'cloud simulation',
        'accurate simulation', 'high fidelity', 'physics accurate',
        'photorealistic', 'modal', 'real physics', 'gpu physics'
    ]

    simulation_keywords = ['simulate', 'simulation', 'run']

    has_isaac = any(keyword in message_lower for keyword in isaac_keywords)
    has_sim = any(keyword in message_lower for keyword in simulation_keywords)

    return has_isaac or (has_sim and 'accurate' in message_lower)


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


async def handle_isaac_simulation_request(context: ConversationContext) -> ChatResponse:
    """Handle Isaac Lab GPU simulation request"""
    try:
        # Check if simulation exists
        if not hasattr(context, 'last_simulation') or not context.last_simulation:
            message = """**Isaac Lab GPU Simulation**

To run an Isaac Lab simulation, I need a robot scene first. You can:

1. **Create a robot:** "Create a mobile robot for warehouse navigation"
2. **Load a preset:** "Load the simple arm preset"
3. **Build custom:** "Build a 2-DOF robot arm"

Once we have a scene, I can run it on Modal's cloud GPUs with Isaac Lab for photorealistic, physics-accurate simulation.

What robot would you like to simulate?"""

            return ChatResponse(
                type="chat",
                message=message
            )

        # We have a simulation - offer Isaac Lab options
        robot_type = context.requirements.get('robot_type', 'robot')

        message = f"""**Run Isaac Lab Simulation**

I can run your {robot_type} simulation on NVIDIA Isaac Lab with cloud GPUs!

**What you'll get:**
- GPU-accelerated physics simulation
- Photorealistic rendering
- Recorded video of the simulation
- Physics-accurate behavior

**Options:**
- Duration: 5-60 seconds
- Video recording: Yes/No
- FPS: 30-60

**Cost estimate:** ~$0.10-0.50 for 10-30 seconds (Modal GPU pricing)

**Reply with:**
- "Yes, run simulation for 10 seconds"
- "Simulate for 30 seconds with video"
- "Just preview" (to use browser-based Rapier instead)

Would you like to proceed with Isaac Lab simulation?"""

        # Store pending action
        context.pending_action = {
            'type': 'isaac_simulation',
            'scene_config': context.last_simulation
        }

        return ChatResponse(
            type="chat",
            message=message,
            export_data={
                "action": "isaac_simulation_confirmation",
                "scene_config": context.last_simulation
            }
        )

    except Exception as e:
        print(f"Isaac simulation request failed: {e}")
        return ChatResponse(
            type="chat",
            message=f"Error setting up Isaac Lab simulation: {str(e)}"
        )


async def handle_confirmation_response(context: ConversationContext, message: str) -> ChatResponse:
    """Handle user's confirmation response to Isaac Lab or training requests"""
    try:
        from isaac_lab.task_mapper import get_task_mapper

        pending = context.pending_action
        action_type = pending.get('type')

        # Parse the confirmation
        task_mapper = get_task_mapper()
        params = task_mapper.parse_user_confirmation(message)

        if not params:
            # Couldn't parse, ask for clarification
            return ChatResponse(
                type="chat",
                message="I didn't understand that. Please reply with 'yes' to proceed or 'no' to cancel."
            )

        if params.get('cancelled'):
            # User cancelled
            del context.pending_action
            return ChatResponse(
                type="chat",
                message="Okay, cancelled. Let me know if you'd like to try something else!"
            )

        if not params.get('confirmed'):
            # User didn't confirm
            return ChatResponse(
                type="chat",
                message="Please reply with 'yes' to proceed or 'no' to cancel."
            )

        # User confirmed!
        if action_type == 'isaac_simulation':
            # Launch Isaac Lab simulation
            duration = params.get('duration', 10)
            scene_config = pending.get('scene_config')

            del context.pending_action

            return ChatResponse(
                type="chat",
                message=f"""**Starting Isaac Lab Simulation**

Launching simulation on Modal GPU...
- Duration: {duration} seconds
- Scene: {context.requirements.get('robot_type', 'robot')}
- Video recording: Yes
- FPS: 30

This will take about 2-3 minutes. Switch to the "Isaac Lab" tab in the simulator to view the configuration panel.

The simulation will run on cloud GPUs and stream video to your browser when ready."""
            )

        elif action_type == 'training':
            # Launch RL training
            algorithm = params.get('algorithm', 'PPO')
            max_iterations = params.get('max_iterations', 1000)
            suggested_task = pending.get('suggested_task')

            del context.pending_action

            return ChatResponse(
                type="chat",
                message=f"""**Starting RL Training**

Launching training on Modal GPU...
- Algorithm: {algorithm}
- Task: {suggested_task or 'Custom scene'}
- Iterations: {max_iterations}
- Parallel envs: 2048

This will take 30-60 minutes. I'll stream progress updates in real-time via the training dashboard.

Training has started! You can monitor progress at: /api/training/progress/[training_id]"""
            )

        else:
            del context.pending_action
            return ChatResponse(
                type="chat",
                message="Unknown action type. Please try again."
            )

    except Exception as e:
        print(f"Confirmation handling error: {e}")
        import traceback
        traceback.print_exc()

        if hasattr(context, 'pending_action'):
            del context.pending_action

        return ChatResponse(
            type="chat",
            message=f"Error processing confirmation: {str(e)}"
        )


async def handle_training_request(context: ConversationContext) -> ChatResponse:
    """Handle RL training request"""
    try:
        # Check if simulation exists
        if not hasattr(context, 'last_simulation') or not context.last_simulation:
            message = """**Robot Training with RL**

To train a robot, I need a robot scene first. You can:

1. **Create a robot:** "Create a robotic arm for pick and place"
2. **Use a pre-built task:** "Use Isaac Lab's Franka reaching task"

Training uses reinforcement learning (PPO, SAC, or RSL-RL) on cloud GPUs.

What robot would you like to train?"""

            return ChatResponse(
                type="chat",
                message=message
            )

        # Try to suggest an Isaac Lab task
        try:
            from isaac_lab.scene_converter import suggest_training_task
            suggested_task = suggest_training_task(context.last_simulation)
        except:
            suggested_task = None

        robot_type = context.requirements.get('robot_type', 'robot')
        task = context.requirements.get('task', 'unknown task')

        message = f"""**Train Robot Policy**

I can train your {robot_type} for {task} using reinforcement learning on cloud GPUs!

**Training Configuration:**"""

        if suggested_task:
            message += f"\n- **Suggested Task:** {suggested_task} (pre-built Isaac Lab task)"
            message += "\n- Or: Train on your custom scene"
        else:
            message += "\n- Custom scene training"

        message += """
- **Algorithm:** PPO (default), SAC, or RSL-RL
- **Environments:** 2048 parallel robots
- **Iterations:** 1000 (adjustable)
- **Duration:** ~30-60 minutes on A10G GPU

**Cost estimate:** ~$5-15 for full training run (Modal GPU pricing)

**What you'll get:**
- Trained policy model (.pth file)
- Training logs and metrics
- Ready to deploy on real robot

**Reply with:**
- "Yes, train with PPO"
- "Train using SAC algorithm"""

        if suggested_task:
            message += f'\n- "Use the {suggested_task} task"'

        message += '\n- "Just simulate" (to skip training)\n\nWould you like to proceed with training?'

        # Store pending action
        context.pending_action = {
            'type': 'training',
            'suggested_task': suggested_task,
            'scene_config': context.last_simulation
        }

        return ChatResponse(
            type="chat",
            message=message,
            export_data={
                "action": "training_confirmation",
                "suggested_task": suggested_task,
                "scene_config": context.last_simulation
            }
        )

    except Exception as e:
        print(f"Training request failed: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            type="chat",
            message=f"Error setting up training: {str(e)}"
        )


async def handle_algorithm_request(context: ConversationContext, message: str) -> ChatResponse:
    """Handle algorithm generation request"""
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
            message=f"""**Generating {algorithm_name} Algorithm**

I'll create a custom {algorithm_name.lower()} algorithm for your {robot_type} robot!

**Algorithm Details:**
- Type: {algorithm_name}
- Robot: {robot_type.capitalize()}
- Task: {task}
- Generation: Using latest research and best practices

This will take 10-20 seconds to generate optimized code...

Once generated, you'll be able to:
- View the algorithm code
- Tune parameters in real-time
- Test it in the simulation
- Export it to React, ROS, or Python

**The algorithm is being generated now!** Switch to the "Algorithms" tab to view it when ready."""
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

        # Check if this is an Isaac Lab simulation request
        if detect_isaac_simulation_request(message):
            return await handle_isaac_simulation_request(context)

        # Check if this is an algorithm generation request (BEFORE training check!)
        # This prevents "generate algorithms for better travel" from triggering training
        if detect_algorithm_request(message):
            return await handle_algorithm_request(context, message)

        # Check if this is a training request
        if detect_training_request(message):
            return await handle_training_request(context)

        # Check if this is a confirmation/response to Isaac Lab or training
        if hasattr(context, 'pending_action'):
            return await handle_confirmation_response(context, message)

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

            # Build conversational response message
            robot_type = requirements.get('robot_type', 'robot')
            environment = requirements.get('environment', 'environment')
            task = requirements.get('task', 'task')

            response_message = f"""Perfect! I've created your simulation:

- {robot_type.capitalize()} robot
- {environment.capitalize()} environment
- Task: {task}
- Objects: {', '.join(requirements.get('objects', ['boxes']))}

The simulation is now ready in the 3D viewer!

**Next steps:**
- Generate algorithms for your robot
- Say "compare algorithms" to benchmark and find the best one
- Say "download as React" or "export as ROS" to get your complete project"""

            return ChatResponse(
                type="simulation_ready",
                message=response_message,
                simulation=simulation_config,
                requirements=requirements
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

                # Build response message acknowledging we're generating with partial info
                robot_type = requirements.get('robot_type', 'mobile robot')
                environment = requirements.get('environment', 'simulation environment')
                task = requirements.get('task', 'autonomous navigation')

                response_message = f"""Got it! I'll create the simulation with the information you've provided:

- Robot: {robot_type}
- Environment: {environment}
- Task: {task}

The simulation is now ready in the 3D viewer! You can refine it by chatting with me.

**Next steps:**
- Generate algorithms for your robot
- Say "compare algorithms" to benchmark and find the best one
- Say "download as React" or "export as ROS" to get your complete project"""

                return ChatResponse(
                    type="simulation_ready",
                    message=response_message,
                    simulation=simulation_config,
                    requirements=requirements
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
