from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat.context_manager import ConversationContext
from chat.question_generator import SimulationQuestionGenerator
from simulation.scene_generator import SceneGenerator

router = APIRouter()

# Store active conversations (in production, use Redis or database)
active_conversations: Dict[str, ConversationContext] = {}


class ChatRequest(BaseModel):
    userId: str
    message: str


class ChatResponse(BaseModel):
    type: str  # "clarification_needed", "simulation_ready", "chat"
    message: str
    questions: Optional[list] = None
    simulation: Optional[dict] = None
    requirements: Optional[dict] = None


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main conversational chat endpoint

    Handles the conversation flow:
    1. Extract requirements from user message
    2. Check if we have enough info
    3. Either ask questions or generate simulation
    """

    user_id = request.userId
    message = request.message

    try:
        # Get or create conversation context
        if user_id not in active_conversations:
            active_conversations[user_id] = ConversationContext()

        context = active_conversations[user_id]

        # Update context with user's message
        requirements = context.update_from_message(message)

        # Check if we can generate simulation
        if context.is_ready_to_generate():
            # We have enough info! Generate simulation
            scene_generator = SceneGenerator()
            simulation_config = scene_generator.generate_from_requirements(requirements)

            # Build response message
            robot_type = requirements.get('robot_type', 'robot')
            environment = requirements.get('environment', 'environment')
            task = requirements.get('task', 'task')

            response_message = f"""Perfect! I've created your simulation:

✓ {robot_type.capitalize()} robot
✓ {environment.capitalize()} environment
✓ Task: {task}
✓ Objects: {', '.join(requirements.get('objects', ['boxes']))}

The simulation is now ready in the 3D viewer!"""

            return ChatResponse(
                type="simulation_ready",
                message=response_message,
                simulation=simulation_config,
                requirements=requirements
            )
        else:
            # Need more information - ask questions
            question_gen = SimulationQuestionGenerator()
            missing_info = context.get_missing_info()

            # Generate questions
            questions_list = question_gen.generate_questions(missing_info)
            question_message = question_gen.format_questions_as_message(questions_list)

            return ChatResponse(
                type="clarification_needed",
                message=question_message,
                questions=questions_list,
                requirements=requirements
            )

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
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
