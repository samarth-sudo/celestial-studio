import json
import requests
import os
from typing import Dict, List, Optional

try:
    from chat.memory_manager import get_memory_manager
except ImportError:
    from backend.chat.memory_manager import get_memory_manager


class ConversationContext:
    """Manages conversation state and tracks simulation requirements"""

    def __init__(self, user_id: str = "default_user"):
        self.requirements = {}
        self.conversation_history = []
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.user_id = user_id

        # Question limit to prevent infinite loops
        self.question_count = 0
        self.MAX_QUESTIONS = 2  # Maximum clarification questions before forcing generation

        # Initialize memory manager for long-term storage
        self.memory = get_memory_manager()

        # Load recent conversation history from ChromaDB
        self._load_recent_history()

    def update_from_message(self, user_message: str) -> Dict:
        """Extract requirements from user's message and update context

        This now extracts requirements silently in the background.
        The chat endpoint will generate conversational responses separately.
        """

        # Build prompt for Ollama to extract requirements
        prompt = f"""You are extracting simulation requirements from a user's message.

User's message: "{user_message}"

Previous requirements gathered: {json.dumps(self.requirements)}

Extract ONLY what the user mentioned in this message and merge with previous requirements.

Return ONLY valid JSON (no markdown, no explanations) with these fields:
{{
    "robot_type": "mobile" or "arm" or "drone" or "humanoid" or null,
    "robot_dof": number (degrees of freedom for arm robots) or null,
    "task": "description of what robot should do" or null,
    "environment": "warehouse" or "office" or "outdoor" or "factory" or null,
    "budget": number or null,
    "objects": ["list", "of", "objects"] or [],
    "task_objects": [{{"type": "box/cube/sphere", "color": "red/blue/green/yellow/orange/purple/etc", "count": number, "role": "target/distractor/obstacle"}}] or [],
    "special_needs": ["list", "of", "requirements"] or []
}}

IMPORTANT: For task_objects, extract specific object details from the user's description:
- Identify colors mentioned (red, blue, green, yellow, orange, etc.)
- Count how many objects of each type/color
- Determine object role: "target" (object to pick/interact with) or "distractor" (other objects)
- **CRITICAL**: When user says "N different color objects", create N SEPARATE task_object entries, each with a UNIQUE color

Examples:
- "warehouse robot" → {{"robot_type": "mobile", "environment": "warehouse"}}
- "4 dof arm" → {{"robot_type": "arm", "robot_dof": 4}}
- "pick red cube from 4 different color cubes" → {{"task": "pick red cube", "task_objects": [{{"type": "cube", "color": "red", "count": 1, "role": "target"}}, {{"type": "cube", "color": "blue", "count": 1, "role": "distractor"}}, {{"type": "cube", "color": "green", "count": 1, "role": "distractor"}}, {{"type": "cube", "color": "yellow", "count": 1, "role": "distractor"}}, {{"type": "cube", "color": "orange", "count": 1, "role": "distractor"}}]}}
- "3 red boxes and 2 blue boxes" → {{"task_objects": [{{"type": "box", "color": "red", "count": 3, "role": "distractor"}}, {{"type": "box", "color": "blue", "count": 2, "role": "distractor"}}]}}
- "5 different colored spheres" → {{"task_objects": [{{"type": "sphere", "color": "red", "count": 1, "role": "distractor"}}, {{"type": "sphere", "color": "blue", "count": 1, "role": "distractor"}}, {{"type": "sphere", "color": "green", "count": 1, "role": "distractor"}}, {{"type": "sphere", "color": "yellow", "count": 1, "role": "distractor"}}, {{"type": "sphere", "color": "purple", "count": 1, "role": "distractor"}}]}}

JSON:"""

        try:
            # Call Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen2.5-coder:7b",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1
                },
                timeout=60  # Increased from 30 to 60 seconds for complex extractions
            )

            result = response.json()
            generated_text = result.get('response', '{}')

            # Extract JSON from response (handle markdown code blocks)
            json_text = generated_text.strip()
            if json_text.startswith('```'):
                # Remove markdown code block markers
                lines = json_text.split('\n')
                json_text = '\n'.join([l for l in lines if not l.startswith('```')])

            # Parse extracted requirements
            extracted = json.loads(json_text)

            # Merge with existing requirements (new values override old)
            for key, value in extracted.items():
                if value is not None and value != [] and value != "":
                    self.requirements[key] = value

            # Apply smart defaults based on robot type (if environment not specified)
            if 'robot_type' in self.requirements and 'environment' not in self.requirements:
                robot_type = self.requirements['robot_type']
                if robot_type == 'arm':
                    self.requirements['environment'] = 'tabletop'
                    print(f"✨ Auto-set environment to 'tabletop' for arm robot")
                elif robot_type == 'mobile':
                    self.requirements['environment'] = 'warehouse'
                    print(f"✨ Auto-set environment to 'warehouse' for mobile robot")
                elif robot_type == 'drone':
                    self.requirements['environment'] = 'open_space'
                    print(f"✨ Auto-set environment to 'open_space' for drone")
                elif robot_type == 'humanoid':
                    self.requirements['environment'] = 'indoor'
                    print(f"✨ Auto-set environment to 'indoor' for humanoid")

        except Exception as e:
            print(f"Error extracting requirements: {e}")
            # Fallback: simple keyword detection
            self.requirements = self._fallback_extraction(user_message, self.requirements)

        # Add to conversation history (in-memory)
        self.conversation_history.append(user_message)

        # Store in ChromaDB for long-term memory
        self.memory.add_conversation_message(
            user_id=self.user_id,
            message=user_message,
            role="user",
            metadata={"requirements": self.requirements}
        )

        return self.requirements

    def _fallback_extraction(self, message: str, current_reqs: Dict) -> Dict:
        """Simple keyword-based extraction as fallback"""
        message_lower = message.lower()
        reqs = current_reqs.copy()

        # Robot type detection
        if 'mobile' in message_lower or 'wheel' in message_lower:
            reqs['robot_type'] = 'mobile'
        elif 'arm' in message_lower or 'manipulator' in message_lower:
            reqs['robot_type'] = 'arm'
        elif 'drone' in message_lower or 'quadcopter' in message_lower:
            reqs['robot_type'] = 'drone'

        # Environment detection
        if 'warehouse' in message_lower:
            reqs['environment'] = 'warehouse'
        elif 'office' in message_lower:
            reqs['environment'] = 'office'
        elif 'outdoor' in message_lower:
            reqs['environment'] = 'outdoor'

        # Task detection
        if 'pick' in message_lower or 'grab' in message_lower:
            reqs['task'] = 'pick and place objects'
            if 'objects' not in reqs:
                reqs['objects'] = ['boxes']
        elif 'navigate' in message_lower or 'avoid' in message_lower:
            if 'avoid' in message_lower or 'obstacle' in message_lower:
                reqs['task'] = 'navigate and avoid obstacles'
                if 'objects' not in reqs:
                    reqs['objects'] = ['obstacles']
            else:
                reqs['task'] = 'navigate environment'
        elif 'deliver' in message_lower:
            reqs['task'] = 'deliver items from point A to B'
        elif 'inspect' in message_lower or 'survey' in message_lower:
            reqs['task'] = 'inspect or survey an area'
        elif 'follow' in message_lower and 'path' in message_lower:
            reqs['task'] = 'follow a specific path'

        # Budget detection
        if '$' in message:
            import re
            budget_match = re.search(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', message)
            if budget_match:
                budget_str = budget_match.group(1).replace(',', '')
                reqs['budget'] = int(float(budget_str))

        # Objects detection
        objects = []
        if 'box' in message_lower:
            objects.append('boxes')
        if 'obstacle' in message_lower:
            objects.append('obstacles')
        if 'shelf' in message_lower or 'shelves' in message_lower:
            objects.append('shelves')
        if objects:
            reqs['objects'] = objects

        # Apply smart defaults based on robot type (if environment not specified)
        if 'robot_type' in reqs and 'environment' not in reqs:
            robot_type = reqs['robot_type']
            if robot_type == 'arm':
                reqs['environment'] = 'tabletop'
            elif robot_type == 'mobile':
                reqs['environment'] = 'warehouse'
            elif robot_type == 'drone':
                reqs['environment'] = 'open_space'
            elif robot_type == 'humanoid':
                reqs['environment'] = 'indoor'

        return reqs

    def is_ready_to_generate(self) -> bool:
        """Check if we have minimum requirements to generate simulation

        Returns True if:
        1. We have all required fields (robot_type, task, environment), OR
        2. We've asked maximum number of questions (force generation with partial info)
        """
        # Force generation if we've asked too many questions
        if self.question_count >= self.MAX_QUESTIONS:
            print(f"⚠️ Question limit reached ({self.question_count}/{self.MAX_QUESTIONS}), forcing generation")
            return True

        # Check for required fields
        required_fields = ['robot_type', 'task', 'environment']
        has_required = all(field in self.requirements for field in required_fields)

        # Also check that values are not empty
        if has_required:
            for field in required_fields:
                if not self.requirements[field]:
                    return False

        return has_required

    def get_missing_info(self) -> List[str]:
        """Get list of missing required information"""
        required_fields = {
            'robot_type': 'type of robot',
            'task': 'what the robot should do',
            'environment': 'where it will operate'
        }

        missing = []
        for field, description in required_fields.items():
            if field not in self.requirements or not self.requirements[field]:
                missing.append(description)

        return missing

    def generate_conversational_response(self, user_message: str) -> str:
        """Generate a friendly, ChatGPT-like conversational response

        This acknowledges what the user said and naturally asks for missing info.
        Returns None if we should stop asking and generate simulation.
        """
        # Increment question counter
        self.question_count += 1
        print(f"📊 Question count: {self.question_count}/{self.MAX_QUESTIONS}")

        # If we've asked too many questions, return None to trigger generation
        if self.question_count >= self.MAX_QUESTIONS:
            print(f"⚠️ Maximum questions exceeded, stopping clarification")
            return None

        # Build conversation history for context
        history_text = "\n".join([f"User: {msg}" for msg in self.conversation_history[-3:]])  # Last 3 messages

        missing_info = self.get_missing_info()

        if not missing_info:
            # We have everything! Just acknowledge
            return None  # Will generate simulation

        # Build prompt for natural conversation
        prompt = f"""You are a friendly AI assistant helping a user create a robot simulation.

Conversation so far:
{history_text}
User: {user_message}

Current requirements collected:
{json.dumps(self.requirements, indent=2)}

Still needed: {', '.join(missing_info)}

Generate a friendly, natural response that:
1. Acknowledges what the user just said
2. Asks about the missing information naturally (like ChatGPT)
3. Keep it conversational and helpful
4. Don't mention "requirements" or "simulation config" - just chat naturally
5. Ask about ONE missing thing at a time (don't overwhelm with multiple questions)

Examples of good responses:
- "Got it! A warehouse robot sounds interesting. What task would you like it to perform?"
- "Perfect! I'm picturing a mobile robot. Where should this robot operate - warehouse, office, or outdoor environment?"
- "Nice! So we have a robot that picks objects. What kind of environment should I set up for testing this?"

Your response (2-3 sentences max, friendly tone):"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen2.5-coder:7b",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7  # Higher for more natural conversation
                },
                timeout=30
            )

            result = response.json()
            conversational_response = result.get('response', '').strip()

            # Remove any quotes if LLM added them
            if conversational_response.startswith('"') and conversational_response.endswith('"'):
                conversational_response = conversational_response[1:-1]

            return conversational_response

        except Exception as e:
            print(f"Error generating conversational response: {e}")
            # Fallback to simple question
            missing_item = missing_info[0] if missing_info else "robot details"
            return f"Thanks! Could you tell me more about the {missing_item}?"

    def _load_recent_history(self):
        """Load recent conversation history from ChromaDB"""
        try:
            messages = self.memory.get_conversation_history(
                user_id=self.user_id,
                limit=10  # Load last 10 messages
            )
            self.conversation_history = [msg['content'] for msg in messages if msg['role'] == 'user']
            print(f"📚 Loaded {len(self.conversation_history)} recent messages from memory")
        except Exception as e:
            print(f"⚠️ Failed to load conversation history: {e}")
            self.conversation_history = []

    def get_llm_context(self, current_message: str) -> str:
        """
        Get token-efficient context for LLM using ChromaDB semantic search

        This replaces sending all conversation history by only sending
        relevant past context in TOON format

        Args:
            current_message: Current user message

        Returns:
            Formatted context string for LLM prompt
        """
        try:
            # Get relevant context from ChromaDB (uses semantic search + TOON format)
            context = self.memory.get_relevant_context(
                user_id=self.user_id,
                current_message=current_message,
                max_messages=10,
                max_algorithms=3
            )
            return context
        except Exception as e:
            print(f"⚠️ Failed to get LLM context: {e}")
            # Fallback to last 3 messages
            recent = "\n".join(self.conversation_history[-3:])
            return f"Recent conversation:\n{recent}"

    def reset(self):
        """Reset conversation context"""
        self.requirements = {}
        self.conversation_history = []
