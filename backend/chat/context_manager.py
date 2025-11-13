import json
import requests
import os
from typing import Dict, List, Optional


class ConversationContext:
    """Manages conversation state and tracks simulation requirements"""

    def __init__(self):
        self.requirements = {}
        self.conversation_history = []
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def update_from_message(self, user_message: str) -> Dict:
        """Extract requirements from user's message and update context"""

        # Build prompt for Ollama to extract requirements
        prompt = f"""Extract simulation requirements from this user message:
"{user_message}"

Previous context: {json.dumps(self.requirements)}

Return ONLY valid JSON (no markdown, no explanations) with these fields:
{{
    "robot_type": "mobile" or "arm" or "drone" or "humanoid" or null,
    "task": "description of what robot should do" or null,
    "environment": "warehouse" or "office" or "outdoor" or "factory" or null,
    "budget": number or null,
    "objects": ["list", "of", "objects"] or [],
    "special_needs": ["list", "of", "requirements"] or []
}}

Examples:
- "warehouse robot" → {{"robot_type": "mobile", "environment": "warehouse"}}
- "pick up boxes" → {{"task": "pick up boxes", "objects": ["boxes"]}}
- "budget $3000" → {{"budget": 3000}}

User message: "{user_message}"

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
                timeout=30
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

        except Exception as e:
            print(f"Error extracting requirements: {e}")
            # Fallback: simple keyword detection
            self.requirements = self._fallback_extraction(user_message, self.requirements)

        # Add to conversation history
        self.conversation_history.append(user_message)

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
        elif 'navigate' in message_lower:
            reqs['task'] = 'navigate environment'

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

        return reqs

    def is_ready_to_generate(self) -> bool:
        """Check if we have minimum requirements to generate simulation"""
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

    def reset(self):
        """Reset conversation context"""
        self.requirements = {}
        self.conversation_history = []
