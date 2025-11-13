from typing import List, Dict


class SimulationQuestionGenerator:
    """Generates clarifying questions based on missing information"""

    def __init__(self):
        self.question_templates = {
            "type of robot": {
                "question": "What type of robot would you like to simulate?",
                "options": [
                    "Mobile robot (wheels, navigation)",
                    "Robotic arm (manipulation, pick-and-place)",
                    "Drone (quadcopter, flying)",
                    "Humanoid (walking, human-like)"
                ]
            },
            "what the robot should do": {
                "question": "What should the robot do?",
                "options": [
                    "Navigate and avoid obstacles",
                    "Pick and place objects",
                    "Inspect or survey an area",
                    "Deliver items from point A to B",
                    "Follow a specific path"
                ]
            },
            "where it will operate": {
                "question": "Where will the robot operate?",
                "options": [
                    "Warehouse (indoor, structured)",
                    "Office (indoor, furniture)",
                    "Outdoor (unstructured terrain)",
                    "Factory (industrial setting)"
                ]
            },
            "budget": {
                "question": "What's your budget range? (optional)",
                "options": [
                    "Under $1,000 (hobby/learning)",
                    "$1,000 - $5,000 (educational)",
                    "$5,000 - $20,000 (research)",
                    "Over $20,000 (industrial)"
                ]
            },
            "objects": {
                "question": "What objects should be in the scene?",
                "options": [
                    "Boxes (pick-and-place targets)",
                    "Shelves (storage units)",
                    "Obstacles (things to avoid)",
                    "People (human interaction)",
                    "Custom objects"
                ]
            }
        }

    def generate_questions(self, missing_info: List[str]) -> List[Dict]:
        """Generate natural questions based on missing information"""

        questions = []

        for info_type in missing_info:
            if info_type in self.question_templates:
                template = self.question_templates[info_type]
                questions.append({
                    "text": template["question"],
                    "options": template.get("options", []),
                    "field": info_type
                })

        return questions

    def format_questions_as_message(self, questions: List[Dict]) -> str:
        """Format questions into a natural conversation message"""

        if not questions:
            return "I have all the information I need!"

        message_parts = ["I can help with that! I need a few more details:\n"]

        for i, q in enumerate(questions, 1):
            message_parts.append(f"\n{i}. {q['text']}")
            if q.get('options'):
                message_parts.append("   Options:")
                for opt in q['options']:
                    message_parts.append(f"   • {opt}")

        message_parts.append("\n\nJust describe what you want in your own words!")

        return "\n".join(message_parts)

    def get_quick_questions(self, missing_info: List[str]) -> str:
        """Generate a concise question message"""

        if not missing_info:
            return "Perfect! I have everything I need."

        questions = []
        for info in missing_info:
            if info == "type of robot":
                questions.append("What type of robot? (mobile/arm/drone)")
            elif info == "what the robot should do":
                questions.append("What should it do?")
            elif info == "where it will operate":
                questions.append("Where will it operate?")

        if len(questions) == 1:
            return f"Quick question: {questions[0]}"
        else:
            return "I need to know:\n" + "\n".join([f"• {q}" for q in questions])
