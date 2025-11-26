"""
Natural Language Robot Command Parser

Parses natural language commands into structured robot actions.
Supports mobile robots, arms, and drones with various command types.

Examples:
- "move forward 2 meters"
- "rotate 90 degrees clockwise"
- "grab the red cube"
- "fly to height 5 meters"
"""

import re
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field


class CommandType(str, Enum):
    """Types of robot commands"""
    MOVE = "move"
    ROTATE = "rotate"
    GRAB = "grab"
    RELEASE = "release"
    GOTO = "goto"
    WAIT = "wait"
    RESET = "reset"
    HOVER = "hover"  # Drone specific
    LAND = "land"    # Drone specific


class Direction(str, Enum):
    """Movement directions"""
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    CLOCKWISE = "clockwise"
    COUNTERCLOCKWISE = "counterclockwise"


class RobotCommand(BaseModel):
    """Structured robot command"""
    command_type: CommandType
    robot_type: str = Field(default="mobile", description="mobile, arm, or drone")

    # Movement parameters
    direction: Optional[Direction] = None
    distance: Optional[float] = None  # meters
    angle: Optional[float] = None  # degrees

    # Position parameters (for goto commands)
    position: Optional[List[float]] = Field(default=None, description="[x, y, z]")

    # Gripper parameters
    target_object: Optional[str] = None

    # Timing parameters
    duration: Optional[float] = None  # seconds

    # Original text for reference
    original_text: str = ""


class CommandParser:
    """
    Parse natural language into robot commands

    Uses pattern matching and keyword extraction to convert
    free-form text into structured RobotCommand objects.
    """

    def __init__(self, robot_type: str = "mobile"):
        """
        Args:
            robot_type: Type of robot (mobile, arm, drone)
        """
        self.robot_type = robot_type

        # Direction keywords
        self.direction_keywords = {
            "forward": Direction.FORWARD,
            "forwards": Direction.FORWARD,
            "ahead": Direction.FORWARD,
            "front": Direction.FORWARD,

            "backward": Direction.BACKWARD,
            "backwards": Direction.BACKWARD,
            "back": Direction.BACKWARD,
            "reverse": Direction.BACKWARD,

            "left": Direction.LEFT,

            "right": Direction.RIGHT,

            "up": Direction.UP,
            "upward": Direction.UP,
            "upwards": Direction.UP,
            "ascend": Direction.UP,

            "down": Direction.DOWN,
            "downward": Direction.DOWN,
            "downwards": Direction.DOWN,
            "descend": Direction.DOWN,

            "clockwise": Direction.CLOCKWISE,
            "cw": Direction.CLOCKWISE,

            "counterclockwise": Direction.COUNTERCLOCKWISE,
            "counter-clockwise": Direction.COUNTERCLOCKWISE,
            "ccw": Direction.COUNTERCLOCKWISE,
        }

        # Command patterns
        self.patterns = {
            CommandType.MOVE: [
                r"(?:move|go|drive|travel)\s+(\w+)(?:\s+(\d+(?:\.\d+)?)\s*(?:meters?|m|cm|centimeters?|feet?|ft))?",
                r"(\d+(?:\.\d+)?)\s*(?:meters?|m)\s+(\w+)",
            ],
            CommandType.ROTATE: [
                r"(?:rotate|turn|spin)\s+(?:(\d+(?:\.\d+)?)\s*(?:degrees?|deg|°))?\s*(\w+)?",
                r"(?:rotate|turn|spin)\s+(\w+)\s+(?:(\d+(?:\.\d+)?)\s*(?:degrees?|deg|°))?",
            ],
            CommandType.GRAB: [
                r"(?:grab|grasp|pick\s+up|take)\s+(?:the\s+)?(.+)",
            ],
            CommandType.RELEASE: [
                r"(?:release|drop|let\s+go|put\s+down)",
            ],
            CommandType.GOTO: [
                r"(?:go\s+to|move\s+to|navigate\s+to)\s+position\s+\(?([-\d.]+)[,\s]+([-\d.]+)(?:[,\s]+([-\d.]+))?\)?",
                r"(?:go\s+to|move\s+to)\s+\[?([-\d.]+)[,\s]+([-\d.]+)(?:[,\s]+([-\d.]+))?\]?",
            ],
            CommandType.WAIT: [
                r"(?:wait|pause|stop)\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:seconds?|s|ms|milliseconds?)?",
            ],
            CommandType.RESET: [
                r"(?:reset|home|initialize|return\s+to\s+start)",
            ],
            CommandType.HOVER: [
                r"(?:hover|hold\s+position)\s*(?:at\s+)?(?:(\d+(?:\.\d+)?)\s*(?:meters?|m))?",
            ],
            CommandType.LAND: [
                r"(?:land|descend\s+to\s+ground|touch\s+down)",
            ],
        }

    def parse(self, text: str) -> Optional[RobotCommand]:
        """
        Parse natural language text into a robot command

        Args:
            text: Natural language command

        Returns:
            RobotCommand if successful, None if parsing fails
        """
        text = text.lower().strip()

        # Try each command type
        for cmd_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return self._create_command(cmd_type, match, text)

        return None

    def _create_command(
        self,
        cmd_type: CommandType,
        match: re.Match,
        original_text: str
    ) -> RobotCommand:
        """Create a RobotCommand from regex match"""

        command = RobotCommand(
            command_type=cmd_type,
            robot_type=self.robot_type,
            original_text=original_text
        )

        if cmd_type == CommandType.MOVE:
            # Extract direction and distance
            groups = match.groups()

            # Try to find direction in either group
            direction_str = None
            distance_str = None

            for group in groups:
                if group and group in self.direction_keywords:
                    direction_str = group
                elif group and re.match(r'\d+(?:\.\d+)?', group):
                    distance_str = group

            if direction_str:
                command.direction = self.direction_keywords[direction_str]

            if distance_str:
                command.distance = float(distance_str)
            else:
                # Default distance if not specified
                command.distance = 1.0

        elif cmd_type == CommandType.ROTATE:
            # Extract angle and direction
            groups = match.groups()

            angle_str = None
            direction_str = None

            for group in groups:
                if group:
                    if re.match(r'\d+(?:\.\d+)?', group):
                        angle_str = group
                    elif group in self.direction_keywords:
                        direction_str = group

            if angle_str:
                command.angle = float(angle_str)
            else:
                command.angle = 90.0  # Default rotation

            if direction_str:
                command.direction = self.direction_keywords[direction_str]
            else:
                # Default to clockwise if not specified
                command.direction = Direction.CLOCKWISE

        elif cmd_type == CommandType.GRAB:
            # Extract target object
            target = match.group(1).strip()
            command.target_object = target

        elif cmd_type == CommandType.GOTO:
            # Extract position coordinates
            groups = match.groups()
            position = []
            for group in groups:
                if group is not None:
                    position.append(float(group))

            # Ensure we have at least x, y (add z=0 if needed)
            if len(position) == 2:
                position.append(0.0)

            command.position = position

        elif cmd_type == CommandType.WAIT:
            # Extract duration
            duration_str = match.group(1)
            command.duration = float(duration_str)

        elif cmd_type == CommandType.HOVER:
            # Extract altitude for drones
            if match.groups() and match.group(1):
                altitude = float(match.group(1))
                command.position = [0, 0, altitude]

        return command

    def parse_sequence(self, text: str) -> List[RobotCommand]:
        """
        Parse multiple commands from text (separated by 'then', 'and', semicolons, periods)

        Args:
            text: Text containing multiple commands

        Returns:
            List of RobotCommand objects
        """
        # Split on common separators
        separators = r'(?:\s+then\s+|\s+and\s+then\s+|[;.]|\s+and\s+)'
        parts = re.split(separators, text, flags=re.IGNORECASE)

        commands = []
        for part in parts:
            part = part.strip()
            if part:
                cmd = self.parse(part)
                if cmd:
                    commands.append(cmd)

        return commands

    def to_action(self, command: RobotCommand) -> Dict[str, Any]:
        """
        Convert RobotCommand to action format for teleoperation system

        Args:
            command: Structured robot command

        Returns:
            Action dictionary compatible with GenesisTeleopServer
        """
        action = {
            "type": command.command_type.value,
            "robot_type": command.robot_type,
        }

        if command.command_type == CommandType.MOVE:
            # Convert direction and distance to velocity
            velocity = self._direction_to_velocity(
                command.direction,
                command.distance or 1.0,
                command.robot_type
            )
            action["velocity"] = velocity
            action["duration"] = command.distance or 1.0  # Assume 1 m/s

        elif command.command_type == CommandType.ROTATE:
            action["angle"] = command.angle
            action["direction"] = command.direction.value if command.direction else "clockwise"

        elif command.command_type == CommandType.GRAB:
            action["target"] = command.target_object
            action["gripper_action"] = "close"

        elif command.command_type == CommandType.RELEASE:
            action["gripper_action"] = "open"

        elif command.command_type == CommandType.GOTO:
            action["target_position"] = command.position

        elif command.command_type == CommandType.WAIT:
            action["duration"] = command.duration

        return action

    def _direction_to_velocity(
        self,
        direction: Optional[Direction],
        magnitude: float,
        robot_type: str
    ) -> List[float]:
        """Convert direction to velocity vector"""

        if robot_type == "mobile":
            # Mobile robot: [vx, vy, omega]
            velocity_map = {
                Direction.FORWARD: [magnitude, 0, 0],
                Direction.BACKWARD: [-magnitude, 0, 0],
                Direction.LEFT: [0, magnitude, 0],
                Direction.RIGHT: [0, -magnitude, 0],
            }
            return velocity_map.get(direction, [0, 0, 0])

        elif robot_type == "drone":
            # Drone: [vx, vy, vz, omega]
            velocity_map = {
                Direction.FORWARD: [magnitude, 0, 0, 0],
                Direction.BACKWARD: [-magnitude, 0, 0, 0],
                Direction.LEFT: [0, magnitude, 0, 0],
                Direction.RIGHT: [0, -magnitude, 0, 0],
                Direction.UP: [0, 0, magnitude, 0],
                Direction.DOWN: [0, 0, -magnitude, 0],
            }
            return velocity_map.get(direction, [0, 0, 0, 0])

        return [0, 0, 0]


# Example usage
if __name__ == "__main__":
    parser = CommandParser(robot_type="mobile")

    # Test examples
    test_commands = [
        "move forward 2 meters",
        "rotate 90 degrees clockwise",
        "go backward 1.5 meters",
        "turn left 45 degrees",
        "grab the red cube",
        "release",
        "go to position (1.5, 2.0, 0)",
        "wait 3 seconds",
        "reset",
        "move forward 1 meter then rotate 90 degrees then grab the cube",
    ]

    print("🤖 Robot Command Parser Test\n")

    for text in test_commands:
        print(f"Input: '{text}'")

        # Try single command
        cmd = parser.parse(text)
        if cmd:
            print(f"  ✅ Parsed: {cmd.command_type.value}")
            print(f"     {cmd.model_dump_json(indent=6, exclude_none=True)}")
            action = parser.to_action(cmd)
            print(f"  🎯 Action: {action}")
        else:
            # Try sequence
            cmds = parser.parse_sequence(text)
            if cmds:
                print(f"  ✅ Parsed sequence ({len(cmds)} commands):")
                for i, c in enumerate(cmds, 1):
                    print(f"     {i}. {c.command_type.value}: {c.model_dump_json(exclude_none=True)}")
            else:
                print("  ❌ Failed to parse")

        print()
