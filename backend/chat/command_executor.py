"""
Robot Command Executor

Executes structured robot commands via the teleoperation system.
Handles timing, sequencing, and provides execution feedback.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from enum import Enum

try:
    from .command_parser import RobotCommand, CommandType, Direction
except ImportError:
    from command_parser import RobotCommand, CommandType, Direction


class ExecutionStatus(str, Enum):
    """Status of command execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CommandExecution:
    """Track execution of a single command"""

    def __init__(self, command: RobotCommand):
        self.command = command
        self.status = ExecutionStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.error: Optional[str] = None

    @property
    def duration(self) -> float:
        """Get execution duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for status updates"""
        return {
            "command_type": self.command.command_type.value,
            "status": self.status.value,
            "duration": self.duration,
            "error": self.error,
            "original_text": self.command.original_text,
        }


class CommandExecutor:
    """
    Execute robot commands via teleoperation system

    Converts RobotCommand objects into keyboard states or direct
    actions and executes them with proper timing.
    """

    def __init__(self, teleop_server=None):
        """
        Args:
            teleop_server: Instance of GenesisTeleopServer
        """
        self.teleop_server = teleop_server
        self.current_execution: Optional[CommandExecution] = None
        self.execution_history: List[CommandExecution] = []

        # Status callback (called on execution updates)
        self.status_callback: Optional[Callable[[Dict], None]] = None

    def set_status_callback(self, callback: Callable[[Dict], None]):
        """Set callback for execution status updates"""
        self.status_callback = callback

    async def execute(
        self,
        command: RobotCommand,
        robot_id: str = "robot-1"
    ) -> CommandExecution:
        """
        Execute a single command

        Args:
            command: RobotCommand to execute
            robot_id: ID of robot to control

        Returns:
            CommandExecution with status
        """
        execution = CommandExecution(command)
        self.current_execution = execution
        self.execution_history.append(execution)

        execution.status = ExecutionStatus.RUNNING
        execution.start_time = time.time()
        self._notify_status()

        try:
            # Execute based on command type
            if command.command_type == CommandType.MOVE:
                await self._execute_move(command, robot_id)

            elif command.command_type == CommandType.ROTATE:
                await self._execute_rotate(command, robot_id)

            elif command.command_type == CommandType.GRAB:
                await self._execute_grab(command, robot_id)

            elif command.command_type == CommandType.RELEASE:
                await self._execute_release(command, robot_id)

            elif command.command_type == CommandType.GOTO:
                await self._execute_goto(command, robot_id)

            elif command.command_type == CommandType.WAIT:
                await self._execute_wait(command)

            elif command.command_type == CommandType.RESET:
                await self._execute_reset(command, robot_id)

            elif command.command_type == CommandType.HOVER:
                await self._execute_hover(command, robot_id)

            elif command.command_type == CommandType.LAND:
                await self._execute_land(command, robot_id)

            execution.status = ExecutionStatus.COMPLETED

        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.error = str(e)

        finally:
            execution.end_time = time.time()
            self.current_execution = None
            self._notify_status()

        return execution

    async def execute_sequence(
        self,
        commands: List[RobotCommand],
        robot_id: str = "robot-1"
    ) -> List[CommandExecution]:
        """
        Execute a sequence of commands

        Args:
            commands: List of RobotCommands
            robot_id: ID of robot to control

        Returns:
            List of CommandExecution objects
        """
        executions = []

        for command in commands:
            execution = await self.execute(command, robot_id)
            executions.append(execution)

            # Stop if command failed
            if execution.status == ExecutionStatus.FAILED:
                break

        return executions

    # Command execution methods

    async def _execute_move(self, command: RobotCommand, robot_id: str):
        """Execute movement command"""
        if not self.teleop_server:
            raise RuntimeError("No teleoperation server connected")

        # Convert direction to keyboard state
        keys = self._direction_to_keys(command.direction, command.robot_type)

        # Calculate duration based on distance (assume 1 m/s)
        duration = command.distance or 1.0

        # Send keyboard state
        result = self.teleop_server.process_keyboard_input(robot_id, keys)

        # Hold for duration
        await asyncio.sleep(duration)

        # Release keys
        result = self.teleop_server.process_keyboard_input(
            robot_id,
            self._get_zero_keys()
        )

    async def _execute_rotate(self, command: RobotCommand, robot_id: str):
        """Execute rotation command"""
        if not self.teleop_server:
            raise RuntimeError("No teleoperation server connected")

        # Determine rotation key
        if command.direction == Direction.CLOCKWISE:
            keys = {"e": True}  # E for clockwise
        else:
            keys = {"q": True}  # Q for counterclockwise

        # Calculate duration (assume 90 deg/sec)
        angle = command.angle or 90.0
        duration = angle / 90.0

        # Send keyboard state
        self.teleop_server.process_keyboard_input(robot_id, keys)

        # Hold for duration
        await asyncio.sleep(duration)

        # Release keys
        self.teleop_server.process_keyboard_input(
            robot_id,
            self._get_zero_keys()
        )

    async def _execute_grab(self, command: RobotCommand, robot_id: str):
        """Execute grab command"""
        if not self.teleop_server:
            raise RuntimeError("No teleoperation server connected")

        # Press and hold space bar (gripper close)
        keys = {"space": True}
        self.teleop_server.process_keyboard_input(robot_id, keys)

        # Hold for 0.5 seconds
        await asyncio.sleep(0.5)

    async def _execute_release(self, command: RobotCommand, robot_id: str):
        """Execute release command"""
        if not self.teleop_server:
            raise RuntimeError("No teleoperation server connected")

        # Release space bar (gripper open)
        keys = {"space": False}
        self.teleop_server.process_keyboard_input(robot_id, keys)

        await asyncio.sleep(0.1)

    async def _execute_goto(self, command: RobotCommand, robot_id: str):
        """Execute goto command (navigate to position)"""
        # This is a higher-level command that would need path planning
        # For now, we'll just log it
        print(f"GoTo command: {command.position} (path planning not implemented)")

        # Placeholder - would integrate with path planner
        await asyncio.sleep(1.0)

    async def _execute_wait(self, command: RobotCommand):
        """Execute wait command"""
        duration = command.duration or 1.0
        await asyncio.sleep(duration)

    async def _execute_reset(self, command: RobotCommand, robot_id: str):
        """Execute reset command"""
        if not self.teleop_server:
            raise RuntimeError("No teleoperation server connected")

        self.teleop_server.reset_robot(robot_id)
        await asyncio.sleep(0.5)

    async def _execute_hover(self, command: RobotCommand, robot_id: str):
        """Execute hover command (drones)"""
        if not self.teleop_server:
            raise RuntimeError("No teleoperation server connected")

        # Set all keys to zero (maintain altitude)
        keys = self._get_zero_keys()
        self.teleop_server.process_keyboard_input(robot_id, keys)

        # Hold position for 2 seconds
        await asyncio.sleep(2.0)

    async def _execute_land(self, command: RobotCommand, robot_id: str):
        """Execute land command (drones)"""
        if not self.teleop_server:
            raise RuntimeError("No teleoperation server connected")

        # Gradually reduce altitude
        # (In real implementation, would read current altitude and descend)
        await asyncio.sleep(3.0)

    # Helper methods

    def _direction_to_keys(
        self,
        direction: Optional[Direction],
        robot_type: str
    ) -> Dict[str, bool]:
        """Convert direction to keyboard state"""

        keys = self._get_zero_keys()

        if direction == Direction.FORWARD:
            keys["w"] = True
        elif direction == Direction.BACKWARD:
            keys["s"] = True
        elif direction == Direction.LEFT:
            keys["a"] = True
        elif direction == Direction.RIGHT:
            keys["d"] = True
        elif direction == Direction.UP:
            if robot_type == "drone":
                keys["space"] = True
            else:
                keys["n"] = True
        elif direction == Direction.DOWN:
            keys["m"] = True

        return keys

    def _get_zero_keys(self) -> Dict[str, bool]:
        """Get keyboard state with all keys released"""
        return {
            "w": False,
            "a": False,
            "s": False,
            "d": False,
            "q": False,
            "e": False,
            "up": False,
            "down": False,
            "left": False,
            "right": False,
            "n": False,
            "m": False,
            "j": False,
            "k": False,
            "space": False,
            "u": False,
        }

    def _notify_status(self):
        """Notify callback of status change"""
        if self.status_callback and self.current_execution:
            self.status_callback(self.current_execution.to_dict())

    def get_status(self) -> Dict:
        """Get current execution status"""
        return {
            "current_execution": self.current_execution.to_dict() if self.current_execution else None,
            "history": [e.to_dict() for e in self.execution_history[-10:]],  # Last 10
        }


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('..')

    from chat.command_parser import CommandParser

    async def main():
        # Create parser and executor
        parser = CommandParser(robot_type="mobile")
        executor = CommandExecutor()  # No teleop server for testing

        # Set status callback
        def on_status(status: Dict):
            print(f"📊 Status: {status}")

        executor.set_status_callback(on_status)

        # Test commands
        test_commands = [
            "move forward 2 meters",
            "wait 1 second",
            "rotate 90 degrees clockwise",
            "wait 1 second",
            "move backward 1 meter",
        ]

        print("🤖 Command Executor Test\n")

        # Parse commands
        commands = []
        for text in test_commands:
            cmd = parser.parse(text)
            if cmd:
                commands.append(cmd)
                print(f"✅ Parsed: {text}")

        print(f"\n▶️ Executing {len(commands)} commands...\n")

        # Execute sequence
        executions = await executor.execute_sequence(commands)

        print(f"\n✅ Execution complete!\n")
        print("📋 Summary:")
        for i, exe in enumerate(executions, 1):
            status_emoji = "✅" if exe.status == ExecutionStatus.COMPLETED else "❌"
            print(f"  {status_emoji} {i}. {exe.command.command_type.value}: {exe.status.value} ({exe.duration:.2f}s)")

    # Run test
    asyncio.run(main())
