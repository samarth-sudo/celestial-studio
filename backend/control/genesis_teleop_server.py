"""
Genesis Teleoperation Server with Recording
Adapted from: Genesis/examples/keyboard_teleop.py

Keyboard Controls (WebSocket):
WASD     - Move Forward/Left/Back/Right (Mobile Robot)
QE       - Rotate Left/Right (Mobile Robot)
↑↓←→     - XY Movement (Arm)
NM       - Z Movement Up/Down (Arm)
JK       - Rotate Counterclockwise/Clockwise (Arm)
Space    - Gripper (Close when pressed)
U        - Reset Scene
"""

import os
import csv
import time
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
from scipy.spatial.transform import Rotation as R


class GenesisTeleopServer:
    """
    WebSocket-based teleoperation server with trajectory recording

    Supports:
    - Mobile robots (WASD control)
    - Robotic arms (Arrow keys + IK)
    - Drones (WASD + altitude)
    - CSV trajectory recording for imitation learning
    """

    def __init__(self, genesis_service):
        """
        Args:
            genesis_service: Instance of GenesisService managing the simulation
        """
        self.genesis = genesis_service
        self.recording = False
        self.trajectory_data = []
        self.task_name = ""
        self.start_time = 0.0

        # Robot state tracking (for arms using IK)
        self.robot_states = {}

        print("✅ Genesis Teleoperation Server initialized")

    def initialize_robot(self, robot_id: str, robot_type: str):
        """
        Initialize robot-specific state

        Args:
            robot_id: Unique robot identifier
            robot_type: 'mobile', 'arm', or 'drone'
        """
        if robot_type == 'arm':
            # Initialize IK target for arm (from keyboard_teleop.py)
            self.robot_states[robot_id] = {
                'type': 'arm',
                'target_pos': np.array([0.5, 0.0, 0.55]),
                'target_R': R.from_euler("y", np.pi),
            }
        elif robot_type == 'mobile':
            self.robot_states[robot_id] = {
                'type': 'mobile',
                'velocity': np.array([0.0, 0.0, 0.0]),  # vx, vy, omega
            }
        elif robot_type == 'drone':
            self.robot_states[robot_id] = {
                'type': 'drone',
                'base_rpm': 14468,  # From Genesis hover examples
            }

        print(f"🤖 Initialized {robot_type} robot: {robot_id}")

    def process_keyboard_input(self, robot_id: str, pressed_keys: Dict[str, bool]) -> dict:
        """
        Convert WebSocket keyboard state to robot actions

        Args:
            robot_id: Robot to control
            pressed_keys: {'w': True, 'a': False, 's': False, ...}

        Returns:
            dict with 'action', 'state', and 'recording' status
        """
        if robot_id not in self.robot_states:
            return {
                'error': f'Robot {robot_id} not initialized',
                'action': None,
                'state': None
            }

        robot_state = self.robot_states[robot_id]
        robot_type = robot_state['type']

        # Process based on robot type
        if robot_type == 'mobile':
            action, state = self._process_mobile_keys(robot_id, pressed_keys)
        elif robot_type == 'arm':
            action, state = self._process_arm_keys(robot_id, pressed_keys)
        elif robot_type == 'drone':
            action, state = self._process_drone_keys(robot_id, pressed_keys)
        else:
            return {'error': 'Unknown robot type', 'action': None, 'state': None}

        # Record trajectory if active
        if self.recording:
            self._record_step(robot_id, action, state)

        return {
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'state': state,
            'recording': self.recording,
            'steps_recorded': len(self.trajectory_data)
        }

    def _process_mobile_keys(self, robot_id: str, keys: dict) -> tuple:
        """
        Mobile robot control (WASD + QE rotation)
        From Genesis examples pattern
        """
        # Linear velocity
        vx = 1.0 if keys.get('w', False) else (-1.0 if keys.get('s', False) else 0.0)
        vy = 1.0 if keys.get('a', False) else (-1.0 if keys.get('d', False) else 0.0)

        # Angular velocity
        omega = 0.5 if keys.get('q', False) else (-0.5 if keys.get('e', False) else 0.0)

        action = np.array([vx, vy, omega])

        # Update state
        self.robot_states[robot_id]['velocity'] = action

        state = {
            'velocity': action.tolist(),
            'type': 'mobile'
        }

        return action, state

    def _process_arm_keys(self, robot_id: str, keys: dict) -> tuple:
        """
        Robotic arm control with IK (Arrow keys + NM for Z + JK for rotation)
        Adapted from Genesis/examples/keyboard_teleop.py
        """
        robot_state = self.robot_states[robot_id]
        target_pos = robot_state['target_pos'].copy()
        target_R = robot_state['target_R']

        # Position control (small increments like Genesis example)
        dpos = 0.002  # 2mm per key press
        drot = 0.01   # ~0.57 degrees per key press

        # XY movement (arrow keys)
        if keys.get('arrowup', False) or keys.get('up', False):
            target_pos[0] -= dpos  # Move forward (North)
        if keys.get('arrowdown', False) or keys.get('down', False):
            target_pos[0] += dpos  # Move backward (South)
        if keys.get('arrowleft', False) or keys.get('left', False):
            target_pos[1] -= dpos  # Move left (West)
        if keys.get('arrowright', False) or keys.get('right', False):
            target_pos[1] += dpos  # Move right (East)

        # Z movement (N/M keys)
        if keys.get('n', False):
            target_pos[2] += dpos  # Move up
        if keys.get('m', False):
            target_pos[2] -= dpos  # Move down

        # Rotation (J/K keys)
        if keys.get('j', False):
            target_R = R.from_euler("z", drot) * target_R  # Counterclockwise
        if keys.get('k', False):
            target_R = R.from_euler("z", -drot) * target_R  # Clockwise

        # Gripper control
        gripper_closed = keys.get('space', False) or keys.get(' ', False)

        # Update robot state
        self.robot_states[robot_id]['target_pos'] = target_pos
        self.robot_states[robot_id]['target_R'] = target_R

        # Convert to quaternion (scalar-first format for Genesis)
        target_quat = target_R.as_quat(scalar_first=True)

        # Action is target pose + gripper state
        action = np.concatenate([target_pos, target_quat, [1.0 if gripper_closed else 0.0]])

        state = {
            'target_pos': target_pos.tolist(),
            'target_quat': target_quat.tolist(),
            'gripper_closed': gripper_closed,
            'type': 'arm'
        }

        return action, state

    def _process_drone_keys(self, robot_id: str, keys: dict) -> tuple:
        """
        Drone control (WASD for movement, Space/Shift for altitude)
        From Genesis/examples/drone/interactive_drone.py
        """
        robot_state = self.robot_states[robot_id]
        base_rpm = robot_state['base_rpm']

        # Propeller RPMs (4 motors)
        action = np.array([base_rpm] * 4, dtype=np.float32)

        drpm = 200  # RPM differential for control

        # Pitch (W/S)
        if keys.get('w', False):
            action[0] += drpm  # Pitch forward
            action[2] -= drpm
        if keys.get('s', False):
            action[0] -= drpm  # Pitch backward
            action[2] += drpm

        # Roll (A/D)
        if keys.get('a', False):
            action[1] -= drpm  # Roll left
            action[3] += drpm
        if keys.get('d', False):
            action[1] += drpm  # Roll right
            action[3] -= drpm

        # Altitude (Space/Shift)
        if keys.get('space', False) or keys.get(' ', False):
            action += drpm  # Increase altitude
        # Note: Shift key might need special handling in frontend

        # Yaw (Q/E)
        if keys.get('q', False):
            action[0] += drpm / 2
            action[2] += drpm / 2
            action[1] -= drpm / 2
            action[3] -= drpm / 2
        if keys.get('e', False):
            action[0] -= drpm / 2
            action[2] -= drpm / 2
            action[1] += drpm / 2
            action[3] += drpm / 2

        state = {
            'rpm': action.tolist(),
            'type': 'drone'
        }

        return action, state

    def start_recording(self, task_name: str) -> dict:
        """
        Start recording demonstration trajectory

        Args:
            task_name: Name of the task being demonstrated

        Returns:
            Status dict
        """
        self.recording = True
        self.trajectory_data = []
        self.task_name = task_name
        self.start_time = time.time()

        print(f"🔴 Recording started: {task_name}")

        return {
            'status': 'recording_started',
            'task_name': task_name,
            'timestamp': self.start_time
        }

    def stop_recording(self) -> dict:
        """
        Stop recording and save trajectory to CSV
        Format: Compatible with imitation learning (from ipc_arm_cloth.py pattern)

        Returns:
            Status dict with filename and statistics
        """
        if not self.recording:
            return {'status': 'not_recording', 'error': 'No active recording'}

        self.recording = False

        # Create trajectories directory if it doesn't exist
        os.makedirs('trajectories', exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectories/{self.task_name}_{timestamp}.csv"

        # CSV field names (from Genesis ipc_arm_cloth.py pattern)
        fieldnames = [
            "step", "time",
            "pos_x", "pos_y", "pos_z",
            "quat_w", "quat_x", "quat_y", "quat_z",  # scalar-first format
            "action_0", "action_1", "action_2", "action_3",
            "action_4", "action_5", "action_6",
            "gripper_closed"
        ]

        # Save to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trajectory_data)

        num_steps = len(self.trajectory_data)
        duration = time.time() - self.start_time

        print(f"✅ Trajectory saved: {filename}")
        print(f"   Steps: {num_steps}, Duration: {duration:.2f}s")

        return {
            'status': 'recording_saved',
            'filename': filename,
            'num_steps': num_steps,
            'duration': duration,
            'task_name': self.task_name
        }

    def _record_step(self, robot_id: str, action: np.ndarray, state: dict):
        """
        Record one step of demonstration
        CSV format compatible with behavior cloning pipelines
        """
        # Get current time relative to recording start
        elapsed_time = time.time() - self.start_time

        # Extract position and quaternion from state
        if state['type'] == 'arm':
            pos = state['target_pos']
            quat = state['target_quat']  # Already in scalar-first format
            gripper = 1 if state['gripper_closed'] else 0
        elif state['type'] == 'mobile':
            pos = [0, 0, 0]  # Mobile robots don't have end-effector pose
            quat = [1, 0, 0, 0]
            gripper = 0
        elif state['type'] == 'drone':
            pos = [0, 0, 0]
            quat = [1, 0, 0, 0]
            gripper = 0
        else:
            pos = [0, 0, 0]
            quat = [1, 0, 0, 0]
            gripper = 0

        # Ensure action has at least 7 elements (pad with zeros if needed)
        action_padded = np.pad(action, (0, max(0, 7 - len(action))), 'constant')

        # Create step data
        step_data = {
            "step": len(self.trajectory_data),
            "time": elapsed_time,
            "pos_x": pos[0],
            "pos_y": pos[1],
            "pos_z": pos[2],
            "quat_w": quat[0],  # scalar first
            "quat_x": quat[1],
            "quat_y": quat[2],
            "quat_z": quat[3],
            "action_0": action_padded[0],
            "action_1": action_padded[1],
            "action_2": action_padded[2],
            "action_3": action_padded[3],
            "action_4": action_padded[4],
            "action_5": action_padded[5],
            "action_6": action_padded[6],
            "gripper_closed": gripper
        }

        self.trajectory_data.append(step_data)

    def reset_robot(self, robot_id: str):
        """Reset robot to initial state (from keyboard_teleop.py 'U' key)"""
        if robot_id in self.robot_states:
            robot_type = self.robot_states[robot_id]['type']

            if robot_type == 'arm':
                # Reset to initial pose
                self.robot_states[robot_id]['target_pos'] = np.array([0.5, 0.0, 0.55])
                self.robot_states[robot_id]['target_R'] = R.from_euler("y", np.pi)
                print(f"🔄 Reset arm {robot_id} to initial pose")
            elif robot_type == 'mobile':
                self.robot_states[robot_id]['velocity'] = np.array([0.0, 0.0, 0.0])
                print(f"🔄 Reset mobile robot {robot_id} velocity")

            return {'status': 'reset', 'robot_id': robot_id}

        return {'error': f'Robot {robot_id} not found'}

    def get_recording_status(self) -> dict:
        """Get current recording status"""
        return {
            'recording': self.recording,
            'task_name': self.task_name if self.recording else None,
            'num_steps': len(self.trajectory_data) if self.recording else 0,
            'elapsed_time': time.time() - self.start_time if self.recording else 0.0
        }
