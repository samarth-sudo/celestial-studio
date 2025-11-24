#!/usr/bin/env python3
"""
Genesis Python Usage Examples
Shows how to use genesis_service.py directly in Python
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from genesis_service import (
    GenesisSimulation,
    GenesisConfig,
    RobotType,
    BackendType
)
import time


def example_1_basic_franka():
    """Example 1: Load and simulate Franka Panda robot"""
    print("=" * 60)
    print("Example 1: Basic Franka Panda Simulation")
    print("=" * 60)

    # Configure Genesis
    config = GenesisConfig(
        backend=BackendType.METAL,  # Auto-detects best backend
        fps=60,
        show_viewer=False,  # Headless mode
        render_width=1280,
        render_height=720,
    )

    # Create simulation
    sim = GenesisSimulation(config)
    sim.initialize()

    # Add Franka Panda robot
    robot = sim.add_robot(
        robot_id="franka1",
        robot_type=RobotType.FRANKA,
        position=(0, 0, 0.5)
    )

    # Build and start
    sim.build_scene()
    sim.start()

    # Simulate for 100 steps
    for i in range(100):
        state = sim.step()
        if i % 20 == 0:
            print(f"Step {i}: FPS={state['fps']:.1f}, Robots={len(state['robots'])}")

    print(f"✅ Simulated {sim.step_count} steps\n")


def example_2_multi_robot():
    """Example 2: Multiple robots (Franka + Go2 + Drone)"""
    print("=" * 60)
    print("Example 2: Multi-Robot Simulation")
    print("=" * 60)

    config = GenesisConfig(
        backend=BackendType.METAL,
        fps=60,
        show_viewer=False,
    )

    sim = GenesisSimulation(config)
    sim.initialize()

    # Add multiple robots
    print("Adding robots...")
    sim.add_robot("franka", RobotType.FRANKA, position=(0, 0, 0.5))
    sim.add_robot("go2", RobotType.GO2, position=(2, 0, 0.3))
    sim.add_robot("drone", RobotType.DRONE, position=(0, 2, 1.5))

    # Add obstacles
    print("Adding obstacles...")
    sim.add_obstacle("box1", position=(1, 1, 0.25), size=(0.5, 0.5, 0.5))
    sim.add_obstacle("box2", position=(-1, 1, 0.25), size=(0.5, 0.5, 0.5))

    # Build and simulate
    sim.build_scene()
    sim.start()

    for i in range(50):
        state = sim.step()

    print(f"✅ Simulated {len(sim.robots)} robots for {sim.step_count} steps\n")


def example_3_discover_models():
    """Example 3: Discover available robot models"""
    print("=" * 60)
    print("Example 3: Discover Available Robot Models")
    print("=" * 60)

    models = GenesisSimulation.discover_available_models()

    print(f"\n📦 URDF Models ({len(models['urdf'])} total):")
    print("-" * 60)
    urdf_by_type = {}
    for model in models['urdf']:
        name = model['name']
        if name not in urdf_by_type:
            urdf_by_type[name] = []
        urdf_by_type[name].append(model['path'])

    for robot_name, paths in urdf_by_type.items():
        print(f"\n  🤖 {robot_name}:")
        for path in paths:
            print(f"     • {path}")

    print(f"\n\n📦 MJCF/XML Models ({len(models['xml'])} total):")
    print("-" * 60)
    xml_by_type = {}
    for model in models['xml']:
        name = model['name']
        if name not in xml_by_type:
            xml_by_type[name] = []
        xml_by_type[name].append(model['path'])

    for robot_name, paths in xml_by_type.items():
        print(f"\n  🤖 {robot_name}:")
        for path in paths:
            print(f"     • {path}")

    print("\n")


def example_4_with_algorithm():
    """Example 4: Robot with custom algorithm"""
    print("=" * 60)
    print("Example 4: Robot with Custom Algorithm")
    print("=" * 60)

    config = GenesisConfig(
        backend=BackendType.METAL,
        fps=60,
        show_viewer=False,
    )

    sim = GenesisSimulation(config)
    sim.initialize()

    # Add mobile robot
    sim.add_robot("mobile1", RobotType.MOBILE_ROBOT, position=(0, 0, 0.2))

    # Define simple algorithm
    def simple_controller(robot_state):
        """Simple algorithm that returns target position"""
        # Move in a circle
        t = robot_state.get('time', 0)
        import math
        x = math.cos(t) * 2
        y = math.sin(t) * 2
        return {'target_position': [x, y, 0.2]}

    # Set algorithm
    sim.set_algorithm("mobile1", simple_controller)

    sim.build_scene()
    sim.start()

    # Simulate
    for i in range(50):
        state = sim.step()

    print(f"✅ Simulated robot with algorithm for {sim.step_count} steps\n")


def example_5_frame_capture():
    """Example 5: Capture and save frame"""
    print("=" * 60)
    print("Example 5: Frame Capture (would save to file)")
    print("=" * 60)

    config = GenesisConfig(
        backend=BackendType.METAL,
        fps=60,
        show_viewer=False,
        render_width=1920,
        render_height=1080,
    )

    sim = GenesisSimulation(config)
    sim.initialize()

    # Add robot
    sim.add_robot("franka", RobotType.FRANKA, position=(0, 0, 0.5))
    sim.build_scene()
    sim.start()

    # Simulate a bit
    for i in range(10):
        sim.step()

    # Get frame
    frame = sim.last_frame
    if frame is not None:
        print(f"✅ Captured frame: shape={frame.shape}, dtype={frame.dtype}")
        print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]} pixels")
        # In real use: save with PIL or cv2
        # Image.fromarray(frame).save('output.jpg')
    else:
        print("⚠️  No frame captured (viewer disabled in headless mode)")

    print()


if __name__ == "__main__":
    print("\n")
    print("🚀" * 30)
    print("Genesis Python Usage Examples")
    print("🚀" * 30)
    print("\n")

    # Run all examples
    try:
        example_3_discover_models()
        example_1_basic_franka()
        example_2_multi_robot()
        example_4_with_algorithm()
        example_5_frame_capture()

        print("=" * 60)
        print("✅ All Examples Completed Successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
