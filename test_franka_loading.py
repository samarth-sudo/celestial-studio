#!/usr/bin/env python3
"""
Test script to verify Franka Panda model loading from local assets
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from genesis_service import GenesisSimulation, GenesisConfig, RobotType, BackendType
import genesis as gs


def test_franka_loading():
    """Test loading Franka Panda robot"""

    print("=" * 60)
    print("🧪 Testing Franka Panda Model Loading")
    print("=" * 60)
    print()

    # Configure Genesis
    config = GenesisConfig(
        backend=BackendType.METAL,  # Will auto-detect
        fps=60,
        show_viewer=False,  # Headless
        render_width=1280,
        render_height=720,
    )

    # Create simulation
    print("📦 Creating simulation instance...")
    sim = GenesisSimulation(config)

    # Initialize
    print("🚀 Initializing Genesis...")
    sim.initialize()
    print(f"✅ Genesis initialized with {sim.config.backend.value} backend")
    print()

    # Add Franka Panda robot
    print("🤖 Adding Franka Panda robot...")
    try:
        robot = sim.add_robot(
            robot_id="franka_test",
            robot_type=RobotType.FRANKA,
            position=(0, 0, 0.5)
        )

        if robot is not None:
            print("✅ Franka Panda loaded successfully!")
            print(f"   Robot ID: franka_test")
            print(f"   Entity: {robot}")
        else:
            print("❌ Failed to load Franka Panda")
            return False
    except Exception as e:
        print(f"❌ Error loading Franka: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Build scene
    print("🏗️  Building scene...")
    try:
        sim.build_scene()
        print("✅ Scene built successfully!")
    except Exception as e:
        print(f"❌ Error building scene: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Start simulation
    print("▶️  Starting simulation...")
    sim.start()
    print("✅ Simulation started")
    print()

    # Step simulation a few times
    print("⚙️  Running simulation steps...")
    try:
        for i in range(10):
            state = sim.step()
            if i == 0:
                print(f"   Step {i}: {state.get('timestamp', 0):.3f}s")
            elif i == 9:
                print(f"   Step {i}: {state.get('timestamp', 0):.3f}s")

        print(f"✅ Simulation stepped {sim.step_count} times")
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 60)
    print("✅ ALL TESTS PASSED - Franka Panda loaded successfully!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_franka_loading()
    sys.exit(0 if success else 1)
