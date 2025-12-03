"""
Test script for Genesis robot loading
Tests all 13 robot types to verify they load correctly
"""

import sys
sys.path.append('/Users/samarth/Desktop/idea/backend')

from genesis_service import GenesisSimulation, GenesisConfig, RobotType, BackendType
import numpy as np

def test_robot_loading():
    """Test loading each robot type"""

    print("=" * 60)
    print("GENESIS ROBOT LOADING TEST")
    print("=" * 60)

    # Configure Genesis
    config = GenesisConfig(
        backend=BackendType.METAL,  # Use Metal for Apple Silicon
        show_viewer=False,  # Offscreen rendering
        fps=60
    )

    # Test each robot type
    test_robots = [
        ("Franka Panda", RobotType.FRANKA),
        ("Unitree Go2", RobotType.GO2),
        ("ANYmal C", RobotType.ANYMAL),
        ("KUKA iiwa", RobotType.KUKA),
        ("UR5e", RobotType.UR5E),
        ("Shadow Hand", RobotType.SHADOW_HAND),
        ("Crazyflie", RobotType.CRAZYFLIE),
        ("Ant", RobotType.ANT),
        ("Humanoid", RobotType.HUMANOID),
        ("Mobile Robot (primitive)", RobotType.MOBILE_ROBOT),
        ("Robotic Arm (primitive)", RobotType.ROBOTIC_ARM),
        ("Drone (primitive)", RobotType.DRONE),
    ]

    results = {
        "passed": [],
        "failed": []
    }

    for robot_name, robot_type in test_robots:
        print(f"\n{'─' * 60}")
        print(f"Testing: {robot_name} ({robot_type.value})")
        print(f"{'─' * 60}")

        try:
            # Create new simulation for each robot
            sim = GenesisSimulation(config)
            sim.initialize()

            # Try to add robot
            robot = sim.add_robot(
                robot_id=f"test_{robot_type.value}",
                robot_type=robot_type,
                position=(0, 0, 0.5)
            )

            # Build scene
            sim.build_scene()

            # Run a few steps
            sim.start()
            for _ in range(10):
                sim.step()
            sim.stop()

            # Check if robot was added
            if robot is not None and f"test_{robot_type.value}" in sim.robots:
                print(f"✅ SUCCESS: {robot_name} loaded and simulated correctly")
                results["passed"].append(robot_name)
            else:
                print(f"❌ FAILED: {robot_name} - robot not found in simulation")
                results["failed"].append(robot_name)

        except Exception as e:
            print(f"❌ FAILED: {robot_name}")
            print(f"   Error: {str(e)}")
            results["failed"].append(robot_name)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(test_robots)}")
    print(f"✅ Passed: {len(results['passed'])}")
    print(f"❌ Failed: {len(results['failed'])}")

    if results["passed"]:
        print(f"\nPassed Robots:")
        for robot in results["passed"]:
            print(f"  ✅ {robot}")

    if results["failed"]:
        print(f"\nFailed Robots:")
        for robot in results["failed"]:
            print(f"  ❌ {robot}")

    print("\n" + "=" * 60)

    # Return success if all passed
    return len(results["failed"]) == 0


if __name__ == "__main__":
    try:
        success = test_robot_loading()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
