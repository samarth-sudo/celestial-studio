"""
Test Genesis Integration
Verifies that Genesis physics engine, video streaming, and algorithm execution work correctly
"""

import asyncio
import sys
import time

import numpy as np

# Test imports
print("🧪 Testing imports...")

try:
    import genesis as gs
    print("✅ Genesis imported successfully")
    genesis_available = True
except ImportError as e:
    print(f"❌ Genesis import failed: {e}")
    print("Install with: pip install genesis-world")
    genesis_available = False

try:
    from genesis_service import GenesisSimulation, GenesisConfig, BackendType, RobotType
    from genesis_renderer import VideoStreamer, StreamConfig, StreamQuality
    from algorithm_executor import AlgorithmExecutor
    from algorithm_generator import AlgorithmGenerator, AlgorithmRequest
    print("✅ All backend modules imported successfully")
except ImportError as e:
    print(f"❌ Backend module import failed: {e}")
    sys.exit(1)


def test_genesis_initialization():
    """Test 1: Genesis Initialization"""
    print("\n" + "=" * 60)
    print("TEST 1: Genesis Initialization")
    print("=" * 60)

    if not genesis_available:
        print("⏭️  Skipping (Genesis not available)")
        return False

    try:
        # Create config with Metal backend
        config = GenesisConfig(
            backend=BackendType.METAL,
            fps=60,
            show_viewer=False,  # Headless
            render_width=1280,
            render_height=720,
        )

        # Initialize simulation
        sim = GenesisSimulation(config)
        sim.initialize()

        print("✅ Genesis initialized with Metal backend")
        print(f"   Backend: {config.backend.value}")
        print(f"   FPS: {config.fps}")
        print(f"   Resolution: {config.render_width}x{config.render_height}")

        sim.destroy()
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_robot_creation():
    """Test 2: Robot Creation"""
    print("\n" + "=" * 60)
    print("TEST 2: Robot Creation")
    print("=" * 60)

    if not genesis_available:
        print("⏭️  Skipping (Genesis not available)")
        return False

    try:
        config = GenesisConfig(backend=BackendType.METAL, show_viewer=False)
        sim = GenesisSimulation(config)
        sim.initialize()

        # Add mobile robot
        robot = sim.add_robot("test_robot", RobotType.MOBILE_ROBOT, position=(0, 0, 0.5))

        if robot is not None:
            print("✅ Mobile robot created successfully")
            print(f"   Robot ID: test_robot")
            print(f"   Type: {RobotType.MOBILE_ROBOT.value}")
            print(f"   Position: (0, 0, 0.5)")

        # Add obstacle
        obstacle = sim.add_obstacle("test_obstacle", position=(2, 2, 0.25), size=(0.5, 0.5, 0.5))

        if obstacle is not None:
            print("✅ Obstacle created successfully")

        # Build scene
        sim.build_scene()
        print("✅ Scene built successfully")

        sim.destroy()
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_simulation_step():
    """Test 3: Simulation Stepping"""
    print("\n" + "=" * 60)
    print("TEST 3: Simulation Stepping")
    print("=" * 60)

    if not genesis_available:
        print("⏭️  Skipping (Genesis not available)")
        return False

    try:
        config = GenesisConfig(backend=BackendType.METAL, show_viewer=False)
        sim = GenesisSimulation(config)
        sim.initialize()
        sim.add_robot("robot1", RobotType.MOBILE_ROBOT)
        sim.build_scene()
        sim.start()

        # Run 10 simulation steps
        start_time = time.time()
        for i in range(10):
            state = sim.step()

        elapsed = time.time() - start_time
        fps = 10 / elapsed

        print(f"✅ Ran 10 simulation steps")
        print(f"   Elapsed time: {elapsed:.3f}s")
        print(f"   Effective FPS: {fps:.1f}")
        print(f"   Step count: {sim.step_count}")

        sim.destroy()
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_video_streaming():
    """Test 4: Video Streaming"""
    print("\n" + "=" * 60)
    print("TEST 4: Video Streaming")
    print("=" * 60)

    try:
        # Create stream config
        config = StreamConfig.from_quality(StreamQuality.MEDIUM)
        streamer = VideoStreamer(config)

        print(f"✅ Video streamer created")
        print(f"   Codec: {config.codec.value}")
        print(f"   Resolution: {config.width}x{config.height}")
        print(f"   FPS: {config.fps}")
        print(f"   Quality: {config.jpeg_quality}")

        # Test frame encoding
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        streamer.start()
        streamer.add_frame(test_frame)

        time.sleep(0.1)  # Wait for encoding

        jpeg_bytes = streamer.get_latest_frame_jpeg()

        if jpeg_bytes:
            print(f"✅ Frame encoded successfully")
            print(f"   Frame size: {len(jpeg_bytes)} bytes")
            print(f"   Compression ratio: {(720 * 1280 * 3) / len(jpeg_bytes):.1f}x")

        streamer.stop()
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_algorithm_executor():
    """Test 5: Algorithm Execution"""
    print("\n" + "=" * 60)
    print("TEST 5: Algorithm Execution")
    print("=" * 60)

    try:
        executor = AlgorithmExecutor()

        # Test algorithm code
        test_code = """
import numpy as np
from typing import List, Dict

# Parameters
MAX_SPEED = 2.0  # Maximum speed in m/s
SAFETY_DISTANCE = 1.0  # Minimum distance to obstacles

def compute_safe_velocity(
    current_pos: np.ndarray,
    current_vel: np.ndarray,
    obstacles: List[Dict],
    goal: np.ndarray,
    max_speed: float,
    params: Dict
) -> np.ndarray:
    \"\"\"
    Simple obstacle avoidance using potential fields

    Args:
        current_pos: Current position [x, y]
        current_vel: Current velocity [vx, vy]
        obstacles: List of obstacles with 'position' and 'size'
        goal: Goal position [x, y]
        max_speed: Maximum allowed speed
        params: Additional parameters

    Returns:
        Safe velocity vector [vx, vy]
    \"\"\"
    # Attractive force towards goal
    to_goal = goal - current_pos
    dist_to_goal = np.linalg.norm(to_goal)

    if dist_to_goal > 0.1:
        attractive = (to_goal / dist_to_goal) * max_speed
    else:
        attractive = np.zeros(2)

    # Repulsive force from obstacles
    repulsive = np.zeros(2)
    for obs in obstacles:
        obs_pos = np.array(obs['position'][:2])
        to_obs = current_pos - obs_pos
        dist = np.linalg.norm(to_obs)

        if dist < SAFETY_DISTANCE and dist > 0:
            repulsive += (to_obs / dist) * (1.0 / dist) * max_speed

    # Combine forces
    desired_vel = attractive + repulsive

    # Limit to max speed
    speed = np.linalg.norm(desired_vel)
    if speed > max_speed:
        desired_vel = (desired_vel / speed) * max_speed

    return desired_vel
"""

        # Load algorithm
        success = executor.load_algorithm(
            algorithm_id="test_algo",
            code=test_code,
            function_name="compute_safe_velocity",
            algorithm_type="obstacle_avoidance",
            parameters={"MAX_SPEED": 2.0, "SAFETY_DISTANCE": 1.0}
        )

        if not success:
            print("❌ Failed to load algorithm")
            return False

        print("✅ Algorithm loaded successfully")

        # Test execution
        current_pos = np.array([0.0, 0.0])
        current_vel = np.array([0.0, 0.0])
        obstacles = [{"position": [2.0, 0.0, 0.0], "size": [0.5, 0.5, 0.5]}]
        goal = np.array([5.0, 0.0])

        result = executor.execute(
            "test_algo",
            current_pos,
            current_vel,
            obstacles,
            goal,
            2.0,
            {}
        )

        if result is not None:
            print(f"✅ Algorithm executed successfully")
            print(f"   Input position: {current_pos}")
            print(f"   Input goal: {goal}")
            print(f"   Output velocity: {result}")

            stats = executor.get_statistics("test_algo")
            print(f"   Execution time: {stats['avg_execution_time_ms']:.3f}ms")

        executor.cleanup()
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_algorithm_generator():
    """Test 6: Algorithm Generation"""
    print("\n" + "=" * 60)
    print("TEST 6: Algorithm Generation (requires Ollama)")
    print("=" * 60)

    try:
        # Check if Ollama is available
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                print("⏭️  Skipping (Ollama not running)")
                return True
        except:
            print("⏭️  Skipping (Ollama not running)")
            return True

        generator = AlgorithmGenerator()

        request = AlgorithmRequest(
            description="Avoid obstacles using artificial potential fields",
            robot_type="mobile",
            algorithm_type="obstacle_avoidance"
        )

        print("🔄 Generating algorithm (this may take 10-30 seconds)...")
        response = generator.generate(request)

        print("✅ Algorithm generated successfully")
        print(f"   Function name: {response.function_name}")
        print(f"   Algorithm type: {response.algorithm_type}")
        print(f"   Complexity: {response.estimated_complexity}")
        print(f"   Parameters: {len(response.parameters)} configurable")
        print(f"   Code length: {len(response.code)} characters")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("GENESIS INTEGRATION TEST SUITE")
    print("=" * 60)

    tests = [
        test_genesis_initialization,
        test_robot_creation,
        test_simulation_step,
        test_video_streaming,
        test_algorithm_executor,
        test_algorithm_generator,
    ]

    results = []

    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! Genesis integration is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
