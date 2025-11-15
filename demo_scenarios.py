"""
Demo Scenarios for Multi-Robot System

Demonstrates all 6 robot types with their compatible tasks.
Run this script to see examples of each robot in action.

Usage:
    python demo_scenarios.py --robot manipulator
    python demo_scenarios.py --robot mobile
    python demo_scenarios.py --robot quadruped
    python demo_scenarios.py --robot humanoid
    python demo_scenarios.py --robot aerial
    python demo_scenarios.py --all
"""

import argparse
import numpy as np
from typing import Dict, Any

# Import all necessary modules
from backend.assets.robot_configs import (
    FRANKA_PANDA_CONFIG,
    SIMPLE_MOBILE_CONFIG,
    ANYMAL_C_CONFIG,
    SIMPLE_HUMANOID_CONFIG,
    SIMPLE_QUADCOPTER_CONFIG,
)
from backend.simulation.pybullet_interface import PyBulletRobotInterface
from backend.controllers import (
    DifferentialIKController,
    MobileController,
    QuadrupedController,
    DroneController,
    GaitType
)
from backend.tasks import (
    ReachTask,
    LiftTask,
    NavigationTask,
    WalkingTask,
    FlightTask
)


# ============================================================================
# Demo Scenarios
# ============================================================================

def demo_manipulator(use_gui: bool = True):
    """
    Demo: Franka Panda Manipulator - Reach Task

    Task: Move end-effector to target position
    Controller: Differential IK
    """
    print("\n" + "=" * 60)
    print("🦾 DEMO: Manipulator - Reach Task")
    print("=" * 60)
    print(f"Robot: Franka Panda")
    print(f"Task: Reach target position [0.5, 0.0, 0.3]")
    print(f"Controller: Differential IK\n")

    # Create robot
    robot = PyBulletRobotInterface(FRANKA_PANDA_CONFIG, use_gui=use_gui)

    # Create task
    target_pos = np.array([0.5, 0.0, 0.3])
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    task = ReachTask(robot, target_pos=target_pos, target_quat=target_quat)

    # Create controller
    controller = DifferentialIKController()

    # Reset task
    obs = task.reset()
    print(f"Starting distance to target: {obs['distance_to_target']:.3f}m\n")

    # Run simulation
    done = False
    total_reward = 0.0

    while not done and task.step_count < 1000:
        # Get current state
        current_pos = obs["ee_pos"]
        current_quat = obs["ee_quat"]

        # Compute control
        joint_targets = controller.compute_joint_targets(
            robot, target_pos, target_quat
        )

        # Step task
        obs, reward, done, info = task.step(joint_targets)
        total_reward += reward

        # Print progress every 100 steps
        if task.step_count % 100 == 0:
            dist = obs['distance_to_target']
            print(f"  Step {task.step_count:4d} | Distance: {dist:.4f}m | Reward: {reward:.2f}")

    # Results
    print(f"\n{'='*60}")
    print(f"Episode Results:")
    print(f"  Success: {info['success']}")
    print(f"  Total steps: {task.step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final distance: {obs['distance_to_target']:.4f}m")
    print(f"{'='*60}\n")

    robot.disconnect()


def demo_mobile(use_gui: bool = True):
    """
    Demo: Mobile Robot - Navigation Task

    Task: Navigate through waypoints
    Controller: Differential drive
    """
    print("\n" + "=" * 60)
    print("🤖 DEMO: Mobile Robot - Navigation Task")
    print("=" * 60)
    print(f"Robot: Simple Mobile Robot")
    print(f"Task: Navigate waypoints [[1,0], [1,1], [0,1]]")
    print(f"Controller: Differential Drive\n")

    # Create robot
    robot = PyBulletRobotInterface(SIMPLE_MOBILE_CONFIG, use_gui=use_gui)

    # Create task
    waypoints = [
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 1.0]),
    ]
    task = NavigationTask(robot, waypoints=waypoints)

    # Create controller
    controller = MobileController()

    # Reset task
    obs = task.reset()
    print(f"Total waypoints: {len(waypoints)}\n")

    # Run simulation
    done = False
    total_reward = 0.0

    while not done and task.step_count < 2000:
        # Get current state
        current_pos = obs["position"]
        current_yaw = obs["yaw"]
        target_wp = obs["target_waypoint"]

        # Compute control
        linear_vel, angular_vel = controller.compute_velocity_to_goal(
            current_pos, current_yaw, target_wp
        )
        left_vel, right_vel = controller.compute_wheel_velocities(
            linear_vel, angular_vel
        )

        # Step task
        obs, reward, done, info = task.step(np.array([left_vel, right_vel]))
        total_reward += reward

        # Print progress
        if task.step_count % 200 == 0:
            dist = obs['distance_to_waypoint']
            wp_idx = obs['waypoint_index']
            print(f"  Step {task.step_count:4d} | WP {wp_idx}/{len(waypoints)} | Distance: {dist:.3f}m")

    # Results
    print(f"\n{'='*60}")
    print(f"Episode Results:")
    print(f"  Success: {info['success']}")
    print(f"  Waypoints reached: {info['waypoints_reached']}/{len(waypoints)}")
    print(f"  Total steps: {task.step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"{'='*60}\n")

    robot.disconnect()


def demo_quadruped(use_gui: bool = True):
    """
    Demo: Quadruped - Walking Task

    Task: Walk forward 5 meters
    Controller: Gait generation (Trot)
    """
    print("\n" + "=" * 60)
    print("🐕 DEMO: Quadruped - Walking Task")
    print("=" * 60)
    print(f"Robot: ANYmal C")
    print(f"Task: Walk forward 5 meters")
    print(f"Controller: Gait Generator (Trot)\n")

    # Create robot
    robot = PyBulletRobotInterface(ANYMAL_C_CONFIG, use_gui=use_gui)

    # Create task
    from backend.tasks.walking_task import WalkingTaskConfig
    config = WalkingTaskConfig(target_distance=5.0, target_velocity=0.5)
    task = WalkingTask(robot, config=config)

    # Create controller
    controller = QuadrupedController()

    # Reset task
    obs = task.reset()
    print(f"Target distance: {config.target_distance}m\n")

    # Run simulation
    done = False
    total_reward = 0.0
    time_elapsed = 0.0
    dt = 1/240.0

    while not done and task.step_count < 3000:
        # Generate gait
        foot_positions = controller.compute_gait(
            gait_type=GaitType.TROT,
            forward_velocity=config.target_velocity,
            time=time_elapsed
        )

        # Convert to joint angles
        joint_angles = controller.compute_joint_angles(foot_positions)

        # Flatten joint angles (leg order: LF, RF, LH, RH)
        all_angles = np.concatenate([
            joint_angles["LF"], joint_angles["RF"],
            joint_angles["LH"], joint_angles["RH"]
        ])

        # Step task
        obs, reward, done, info = task.step(all_angles)
        total_reward += reward
        time_elapsed += dt

        # Print progress
        if task.step_count % 300 == 0:
            dist = info['distance_traveled']
            height = obs['height']
            tilt = np.degrees(np.sqrt(obs['roll']**2 + obs['pitch']**2))
            print(f"  Step {task.step_count:4d} | Distance: {dist:.2f}m | Height: {height:.2f}m | Tilt: {tilt:.1f}°")

    # Results
    print(f"\n{'='*60}")
    print(f"Episode Results:")
    print(f"  Success: {info['success']}")
    print(f"  Distance traveled: {info['distance_traveled']:.2f}m")
    print(f"  Total steps: {task.step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"{'='*60}\n")

    robot.disconnect()


def demo_aerial(use_gui: bool = True):
    """
    Demo: Drone - Flight Task

    Task: Fly through 3D waypoints
    Controller: Cascade PID
    """
    print("\n" + "=" * 60)
    print("🚁 DEMO: Drone - Flight Task")
    print("=" * 60)
    print(f"Robot: Simple Quadcopter")
    print(f"Task: Fly through 3D waypoints")
    print(f"Controller: Cascade PID\n")

    # Create robot
    robot = PyBulletRobotInterface(SIMPLE_QUADCOPTER_CONFIG, use_gui=use_gui)

    # Create task
    waypoints = [
        np.array([1.0, 0.0, 2.0]),
        np.array([1.0, 1.0, 2.5]),
        np.array([0.0, 1.0, 2.0]),
    ]
    task = FlightTask(robot, waypoints=waypoints)

    # Create controller
    controller = DroneController(mass=1.5)

    # Reset task
    obs = task.reset()
    print(f"Total waypoints: {len(waypoints)}\n")

    # Run simulation
    done = False
    total_reward = 0.0

    while not done and task.step_count < 2000:
        # Get current state
        current_pos = obs["position"]
        current_quat = obs["orientation"]
        current_vel = obs["linear_velocity"]
        target_wp = obs["target_waypoint"]

        # Compute control
        thrust, moments = controller.compute_hover(
            current_pos, target_wp, current_vel, current_quat
        )

        # Step task (simplified - actual would apply thrust/moments)
        obs, reward, done, info = task.step()
        total_reward += reward

        # Print progress
        if task.step_count % 200 == 0:
            dist = obs['distance_to_waypoint']
            alt = obs['altitude']
            wp_idx = obs['waypoint_index']
            print(f"  Step {task.step_count:4d} | WP {wp_idx}/{len(waypoints)} | Distance: {dist:.3f}m | Alt: {alt:.2f}m")

    # Results
    print(f"\n{'='*60}")
    print(f"Episode Results:")
    print(f"  Success: {info['success']}")
    print(f"  Waypoints reached: {info['waypoints_reached']}/{len(waypoints)}")
    print(f"  Total steps: {task.step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"{'='*60}\n")

    robot.disconnect()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Robot Demo Scenarios')
    parser.add_argument(
        '--robot',
        choices=['manipulator', 'mobile', 'quadruped', 'humanoid', 'aerial'],
        help='Robot type to demo'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all demos'
    )
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run without GUI'
    )

    args = parser.parse_args()
    use_gui = not args.no_gui

    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  MULTI-ROBOT SYSTEM - DEMO SCENARIOS".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    if args.all:
        # Run all demos
        demos = [
            ('manipulator', demo_manipulator),
            ('mobile', demo_mobile),
            ('quadruped', demo_quadruped),
            ('aerial', demo_aerial),
        ]

        for robot_type, demo_fn in demos:
            try:
                demo_fn(use_gui=use_gui)
                input("\nPress Enter to continue to next demo...")
            except KeyboardInterrupt:
                print("\n\nDemo interrupted by user.")
                break
            except Exception as e:
                print(f"\n❌ Demo failed: {e}")
                continue

    elif args.robot:
        # Run specific demo
        demo_map = {
            'manipulator': demo_manipulator,
            'mobile': demo_mobile,
            'quadruped': demo_quadruped,
            'aerial': demo_aerial,
        }

        try:
            demo_map[args.robot](use_gui=use_gui)
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python demo_scenarios.py --robot manipulator")
        print("  python demo_scenarios.py --all")
        print("  python demo_scenarios.py --robot quadruped --no-gui")

    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    main()
