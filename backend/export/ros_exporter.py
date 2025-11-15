"""
ROS Package Exporter

Generates ROS workspace with Python nodes for real robot deployment:
- Path planning nodes
- Obstacle avoidance nodes
- Vision processing nodes
- Launch files
- URDF robot descriptions
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class ROSExporter:
    """Exports simulation as ROS package"""

    def export(
        self,
        algorithms: List[Dict[str, Any]],
        scene_config: Dict[str, Any],
        robots: List[Dict[str, Any]],
        project_name: str
    ) -> Dict[str, str]:
        """
        Generate all files for ROS workspace

        Returns:
            Dictionary of {filepath: content}
        """
        files = {}
        package_name = project_name.replace('-', '_')

        # Workspace README
        files['README.md'] = self._generate_readme(project_name, algorithms, robots)

        # Package configuration
        files[f'src/{package_name}/package.xml'] = self._generate_package_xml(package_name)
        files[f'src/{package_name}/CMakeLists.txt'] = self._generate_cmakelists(package_name)

        # Python nodes for each algorithm
        for i, algo in enumerate(algorithms):
            node_name = f"{algo.get('type', 'algorithm')}_{i+1}_node.py"
            files[f'src/{package_name}/scripts/{node_name}'] = self._generate_algorithm_node(algo)

        # Launch file
        files[f'src/{package_name}/launch/simulation.launch'] = self._generate_launch_file(package_name, algorithms)

        # Configuration files
        files[f'src/{package_name}/config/params.yaml'] = self._generate_params_yaml(scene_config)
        files[f'src/{package_name}/config/scene.yaml'] = json.dumps(scene_config, indent=2)

        # URDF robot descriptions
        for robot in robots:
            robot_name = robot.get('type', 'robot')
            files[f'src/{package_name}/urdf/{robot_name}.urdf'] = self._generate_urdf(robot)

        # Setup script
        files['setup.sh'] = self._generate_setup_script()

        return files

    def _generate_readme(self, project_name: str, algorithms: List[Dict], robots: List[Dict]) -> str:
        """Generate ROS package README"""
        algo_list = "\\n".join([f"- {algo.get('name', 'Algorithm')} ({algo.get('type', 'unknown')})" for algo in algorithms])
        robot_list = "\\n".join([f"- {robot.get('type', 'robot').capitalize()}" for robot in robots])

        return f"""# {project_name} - ROS Package

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Celestial Studio

## Overview

This ROS workspace contains algorithm implementations and robot configurations
for autonomous navigation and manipulation tasks.

## Algorithms Included
{algo_list}

## Robots
{robot_list}

## Prerequisites

- ROS Noetic (Ubuntu 20.04) or ROS Melodic (Ubuntu 18.04)
- Python 3.8+
- catkin tools

## Installation

```bash
# Source ROS
source /opt/ros/noetic/setup.bash

# Build workspace
cd ros_simulation_ws
catkin_make

# Source workspace
source devel/setup.bash
```

## Usage

### Launch Simulation

```bash
roslaunch {project_name.replace('-', '_')} simulation.launch
```

### Run Individual Nodes

```bash
# Path planning node
rosrun {project_name.replace('-', '_')} path_planning_1_node.py

# Obstacle avoidance node
rosrun {project_name.replace('-', '_')} obstacle_avoidance_1_node.py
```

## Topics

### Published Topics
- `/path` (nav_msgs/Path) - Planned path
- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands
- `/obstacles` (sensor_msgs/PointCloud2) - Detected obstacles

### Subscribed Topics
- `/odom` (nav_msgs/Odometry) - Robot odometry
- `/scan` (sensor_msgs/LaserScan) - Laser scan data
- `/camera/rgb/image_raw` (sensor_msgs/Image) - Camera feed

## Parameters

Edit `config/params.yaml` to adjust:
- Algorithm parameters
- Robot configuration
- Sensor settings

## Troubleshooting

### Can't find package
```bash
source devel/setup.bash
rospack profile
```

### Build errors
```bash
cd ros_simulation_ws
catkin_make clean
catkin_make
```

## Documentation

- [ROS Wiki](http://wiki.ros.org)
- [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials)
- [Celestial Studio](https://github.com/samarth-sudo/celestial-studio)
"""

    def _generate_package_xml(self, package_name: str) -> str:
        """Generate package.xml"""
        return f"""<?xml version="1.0"?>
<package format="2">
  <name>{package_name}</name>
  <version>1.0.0</version>
  <description>Celestial Studio generated ROS package</description>

  <maintainer email="user@example.com">User</maintainer>
  <license>MIT</license>

  <buildtool_depend>catkin</buildtool_depend>

  <depend>rospy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>tf</depend>

  <export></export>
</package>
"""

    def _generate_cmakelists(self, package_name: str) -> str:
        """Generate CMakeLists.txt"""
        return f"""cmake_minimum_required(VERSION 3.0.2)
project({package_name})

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf
)

catkin_package()

include_directories(
  ${{catkin_INCLUDE_DIRS}}
)

catkin_install_python(PROGRAMS
  scripts/path_planning_1_node.py
  scripts/obstacle_avoidance_1_node.py
  DESTINATION ${{CATKIN_PACKAGE_BIN_DESTINATION}}
)
"""

    def _generate_algorithm_node(self, algo: Dict) -> str:
        """Generate ROS node from algorithm"""
        algo_type = algo.get('type', 'algorithm')
        algo_name = algo.get('name', 'Algorithm')

        # Convert TypeScript to Python (simplified conversion)
        code_comment = f"# Original TypeScript algorithm:\\n# {algo.get('description', 'No description')}"

        return f"""#!/usr/bin/env python3
\"\"\"
{algo_name} ROS Node
Generated by Celestial Studio

{code_comment}
\"\"\"

import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan

class {algo_type.title().replace('_', '')}Node:
    def __init__(self):
        rospy.init_node('{algo_type}_node', anonymous=True)

        # Parameters
        self.max_speed = rospy.get_param('~max_speed', 0.5)
        self.update_rate = rospy.get_param('~update_rate', 10)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.path_pub = rospy.Publisher('/path', Path, queue_size=10)

        # Subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # State variables
        self.current_pose = None
        self.obstacles = []

        rospy.loginfo(f"{algo_name} node started")

    def odom_callback(self, msg):
        \"\"\"Update current robot pose\"\"\"
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        \"\"\"Process laser scan data\"\"\"
        self.obstacles = self.process_scan(msg)

    def process_scan(self, scan):
        \"\"\"Convert scan to obstacle list\"\"\"
        obstacles = []
        for i, distance in enumerate(scan.ranges):
            if distance < scan.range_max:
                angle = scan.angle_min + i * scan.angle_increment
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)
                obstacles.append((x, y))
        return obstacles

    def compute_velocity(self):
        \"\"\"
        Algorithm implementation

        TODO: Implement algorithm logic here based on:
        {algo.get('description', 'No description')}
        \"\"\"
        cmd = Twist()

        if self.current_pose is None:
            return cmd

        # Placeholder algorithm - replace with actual implementation
        cmd.linear.x = self.max_speed
        cmd.angular.z = 0.0

        return cmd

    def run(self):
        \"\"\"Main control loop\"\"\"
        rate = rospy.Rate(self.update_rate)

        while not rospy.is_shutdown():
            cmd_vel = self.compute_velocity()
            self.cmd_vel_pub.publish(cmd_vel)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = {algo_type.title().replace('_', '')}Node()
        node.run()
    except rospy.ROSInterruptException:
        pass
"""

    def _generate_launch_file(self, package_name: str, algorithms: List[Dict]) -> str:
        """Generate launch file"""
        nodes = []
        for i, algo in enumerate(algorithms):
            node_name = f"{algo.get('type', 'algorithm')}_{i+1}"
            script_name = f"{node_name}_node.py"
            nodes.append(f'  <node name="{node_name}" pkg="{package_name}" type="{script_name}" output="screen" />')

        nodes_str = "\\n".join(nodes)

        return f"""<launch>
  <!-- Load parameters -->
  <rosparam command="load" file="$(find {package_name})/config/params.yaml" />

  <!-- Algorithm nodes -->
{nodes_str}

  <!-- RViz visualization -->
  <node name="rviz" pkg="rviz" type="rviz" args="-d $(find {package_name})/config/simulation.rviz" required="false" />
</launch>
"""

    def _generate_params_yaml(self, scene_config: Dict) -> str:
        """Generate parameters YAML"""
        return f"""# Algorithm Parameters
max_speed: 0.5
max_angular_speed: 1.0
update_rate: 10

# Path Planning
path_planning:
  grid_resolution: 0.1
  heuristic_weight: 1.0

# Obstacle Avoidance
obstacle_avoidance:
  safety_margin: 0.5
  prediction_time: 2.0

# Vision
vision:
  detection_threshold: 0.5
  detection_range: 10.0

# Scene Configuration
scene:
  environment: {scene_config.get('environment', 'warehouse')}
  bounds:
    min_x: -10
    max_x: 10
    min_y: -10
    max_y: 10
"""

    def _generate_urdf(self, robot: Dict) -> str:
        """Generate URDF robot description"""
        robot_type = robot.get('type', 'mobile')
        robot_name = f"{robot_type}_robot"

        if robot_type == 'mobile':
            return self._generate_mobile_robot_urdf(robot_name)
        elif robot_type == 'arm':
            return self._generate_arm_robot_urdf(robot_name)
        elif robot_type == 'drone':
            return self._generate_drone_robot_urdf(robot_name)
        else:
            return self._generate_mobile_robot_urdf(robot_name)

    def _generate_mobile_robot_urdf(self, name: str) -> str:
        """Generate mobile robot URDF"""
        return f"""<?xml version="1.0"?>
<robot name="{name}">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.4 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Wheels -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 0" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.2 0" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""

    def _generate_arm_robot_urdf(self, name: str) -> str:
        """Generate robotic arm URDF"""
        return f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
  </link>

  <!-- Add more links and joints for 6-DOF arm -->
  <!-- Simplified version - extend for full arm -->
</robot>
"""

    def _generate_drone_robot_urdf(self, name: str) -> str:
        """Generate drone URDF"""
        return f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.1"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
  </link>

  <!-- Add propellers -->
</robot>
"""

    def _generate_setup_script(self) -> str:
        """Generate setup script"""
        return """#!/bin/bash

echo "Setting up ROS workspace..."

# Check if ROS is installed
if [ -z "$ROS_DISTRO" ]; then
    echo "Error: ROS not found. Please source ROS first:"
    echo "  source /opt/ros/noetic/setup.bash"
    exit 1
fi

echo "Found ROS $ROS_DISTRO"

# Build workspace
echo "Building workspace..."
catkin_make

# Source workspace
echo "Sourcing workspace..."
source devel/setup.bash

echo "✅ Setup complete!"
echo ""
echo "To use the workspace, run:"
echo "  source devel/setup.bash"
echo "  roslaunch <package_name> simulation.launch"
"""
