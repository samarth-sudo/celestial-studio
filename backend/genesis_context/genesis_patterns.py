"""
Common Genesis Code Patterns
Pre-defined patterns for common robotics tasks.
"""

MOBILE_ROBOT_PATTERN = """
import genesis as gs
import numpy as np

# Initialize Genesis
gs.init(backend=gs.metal)

# Create scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=False,
)

# Add ground plane
plane = scene.add_entity(gs.morphs.Plane())

# Add mobile robot (simple box representation)
robot = scene.add_entity(
    gs.morphs.Box(
        size=(0.5, 0.3, 0.2),
        pos=(0, 0, 0.5),
    ),
)

# Build scene
scene.build()

# Main control loop
for step in range(1000):
    # Set robot velocity
    vx = 1.0  # Linear velocity (m/s)
    omega = 0.1  # Angular velocity (rad/s)
    robot.set_velocity(
        linear=(vx, 0, 0),
        angular=(0, 0, omega)
    )

    # Get robot state
    position = robot.get_pos()
    velocity = robot.get_vel()

    # Step simulation
    scene.step()
"""

ROBOTIC_ARM_PATTERN = """
import genesis as gs
import numpy as np

# Initialize Genesis
gs.init(backend=gs.metal)

# Create scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=False,
)

# Add ground and arm
plane = scene.add_entity(gs.morphs.Plane())
arm = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

# Build scene
scene.build()

# Configure joint control
jnt_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
dofs_idx = [arm.get_joint(name).dof_idx_local for name in jnt_names]

# Set PD gains
arm.set_dofs_kp(kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]), dofs_idx_local=dofs_idx)
arm.set_dofs_kv(kv=np.array([450, 450, 350, 350, 200, 200, 200]), dofs_idx_local=dofs_idx)

# Main control loop
target_positions = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
for step in range(1000):
    # Control arm to target positions
    arm.control_dofs_position(target_positions, dofs_idx)

    # Get current joint states
    current_pos = arm.get_dofs_position(dofs_idx_local=dofs_idx)
    current_vel = arm.get_dofs_velocity(dofs_idx_local=dofs_idx)

    # Step simulation
    scene.step()
"""

DRONE_PATTERN = """
import genesis as gs
import numpy as np

# Initialize Genesis
gs.init(backend=gs.metal)

# Create scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=False,
)

# Add ground and drone
plane = scene.add_entity(gs.morphs.Plane())
drone = scene.add_entity(
    gs.morphs.Box(
        size=(0.3, 0.3, 0.1),
        pos=(0, 0, 1.0),
    ),
)

# Build scene
scene.build()

# Main control loop
for step in range(1000):
    # Simple velocity control
    vx, vy, vz = 0.0, 0.0, 0.1  # Hover with slight climb
    roll_rate, pitch_rate, yaw_rate = 0.0, 0.0, 0.0

    drone.set_velocity(
        linear=(vx, vy, vz),
        angular=(roll_rate, pitch_rate, yaw_rate)
    )

    # Get drone state
    position = drone.get_pos()
    orientation = drone.get_euler()  # (roll, pitch, yaw)
    velocity = drone.get_vel()

    # Step simulation
    scene.step()
"""

CAMERA_RENDERING_PATTERN = """
import genesis as gs
import numpy as np
from PIL import Image

# Initialize Genesis
gs.init(backend=gs.metal)

# Create scene
scene = gs.Scene(show_viewer=False)

# Add entities
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.Box(size=(0.5, 0.3, 0.2), pos=(0, 0, 0.5))
)

# Build scene
scene.build()

# Add camera
camera = scene.add_camera(
    res=(1920, 1080),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=40,
    GUI=False,
)

# Main loop with rendering
for step in range(100):
    # Update robot
    robot.set_velocity(linear=(0.1, 0, 0), angular=(0, 0, 0))

    # Step simulation
    scene.step()

    # Render frame every 10 steps
    if step % 10 == 0:
        rgb_array = camera.render(rgb=True)  # Returns numpy array (H, W, 3)
        # Can save or process the image
        # img = Image.fromarray(rgb_array)
        # img.save(f'frame_{step}.png')
"""

PARALLEL_RL_PATTERN = """
import genesis as gs
import numpy as np

# Initialize Genesis
gs.init(backend=gs.metal)

# Number of parallel environments
n_envs = 1024

# Create scene with parallel environments
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=False,
)

# Add entities (automatically replicated across all envs)
plane = scene.add_entity(gs.morphs.Plane())
robots = []
for i in range(n_envs):
    robot = scene.add_entity(
        gs.morphs.Box(size=(0.5, 0.3, 0.2), pos=(0, 0, 0.5))
    )
    robots.append(robot)

# Build scene
scene.build()

# RL training loop
for episode in range(1000):
    # Reset environments
    for robot in robots:
        robot.set_dofs_position(np.zeros(3), range(3))

    # Episode loop
    for step in range(100):
        # Get observations (batched across all envs)
        observations = []
        for robot in robots:
            obs = np.concatenate([robot.get_pos(), robot.get_vel()])
            observations.append(obs)
        observations = np.array(observations)

        # Compute actions (your policy here)
        actions = np.random.randn(n_envs, 3) * 0.1  # Random policy

        # Apply actions
        for i, robot in enumerate(robots):
            robot.set_velocity(
                linear=(actions[i, 0], actions[i, 1], 0),
                angular=(0, 0, actions[i, 2])
            )

        # Step all environments in parallel
        scene.step()
"""

# Pattern registry
PATTERNS = {
    'mobile_robot': MOBILE_ROBOT_PATTERN,
    'robotic_arm': ROBOTIC_ARM_PATTERN,
    'drone': DRONE_PATTERN,
    'camera': CAMERA_RENDERING_PATTERN,
    'rl_training': PARALLEL_RL_PATTERN,
}


def get_pattern(pattern_name: str) -> str:
    """Get code pattern by name"""
    return PATTERNS.get(pattern_name, "")


def get_all_patterns() -> dict:
    """Get all available patterns"""
    return PATTERNS.copy()
