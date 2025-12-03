# Genesis Physics Engine - API Reference

This is a curated reference for the most commonly used Genesis APIs for robot simulation and control.

## 1. Initialization

```python
import genesis as gs

# Initialize Genesis with backend
gs.init(backend=gs.gpu)    # GPU backend (CUDA/Metal)
gs.init(backend=gs.cpu)    # CPU backend (fallback)
gs.init(backend=gs.metal)  # Apple Silicon (M1/M2/M3)
gs.init(backend=gs.cuda)   # NVIDIA GPU
```

## 2. Scene Creation

```python
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,           # Timestep in seconds
        substeps=10,       # Physics substeps per step
        gravity=(0, 0, -9.81),  # Gravity vector
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),      # Camera position (x, y, z)
        camera_lookat=(0.0, 0.0, 0.5),  # Look-at point
        camera_fov=30,                   # Field of view
        max_FPS=60,                      # Maximum FPS
    ),
    show_viewer=True,  # Show GUI window (False for headless)
)
```

## 3. Adding Entities

### 3.1 Basic Shapes (Primitives)

```python
# Ground plane
plane = scene.add_entity(
    gs.morphs.Plane(),
)

# Box
box = scene.add_entity(
    gs.morphs.Box(
        size=(1.0, 1.0, 1.0),
        pos=(0, 0, 1.0),
        euler=(0, 0, 0),  # Euler angles in radians
    ),
)

# Sphere
sphere = scene.add_entity(
    gs.morphs.Sphere(
        radius=0.5,
        pos=(0, 0, 1.0),
    ),
)

# Cylinder
cylinder = scene.add_entity(
    gs.morphs.Cylinder(
        radius=0.3,
        height=1.0,
        pos=(0, 0, 0.5),
    ),
)
```

### 3.2 Loading Robots from Files

```python
# From MJCF (MuJoCo XML)
robot = scene.add_entity(
    gs.morphs.MJCF(
        file='xml/franka_emika_panda/panda.xml',
        pos=(0.0, 0.0, 0.0),
        euler=(0, 0, 0),
    ),
)

# From URDF
robot = scene.add_entity(
    gs.morphs.URDF(
        file='urdf/robot.urdf',
        pos=(0.0, 0.0, 0.0),
        euler=(0, 0, 0),
    ),
)
```

## 4. Building the Scene

```python
# IMPORTANT: Must be called after adding all entities
scene.build()
```

## 5. Robot Control

### 5.1 Getting Joint Information

```python
# Get joint by name
joint = robot.get_joint('joint_name')

# Get DOF (degree of freedom) index
dof_idx = joint.dof_idx_local  # Local to entity
dof_idx_global = joint.dof_idx  # Global in scene

# Get multiple DOFs
jnt_names = ['joint1', 'joint2', 'joint3']
dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]
```

### 5.2 Setting Control Gains (PD Controller)

```python
# Position gains (Kp)
robot.set_dofs_kp(
    kp=np.array([4500, 4500, 3500]),
    dofs_idx_local=dofs_idx,
)

# Velocity gains (Kv)
robot.set_dofs_kv(
    kv=np.array([450, 450, 350]),
    dofs_idx_local=dofs_idx,
)

# Force limits (safety)
robot.set_dofs_force_range(
    lower=np.array([-87, -87, -87]),
    upper=np.array([87, 87, 87]),
    dofs_idx_local=dofs_idx,
)
```

### 5.3 Direct State Control (Hard Reset)

```python
# Set joint positions directly (teleports robot)
robot.set_dofs_position(
    position=np.array([0.0, 1.0, -1.0]),
    dofs_idx_local=dofs_idx,
)

# Set joint velocities directly
robot.set_dofs_velocity(
    velocity=np.array([0.0, 0.0, 0.0]),
    dofs_idx_local=dofs_idx,
)
```

### 5.4 PD Controller (Physics-Based Control)

```python
# Set target joint positions (uses PD controller)
robot.control_dofs_position(
    target=np.array([0.0, 1.5, -1.0]),
    dofs_idx_local=dofs_idx,
)

# Set target joint velocities
robot.control_dofs_velocity(
    target=np.array([0.1, 0.2, 0.0]),
    dofs_idx_local=dofs_idx,
)

# Set direct torque/force
robot.control_dofs_force(
    force=np.array([10.0, 5.0, 2.0]),
    dofs_idx_local=dofs_idx,
)
```

### 5.5 Mobile Robot Control

```python
# For mobile robots (wheeled/tracked)
# Set velocity directly on rigid body
robot.set_velocity(
    linear=(vx, vy, 0),        # Linear velocity (m/s)
    angular=(0, 0, omega),     # Angular velocity (rad/s)
)
```

## 6. Simulation Loop

```python
# Step the simulation
scene.step()  # Advance by dt seconds

# Run simulation loop
for i in range(1000):
    # Update control
    robot.control_dofs_position(target_positions, dofs_idx)

    # Step simulation
    scene.step()

    # Get state (if needed)
    pos = robot.get_dofs_position(dofs_idx)
    vel = robot.get_dofs_velocity(dofs_idx)
```

## 7. Getting Robot State

```python
# Get joint positions
positions = robot.get_dofs_position(dofs_idx_local=dofs_idx)

# Get joint velocities
velocities = robot.get_dofs_velocity(dofs_idx_local=dofs_idx)

# Get base pose (for mobile robots)
pos = robot.get_pos()      # Returns (x, y, z)
quat = robot.get_quat()    # Returns quaternion (w, x, y, z)
euler = robot.get_euler()  # Returns Euler angles (roll, pitch, yaw)

# Get base velocity
lin_vel = robot.get_vel()      # Linear velocity
ang_vel = robot.get_ang()      # Angular velocity
```

## 8. Camera and Rendering

```python
# Add camera sensor
camera = scene.add_camera(
    res=(1920, 1080),          # Resolution
    pos=(3.5, 0.0, 2.5),       # Position
    lookat=(0, 0, 0.5),        # Look-at point
    fov=40,                     # Field of view
    GUI=False,                  # Don't show in GUI
)

# Render frame
rgb_array = camera.render(rgb=True)   # Returns numpy array (H, W, 3)
depth = camera.render(depth=True)     # Depth map
```

## 9. Collision Detection

```python
# Check if entity collides with any other entity
is_colliding = entity.is_colliding()

# Get contact information
contacts = scene.get_contacts()
```

## 10. Parallel Environments (RL Training)

```python
# Create scene with multiple parallel environments
n_envs = 1024
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=False,
)

# Batch operations automatically work on all envs
robots = [scene.add_entity(...) for _ in range(n_envs)]

# Batched control (all robots at once)
scene.step()  # Steps all environments in parallel
```

## 11. Common Patterns

### Pattern 1: Mobile Robot Navigation

```python
import genesis as gs
import numpy as np

gs.init(backend=gs.metal)

scene = gs.Scene(show_viewer=False)

# Add ground and robot
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.Box(size=(0.5, 0.3, 0.2), pos=(0, 0, 0.5))
)

scene.build()

# Control loop
for step in range(1000):
    # Set velocity for mobile robot
    vx = 1.0  # Forward 1 m/s
    omega = 0.1  # Turn 0.1 rad/s
    robot.set_velocity(linear=(vx, 0, 0), angular=(0, 0, omega))

    scene.step()
```

### Pattern 2: Robotic Arm Pick-and-Place

```python
import genesis as gs
import numpy as np

gs.init(backend=gs.metal)

scene = gs.Scene(show_viewer=True)

# Load arm and object
plane = scene.add_entity(gs.morphs.Plane())
arm = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
)
cube = scene.add_entity(
    gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.5, 0, 0.5))
)

scene.build()

# Get arm DOFs
jnt_names = ['joint1', 'joint2', 'joint3', 'joint4',
             'joint5', 'joint6', 'joint7']
dofs_idx = [arm.get_joint(name).dof_idx_local for name in jnt_names]

# Set control gains
arm.set_dofs_kp(kp=np.array([4500]*7), dofs_idx_local=dofs_idx)
arm.set_dofs_kv(kv=np.array([450]*7), dofs_idx_local=dofs_idx)

# Control loop
target_pos = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
for step in range(1000):
    arm.control_dofs_position(target_pos, dofs_idx)
    scene.step()
```

### Pattern 3: Drone Flight

```python
import genesis as gs
import numpy as np

gs.init(backend=gs.metal)

scene = gs.Scene(show_viewer=True)

# Load drone
plane = scene.add_entity(gs.morphs.Plane())
drone = scene.add_entity(
    gs.morphs.URDF(file='urdf/quadcopter.urdf', pos=(0, 0, 1))
)

scene.build()

# Control loop (simplified - real drones need attitude control)
for step in range(1000):
    # Apply upward thrust to counter gravity
    thrust = 9.81 * drone.get_mass() * 1.1  # 10% extra for lift
    drone.set_velocity(linear=(0, 0, 0.1), angular=(0, 0, 0))

    scene.step()
```

## 12. Important Notes

1. **Always call `scene.build()`** after adding all entities
2. **DOF indices**: Use `dof_idx_local` for entity-specific control
3. **Control methods**:
   - `set_dofs_position()` - Instant teleport (no physics)
   - `control_dofs_position()` - PD controller (physics-based)
4. **Coordinate system**: Right-handed, Z-up (x-forward, y-left, z-up)
5. **Units**: Meters, radians, seconds, Newtons
6. **Backend selection**:
   - `gs.metal` - Apple Silicon (M1/M2/M3)
   - `gs.cuda` - NVIDIA GPU
   - `gs.cpu` - CPU fallback (slowest)

## 13. Common Use Cases

| Task | Primary APIs |
|------|-------------|
| Mobile robot | `set_velocity()`, `get_pos()`, `get_vel()` |
| Robot arm | `control_dofs_position()`, `get_dofs_position()` |
| Drone | `set_velocity()`, `control_dofs_force()` |
| Pick-and-place | `control_dofs_position()`, collision detection |
| RL training | Parallel scenes, batch operations |
| Vision | `camera.render(rgb=True)` |
