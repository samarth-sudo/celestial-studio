"""
Reward Functions for Robot Tasks

Collection of reusable reward functions for manipulation tasks, inspired by Isaac Lab.
Provides common reward computations: distance, orientation, velocity penalties, etc.

Usage:
    from backend.tasks.rewards import compute_distance_reward, compute_orientation_reward

    distance_rew = compute_distance_reward(ee_pos, target_pos, threshold=0.01)
    orientation_rew = compute_orientation_reward(ee_quat, target_quat)

References:
    - Isaac Lab: isaaclab/envs/mdp/rewards.py
"""

import numpy as np
from typing import Optional
from backend.utils.math_utils import compute_pose_error, quat_to_axis_angle


# ============================================================================
# Distance-Based Rewards
# ============================================================================

def compute_distance_reward(
    current_pos: np.ndarray,
    target_pos: np.ndarray,
    threshold: float = 0.01,
    kernel: str = "linear",
    sigma: float = 0.1
) -> float:
    """
    Compute reward based on distance to target

    Args:
        current_pos: Current position [x, y, z]
        target_pos: Target position [x, y, z]
        threshold: Success threshold (meters)
        kernel: Reward shaping ("linear", "exponential", "gaussian", "tanh")
        sigma: Kernel width parameter

    Returns:
        Reward scalar (higher is better)

    Example:
        >>> reward = compute_distance_reward(
        ...     np.array([0.5, 0, 0.3]),
        ...     np.array([0.6, 0, 0.3]),
        ...     kernel="exponential"
        ... )
    """
    distance = np.linalg.norm(target_pos - current_pos)

    if kernel == "linear":
        # Simple negative distance
        reward = -distance

    elif kernel == "exponential":
        # Exponential falloff: exp(-distance / sigma)
        reward = np.exp(-distance / sigma)

    elif kernel == "gaussian":
        # Gaussian around target: exp(-(distance^2) / (2*sigma^2))
        reward = np.exp(-(distance ** 2) / (2 * sigma ** 2))

    elif kernel == "tanh":
        # Smooth saturation: 1 - tanh(distance / sigma)
        reward = 1.0 - np.tanh(distance / sigma)

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Bonus for reaching threshold
    if distance < threshold:
        reward += 1.0

    return reward


def compute_reaching_reward(
    ee_pos: np.ndarray,
    object_pos: np.ndarray,
    threshold: float = 0.05,
    scale: float = 1.0
) -> float:
    """
    Reward for reaching towards an object

    Args:
        ee_pos: End-effector position [x, y, z]
        object_pos: Object position [x, y, z]
        threshold: Grasping threshold (meters)
        scale: Reward scaling factor

    Returns:
        Reward scalar
    """
    distance = np.linalg.norm(ee_pos - object_pos)

    # Negative distance (dense reward)
    reward = -distance * scale

    # Large bonus when within grasp range
    if distance < threshold:
        reward += 5.0

    return reward


# ============================================================================
# Orientation Rewards
# ============================================================================

def compute_orientation_reward(
    current_quat: np.ndarray,
    target_quat: np.ndarray,
    threshold: float = 0.1,
    kernel: str = "linear"
) -> float:
    """
    Compute reward based on orientation error

    Args:
        current_quat: Current orientation [w, x, y, z]
        target_quat: Target orientation [w, x, y, z]
        threshold: Success threshold (radians)
        kernel: Reward shaping ("linear", "exponential")

    Returns:
        Reward scalar

    Example:
        >>> reward = compute_orientation_reward(
        ...     np.array([1, 0, 0, 0]),
        ...     np.array([0.707, 0.707, 0, 0])
        ... )
    """
    # Compute orientation error as axis-angle
    _, rot_error = compute_pose_error(
        np.zeros(3), current_quat,
        np.zeros(3), target_quat,
        rot_error_type="axis_angle"
    )

    angle_error = np.linalg.norm(rot_error)

    if kernel == "linear":
        reward = -angle_error
    elif kernel == "exponential":
        reward = np.exp(-angle_error / 0.5)  # 0.5 rad sigma
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Bonus for reaching threshold
    if angle_error < threshold:
        reward += 1.0

    return reward


def compute_alignment_reward(
    current_vec: np.ndarray,
    target_vec: np.ndarray
) -> float:
    """
    Reward for aligning two vectors (e.g., gripper approach vs object normal)

    Uses dot product: reward = vec1 · vec2 / (|vec1| * |vec2|)

    Args:
        current_vec: Current vector [x, y, z]
        target_vec: Target vector [x, y, z]

    Returns:
        Reward in range [-1, 1] (1 = aligned, -1 = opposite)

    Example:
        >>> # Gripper pointing down, object normal pointing up
        >>> reward = compute_alignment_reward(
        ...     np.array([0, 0, -1]),
        ...     np.array([0, 0, 1])
        ... )
        >>> print(reward)  # -1.0 (opposite)
    """
    # Normalize vectors
    current_norm = current_vec / (np.linalg.norm(current_vec) + 1e-8)
    target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-8)

    # Dot product
    alignment = np.dot(current_norm, target_norm)

    return alignment


# ============================================================================
# Velocity Penalties
# ============================================================================

def compute_velocity_penalty(
    joint_vel: np.ndarray,
    scale: float = 0.01
) -> float:
    """
    Penalty for excessive joint velocities (smoothness regularization)

    Args:
        joint_vel: Joint velocities [num_joints]
        scale: Penalty scaling factor

    Returns:
        Penalty (negative value)

    Example:
        >>> penalty = compute_velocity_penalty(joint_vel, scale=0.01)
    """
    velocity_magnitude = np.linalg.norm(joint_vel)
    return -velocity_magnitude * scale


def compute_action_penalty(
    action: np.ndarray,
    prev_action: Optional[np.ndarray] = None,
    scale: float = 0.001
) -> float:
    """
    Penalty for large or jerky actions (encourage smooth control)

    If prev_action provided, penalizes action changes (jerk).
    Otherwise, penalizes action magnitude.

    Args:
        action: Current action [num_joints]
        prev_action: Previous action (optional)
        scale: Penalty scaling factor

    Returns:
        Penalty (negative value)
    """
    if prev_action is not None:
        # Penalize action change (jerk)
        action_diff = action - prev_action
        return -np.linalg.norm(action_diff) * scale
    else:
        # Penalize action magnitude
        return -np.linalg.norm(action) * scale


# ============================================================================
# Grasping Rewards
# ============================================================================

def compute_grasp_reward(
    ee_pos: np.ndarray,
    object_pos: np.ndarray,
    gripper_state: float,
    grasp_threshold: float = 0.02,
    object_lifted: bool = False,
    lift_height: float = 0.1
) -> float:
    """
    Reward for grasping and lifting an object

    Stages:
        1. Approach object (distance reward)
        2. Close gripper near object (grasp bonus)
        3. Lift object (lift bonus)

    Args:
        ee_pos: End-effector position [x, y, z]
        object_pos: Object position [x, y, z]
        gripper_state: Gripper opening (0=closed, 1=open)
        grasp_threshold: Distance for successful grasp (meters)
        object_lifted: Whether object is above initial height
        lift_height: Target lift height (meters)

    Returns:
        Reward scalar

    Example:
        >>> reward = compute_grasp_reward(
        ...     ee_pos=np.array([0.5, 0, 0.2]),
        ...     object_pos=np.array([0.5, 0, 0.05]),
        ...     gripper_state=0.2,  # Mostly closed
        ...     object_lifted=True
        ... )
    """
    reward = 0.0

    # Distance to object
    distance = np.linalg.norm(ee_pos - object_pos)
    reward += -distance * 2.0  # Dense reaching reward

    # Gripper closing bonus (when near object)
    if distance < grasp_threshold:
        reward += (1.0 - gripper_state) * 3.0  # Reward closing gripper

    # Grasp success bonus
    if distance < grasp_threshold and gripper_state < 0.3:
        reward += 5.0

    # Lift bonus
    if object_lifted:
        lift_distance = object_pos[2] - (object_pos[2] - lift_height)
        reward += lift_distance * 10.0  # Scale by height
        reward += 10.0  # Bonus for lifting

    return reward


def compute_contact_reward(
    gripper_closed: bool,
    object_in_gripper: bool
) -> float:
    """
    Simple binary reward for successful contact/grasp

    Args:
        gripper_closed: True if gripper is closed
        object_in_gripper: True if object detected in gripper

    Returns:
        Reward (0 or 1)
    """
    if gripper_closed and object_in_gripper:
        return 1.0
    return 0.0


# ============================================================================
# Task-Specific Rewards
# ============================================================================

def compute_lift_reward(
    object_pos: np.ndarray,
    initial_height: float,
    target_height: float,
    threshold: float = 0.01
) -> float:
    """
    Reward for lifting object to target height

    Args:
        object_pos: Current object position [x, y, z]
        initial_height: Object's initial Z position
        target_height: Target Z position
        threshold: Success threshold (meters)

    Returns:
        Reward scalar
    """
    current_height = object_pos[2]
    height_diff = current_height - initial_height
    target_diff = target_height - initial_height

    # Progress reward (0 to 1)
    if target_diff > 0:
        progress = np.clip(height_diff / target_diff, 0.0, 1.0)
    else:
        progress = 0.0

    reward = progress * 5.0

    # Success bonus
    if abs(current_height - target_height) < threshold:
        reward += 10.0

    return reward


def compute_placing_reward(
    object_pos: np.ndarray,
    target_pos: np.ndarray,
    object_vel: np.ndarray,
    threshold: float = 0.02,
    stability_threshold: float = 0.01
) -> float:
    """
    Reward for placing object at target location

    Considers both position accuracy and stability (low velocity).

    Args:
        object_pos: Current object position [x, y, z]
        target_pos: Target placement position [x, y, z]
        object_vel: Object linear velocity [vx, vy, vz]
        threshold: Position threshold for success (meters)
        stability_threshold: Velocity threshold for stability (m/s)

    Returns:
        Reward scalar
    """
    distance = np.linalg.norm(object_pos - target_pos)
    velocity = np.linalg.norm(object_vel)

    # Distance reward
    reward = -distance * 5.0

    # Stability bonus (low velocity near target)
    if distance < threshold:
        reward += (1.0 - np.tanh(velocity / stability_threshold)) * 5.0

    # Success bonus (at target and stable)
    if distance < threshold and velocity < stability_threshold:
        reward += 10.0

    return reward


# ============================================================================
# Penalty Functions
# ============================================================================

def compute_collision_penalty(
    collision_detected: bool,
    penalty: float = -10.0
) -> float:
    """
    Penalty for collisions (self-collision or with obstacles)

    Args:
        collision_detected: True if collision occurred
        penalty: Penalty value (negative)

    Returns:
        Penalty (0 or negative value)
    """
    return penalty if collision_detected else 0.0


def compute_joint_limit_penalty(
    joint_pos: np.ndarray,
    joint_limits_lower: np.ndarray,
    joint_limits_upper: np.ndarray,
    margin: float = 0.1,
    scale: float = 1.0
) -> float:
    """
    Penalty for approaching joint limits

    Args:
        joint_pos: Current joint positions [num_joints]
        joint_limits_lower: Lower joint limits [num_joints]
        joint_limits_upper: Upper joint limits [num_joints]
        margin: Fraction of range to start penalizing (0.1 = 10%)
        scale: Penalty scaling factor

    Returns:
        Penalty (negative value if near limits)

    Example:
        >>> penalty = compute_joint_limit_penalty(
        ...     joint_pos=np.array([2.8, 0.5]),
        ...     joint_limits_lower=np.array([-3.0, -2.0]),
        ...     joint_limits_upper=np.array([3.0, 2.0]),
        ...     margin=0.1
        ... )
    """
    penalty = 0.0

    for i in range(len(joint_pos)):
        joint_range = joint_limits_upper[i] - joint_limits_lower[i]
        margin_dist = margin * joint_range

        # Lower limit violation
        if joint_pos[i] < joint_limits_lower[i] + margin_dist:
            dist = (joint_limits_lower[i] + margin_dist) - joint_pos[i]
            penalty -= dist * scale

        # Upper limit violation
        if joint_pos[i] > joint_limits_upper[i] - margin_dist:
            dist = joint_pos[i] - (joint_limits_upper[i] - margin_dist)
            penalty -= dist * scale

    return penalty


def compute_timeout_penalty(
    step_count: int,
    max_steps: int,
    penalty: float = -1.0
) -> float:
    """
    Penalty for timeout (episode exceeded max steps)

    Args:
        step_count: Current step count
        max_steps: Maximum allowed steps
        penalty: Penalty value

    Returns:
        Penalty (0 or negative value)
    """
    return penalty if step_count >= max_steps else 0.0


# ============================================================================
# Combined Rewards
# ============================================================================

def compute_reach_reward(
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    pos_weight: float = 1.0,
    rot_weight: float = 0.5,
    joint_vel: Optional[np.ndarray] = None,
    vel_penalty_scale: float = 0.01
) -> float:
    """
    Combined reward for reaching task (position + orientation + smoothness)

    Args:
        ee_pos: End-effector position [x, y, z]
        ee_quat: End-effector orientation [w, x, y, z]
        target_pos: Target position [x, y, z]
        target_quat: Target orientation [w, x, y, z]
        pos_weight: Weight for position reward
        rot_weight: Weight for rotation reward
        joint_vel: Joint velocities (optional, for smoothness penalty)
        vel_penalty_scale: Velocity penalty scaling

    Returns:
        Combined reward
    """
    # Position reward
    pos_reward = compute_distance_reward(ee_pos, target_pos, kernel="exponential")

    # Orientation reward
    rot_reward = compute_orientation_reward(ee_quat, target_quat, kernel="exponential")

    # Velocity penalty (optional)
    vel_penalty = 0.0
    if joint_vel is not None:
        vel_penalty = compute_velocity_penalty(joint_vel, scale=vel_penalty_scale)

    return pos_weight * pos_reward + rot_weight * rot_reward + vel_penalty


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Reward Functions Examples")
    print("=" * 50)

    # Distance reward
    ee_pos = np.array([0.5, 0.0, 0.3])
    target_pos = np.array([0.6, 0.1, 0.3])

    print("\n1. Distance Rewards:")
    for kernel in ["linear", "exponential", "gaussian", "tanh"]:
        reward = compute_distance_reward(ee_pos, target_pos, kernel=kernel)
        print(f"   {kernel:12s}: {reward:+.3f}")

    # Orientation reward
    print("\n2. Orientation Reward:")
    quat1 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    quat2 = np.array([0.707, 0.707, 0.0, 0.0])  # 90° around X
    reward = compute_orientation_reward(quat1, quat2)
    print(f"   Reward: {reward:+.3f}")

    # Grasp reward
    print("\n3. Grasp Reward:")
    object_pos = np.array([0.5, 0.0, 0.05])
    gripper_state = 0.2  # Mostly closed
    reward = compute_grasp_reward(ee_pos, object_pos, gripper_state, object_lifted=False)
    print(f"   Reward: {reward:+.3f}")

    # Combined reach reward
    print("\n4. Combined Reach Reward:")
    joint_vel = np.array([0.1, 0.2, 0.05, 0.1, 0.0, 0.05, 0.0])
    reward = compute_reach_reward(
        ee_pos, quat1,
        target_pos, quat2,
        joint_vel=joint_vel
    )
    print(f"   Total Reward: {reward:+.3f}")

    print("\n" + "=" * 50)
