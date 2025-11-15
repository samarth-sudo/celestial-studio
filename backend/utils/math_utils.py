"""
Robotics Math Utilities

Essential mathematical functions for robot manipulation, adapted from Isaac Lab.
Handles quaternions, transforms, and pose computations needed for inverse kinematics.

References:
- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- Quaternion conventions: [w, x, y, z] (scalar-first)
"""

import numpy as np
from typing import Tuple


# ============================================================================
# Quaternion Operations
# ============================================================================

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions: q1 * q2

    Args:
        q1: Quaternion [w, x, y, z]
        q2: Quaternion [w, x, y, z]

    Returns:
        Product quaternion [w, x, y, z]

    Example:
        >>> q1 = np.array([1, 0, 0, 0])  # Identity
        >>> q2 = np.array([0.707, 0.707, 0, 0])  # 90° around X
        >>> result = quat_mul(q1, q2)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion conjugate (inverse for unit quaternions)

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Conjugate [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Rotate a 3D vector by a quaternion

    Computes: q * v * q^(-1) where v is treated as pure quaternion [0, x, y, z]

    Args:
        quat: Rotation quaternion [w, x, y, z]
        vec: 3D vector [x, y, z]

    Returns:
        Rotated vector [x', y', z']

    Example:
        >>> quat = np.array([0.707, 0.707, 0, 0])  # 90° around X
        >>> vec = np.array([0, 1, 0])  # Y-axis
        >>> result = quat_apply(quat, vec)  # Should give [0, 0, 1] (Z-axis)
    """
    # Convert vector to pure quaternion
    v_quat = np.array([0, vec[0], vec[1], vec[2]])

    # Compute q * v * q^(-1)
    result = quat_mul(quat_mul(quat, v_quat), quat_conjugate(quat))

    # Extract vector part
    return result[1:]


def quat_to_axis_angle(quat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Convert quaternion to axis-angle representation

    Args:
        quat: Unit quaternion [w, x, y, z]
        eps: Small value to avoid division by zero

    Returns:
        Axis-angle vector [ax, ay, az] where magnitude is angle in radians

    Note:
        For small rotations (angle < eps), returns zero vector
    """
    # Compute angle
    angle = 2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))

    # Handle near-zero rotations
    if abs(angle) < eps:
        return np.zeros(3)

    # Extract axis
    sin_half_angle = np.sin(angle / 2.0)
    axis = quat[1:] / sin_half_angle

    # Return axis scaled by angle
    return axis * angle


def axis_angle_to_quat(axis_angle: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Convert axis-angle to quaternion

    Args:
        axis_angle: Axis-angle vector [ax, ay, az]
        eps: Small value threshold

    Returns:
        Unit quaternion [w, x, y, z]
    """
    angle = np.linalg.norm(axis_angle)

    if angle < eps:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = axis_angle / angle
    half_angle = angle / 2.0

    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)

    return np.array([w, xyz[0], xyz[1], xyz[2]])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (ZYX convention) to quaternion

    Args:
        roll: Rotation around X-axis (radians)
        pitch: Rotation around Y-axis (radians)
        yaw: Rotation around Z-axis (radians)

    Returns:
        Quaternion [w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quat_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (ZYX convention)

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        (roll, pitch, yaw) in radians
    """
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ============================================================================
# Pose Transformations
# ============================================================================

def compute_pose_error(
    pos_current: np.ndarray,
    quat_current: np.ndarray,
    pos_target: np.ndarray,
    quat_target: np.ndarray,
    rot_error_type: str = "axis_angle"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute position and orientation error between two poses

    Used for inverse kinematics to determine how far current pose is from target.

    Args:
        pos_current: Current position [x, y, z]
        quat_current: Current orientation [w, x, y, z]
        pos_target: Target position [x, y, z]
        quat_target: Target orientation [w, x, y, z]
        rot_error_type: "axis_angle" or "quat" (axis-angle recommended for IK)

    Returns:
        (position_error, orientation_error)
        - position_error: [dx, dy, dz]
        - orientation_error: [ax, ay, az] if axis_angle, else [x, y, z, w]

    Example:
        >>> pos_err, rot_err = compute_pose_error(
        ...     np.array([0, 0, 0]), np.array([1, 0, 0, 0]),
        ...     np.array([0.1, 0.2, 0.3]), np.array([0.707, 0.707, 0, 0])
        ... )
    """
    # Position error (simple subtraction)
    pos_error = pos_target - pos_current

    # Orientation error: q_error = q_target * q_current^(-1)
    quat_error = quat_mul(quat_target, quat_conjugate(quat_current))

    if rot_error_type == "axis_angle":
        # Convert to axis-angle for IK (better for small rotations)
        rot_error = quat_to_axis_angle(quat_error)
    elif rot_error_type == "quat":
        # Return quaternion directly
        rot_error = quat_error
    else:
        raise ValueError(f"Unknown rot_error_type: {rot_error_type}")

    return pos_error, rot_error


def combine_frame_transforms(
    pos_parent: np.ndarray,
    quat_parent: np.ndarray,
    pos_child: np.ndarray,
    quat_child: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine two transforms: T_result = T_parent * T_child

    Used to compute end-effector pose from base pose and relative offset.

    Args:
        pos_parent: Parent frame position [x, y, z]
        quat_parent: Parent frame orientation [w, x, y, z]
        pos_child: Child frame position (in parent frame) [x, y, z]
        quat_child: Child frame orientation (in parent frame) [w, x, y, z]

    Returns:
        (pos_combined, quat_combined) - Combined transform in world frame

    Example:
        >>> # Robot base at [1, 0, 0], gripper offset [0, 0, 0.1] from hand
        >>> pos, quat = combine_frame_transforms(
        ...     np.array([1, 0, 0]), np.array([1, 0, 0, 0]),
        ...     np.array([0, 0, 0.1]), np.array([1, 0, 0, 0])
        ... )
    """
    # Position: parent_pos + rotate(child_pos by parent_quat)
    pos_combined = pos_parent + quat_apply(quat_parent, pos_child)

    # Orientation: parent_quat * child_quat
    quat_combined = quat_mul(quat_parent, quat_child)

    return pos_combined, quat_combined


def apply_delta_pose(
    pos: np.ndarray,
    quat: np.ndarray,
    delta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply delta pose to current pose (for relative IK control)

    Args:
        pos: Current position [x, y, z]
        quat: Current orientation [w, x, y, z]
        delta: Pose change [dx, dy, dz, droll, dpitch, dyaw]

    Returns:
        (pos_new, quat_new) - Updated pose

    Example:
        >>> # Move 10cm forward and rotate 5° around Z
        >>> pos_new, quat_new = apply_delta_pose(
        ...     np.array([0, 0, 0]), np.array([1, 0, 0, 0]),
        ...     np.array([0.1, 0, 0, 0, 0, np.radians(5)])
        ... )
    """
    # Apply position delta
    pos_new = pos + delta[:3]

    # Convert orientation delta to quaternion
    if len(delta) >= 6:
        roll, pitch, yaw = delta[3:6]
        quat_delta = euler_to_quat(roll, pitch, yaw)
    else:
        # No orientation delta
        quat_delta = np.array([1, 0, 0, 0])

    # Apply orientation delta
    quat_new = quat_mul(quat, quat_delta)

    return pos_new, quat_new


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit length

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        Normalized quaternion
    """
    norm = np.linalg.norm(quat)
    if norm < 1e-10:
        return np.array([1, 0, 0, 0])
    return quat / norm


# ============================================================================
# Utility Functions
# ============================================================================

def transform_points(
    points: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray
) -> np.ndarray:
    """
    Transform points from one frame to another

    Args:
        points: Array of 3D points, shape (N, 3) or (3,)
        pos: Translation [x, y, z]
        quat: Rotation [w, x, y, z]

    Returns:
        Transformed points, same shape as input
    """
    points = np.atleast_2d(points)
    transformed = np.array([quat_apply(quat, p) + pos for p in points])
    return transformed.squeeze() if points.shape[0] == 1 else transformed


def matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def quat_from_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion

    Args:
        mat: 3x3 rotation matrix

    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(mat)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])
