import numpy as np

def slerp_rotation(start_rot, end_rot, t):
    """
    Perform SLERP (Spherical Linear Interpolation) between two rotation matrices.
    Args:
        start_rot: Starting rotation matrix (3x3)
        end_rot: Ending rotation matrix (3x3)
        t: Interpolation parameter [0, 1]
    Returns:
        Interpolated rotation matrix (3x3)
    """
    # Convert rotation matrices to quaternions
    start_quat = rotation_matrix_to_quaternion(start_rot)
    end_quat = rotation_matrix_to_quaternion(end_rot)
    
    # Ensure shortest path (dot product should be positive)
    if np.dot(start_quat, end_quat) < 0:
        end_quat = -end_quat
    
    # SLERP interpolation
    dot_product = np.dot(start_quat, end_quat)
    # Clamp dot product to avoid numerical issues
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    if abs(dot_product) > 0.9995:
        # Quaternions are very close, use linear interpolation
        interpolated_quat = (1 - t) * start_quat + t * end_quat
    else:
        theta = np.arccos(dot_product)
        sin_theta = np.sin(theta)
        interpolated_quat = (np.sin((1 - t) * theta) * start_quat + 
                            np.sin(t * theta) * end_quat) / sin_theta
    
    # Normalize quaternion
    interpolated_quat = interpolated_quat / np.linalg.norm(interpolated_quat)
    
    # Convert back to rotation matrix
    return quaternion_to_rotation_matrix(interpolated_quat)

def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion [w, x, y, z].
    """
    trace = np.trace(R)
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
        
    return np.array([w, x, y, z])

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to rotation matrix.
    """
    w, x, y, z = q
    
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])