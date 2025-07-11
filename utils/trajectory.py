import numpy as np
import pinocchio as pin
from utils.orientation import slerp_rotation

def generate_waypoint_trajectory(waypoints, duration, current_time):
    """
    Generate a trajectory through multiple waypoints using 5th order polynomials.
    Args:
        waypoints: List of SE3 poses representing waypoints
        duration: Total duration of the trajectory
        current_time: Current time since trajectory start
    Returns:
        Current target pose (SE3)
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints for trajectory generation")
    
    # Normalize time to [0, 1]
    t = min(current_time / duration, 1.0)
    
    # Calculate which segment we're in
    num_segments = len(waypoints) - 1
    segment_time = 1.0 / num_segments
    
    # Find current segment
    segment_idx = min(int(t / segment_time), num_segments - 1)
    segment_progress = (t - segment_idx * segment_time) / segment_time
    
    # Get start and end poses for current segment
    start_pose = waypoints[segment_idx]
    end_pose = waypoints[segment_idx + 1]
    
    # Generate 5th order polynomial trajectory for this segment
    return generate_5th_order_trajectory(start_pose, end_pose, segment_time * duration, segment_progress * segment_time * duration)
    
def create_waypoint_trajectory(start_pose, end_pose, intermediate_waypoints=None):
    """
    Create a waypoint trajectory from start to end with optional intermediate waypoints.
    Args:
        start_pose: Starting SE3 pose
        end_pose: Ending SE3 pose
        intermediate_waypoints: List of intermediate SE3 poses (optional)
    Returns:
        List of waypoints including start, intermediate, and end poses
    """
    waypoints = [start_pose]
    
    if intermediate_waypoints:
        waypoints.extend(intermediate_waypoints)
    
    waypoints.append(end_pose)
    return waypoints

def create_arc_trajectory(start_pose, end_pose, arc_height=0.1, num_waypoints=5):
    """
    Create an arc trajectory between two poses.
    Args:
        start_pose: Starting SE3 pose
        end_pose: Ending SE3 pose
        arc_height: Height of the arc above the straight line
        num_waypoints: Number of intermediate waypoints
    Returns:
        List of waypoints forming an arc
    """
    waypoints = [start_pose]
    
    # Calculate midpoint and arc direction
    start_pos = start_pose.translation
    end_pos = end_pose.translation
    mid_pos = (start_pos + end_pos) / 2
    
    # Calculate perpendicular direction for arc
    direction = end_pos - start_pos
    distance = np.linalg.norm(direction)
    if distance > 1e-6:
        direction = direction / distance
        
        # Create perpendicular vector (assuming we want arc in XY plane)
        perp_direction = np.array([-direction[1], direction[0], 0])
        if np.linalg.norm(perp_direction) < 1e-6:
            perp_direction = np.array([0, 0, 1])  # Fallback to Z direction
        
        # Generate intermediate waypoints along arc
        for i in range(1, num_waypoints + 1):
            t = i / (num_waypoints + 1)
            
            # Linear interpolation for position
            pos = (1 - t) * start_pos + t * end_pos
            
            # Add arc offset (parabolic)
            arc_offset = 4 * arc_height * t * (1 - t)  # Parabolic function for smooth arc
            pos += perp_direction * arc_offset
            
            # Interpolate rotation using SLERP
            rot = slerp_rotation(start_pose.rotation, end_pose.rotation, t)
            
            waypoints.append(pin.SE3(rot, pos))
    
    waypoints.append(end_pose)
    return waypoints

def create_circular_trajectory(center_pose, radius, start_angle, end_angle, num_waypoints=10):
    """
    Create a circular trajectory around a center point.
    Args:
        center_pose: Center SE3 pose
        radius: Radius of the circle
        start_angle: Starting angle in radians
        end_angle: Ending angle in radians
        num_waypoints: Number of waypoints along the arc
    Returns:
        List of waypoints forming a circular arc
    """
    waypoints = []
    
    for i in range(num_waypoints + 1):
        t = i / num_waypoints
        angle = start_angle + t * (end_angle - start_angle)
        
        # Calculate position on circle
        x = center_pose.translation[0] + radius * np.cos(angle)
        y = center_pose.translation[1] + radius * np.sin(angle)
        z = center_pose.translation[2]
        
        # Create rotation that faces the direction of motion
        tangent_angle = angle + np.pi/2  # Tangent to circle
        rot = np.array([
            [np.cos(tangent_angle), -np.sin(tangent_angle), 0],
            [np.sin(tangent_angle), np.cos(tangent_angle), 0],
            [0, 0, 1]
        ])
        
        waypoints.append(pin.SE3(rot, np.array([x, y, z])))
    
    return waypoints

def generate_5th_order_trajectory(start_pose, end_pose, duration, current_time):
    """
    Generate a 5th order polynomial trajectory between two poses.
    This provides smooth acceleration and deceleration for both position and rotation.
    """
    # Normalize time to [0, 1]
    t = min(current_time / duration, 1.0)
    
    # 5th order polynomial: s(t) = 6t^5 - 15t^4 + 10t^3
    # This gives zero velocity and acceleration at start and end
    s = 6 * t**5 - 15 * t**4 + 10 * t**3
    
    # Linear interpolation for position
    current_translation = (1.0 - s) * start_pose.translation + s * end_pose.translation
    
    # SLERP (Spherical Linear Interpolation) for rotation
    current_rotation = slerp_rotation(start_pose.rotation, end_pose.rotation, s)
    
    return pin.SE3(current_rotation, current_translation)
