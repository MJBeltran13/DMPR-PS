import math
import numpy as np

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def path_sample(start, goal, turn_radius, step_size):
    """Generate a path from start to goal pose using Reeds-Shepp curves.
    
    Args:
        start: Tuple (x, y, theta) for start pose
        goal: Tuple (x, y, theta) for goal pose
        turn_radius: Minimum turning radius of the vehicle
        step_size: Distance between points in the path
        
    Returns:
        List of (x, y, theta) poses forming the path
    """
    # Extract start and goal positions
    x1, y1, theta1 = start
    x2, y2, theta2 = goal
    
    # Calculate the distance between start and goal
    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx*dx + dy*dy)
    
    # If the distance is very small, just return a simple path
    if distance < step_size:
        return [start, goal]
    
    # Calculate the angle between start and goal
    angle_to_goal = math.atan2(dy, dx)
    
    # Calculate the angle differences
    angle_diff1 = normalize_angle(angle_to_goal - theta1)
    angle_diff2 = normalize_angle(theta2 - angle_to_goal)
    
    # Determine if we need to go forward or backward
    # If the angle difference is greater than 90 degrees, it's better to go backward
    if abs(angle_diff1) > math.pi/2:
        # We need to go backward first
        return generate_backward_path(start, goal, turn_radius, step_size)
    else:
        # We can go forward
        return generate_forward_path(start, goal, turn_radius, step_size)

def generate_forward_path(start, goal, turn_radius, step_size):
    """Generate a forward path from start to goal."""
    x1, y1, theta1 = start
    x2, y2, theta2 = goal
    
    # Calculate the distance between start and goal
    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx*dx + dy*dy)
    
    # Calculate the angle between start and goal
    angle_to_goal = math.atan2(dy, dx)
    
    # Calculate the angle differences
    angle_diff1 = normalize_angle(angle_to_goal - theta1)
    angle_diff2 = normalize_angle(theta2 - angle_to_goal)
    
    # Create a path with three segments: turn, straight, turn
    path = []
    
    # First turn: adjust heading to face the goal
    turn1_angle = angle_diff1
    turn1_radius = min(turn_radius, abs(turn1_angle) * turn_radius / math.pi)
    turn1_arc_length = abs(turn1_angle) * turn1_radius
    
    # Generate points for the first turn
    num_points1 = max(1, int(turn1_arc_length / step_size))
    for i in range(num_points1 + 1):
        t = i / num_points1
        angle = theta1 + t * turn1_angle
        # Calculate position along the arc
        if turn1_angle > 0:
            # Turning left
            x = x1 + turn1_radius * (math.sin(angle) - math.sin(theta1))
            y = y1 + turn1_radius * (math.cos(theta1) - math.cos(angle))
        else:
            # Turning right
            x = x1 + turn1_radius * (math.sin(theta1) - math.sin(angle))
            y = y1 + turn1_radius * (math.cos(angle) - math.cos(theta1))
        
        path.append((x, y, angle))
    
    # Straight segment: move towards the goal
    # Calculate the start and end points of the straight segment
    if turn1_angle > 0:
        # After turning left
        straight_start_x = x1 + turn1_radius * (math.sin(theta1 + turn1_angle) - math.sin(theta1))
        straight_start_y = y1 + turn1_radius * (math.cos(theta1) - math.cos(theta1 + turn1_angle))
        straight_start_theta = theta1 + turn1_angle
    else:
        # After turning right
        straight_start_x = x1 + turn1_radius * (math.sin(theta1) - math.sin(theta1 + turn1_angle))
        straight_start_y = y1 + turn1_radius * (math.cos(theta1 + turn1_angle) - math.cos(theta1))
        straight_start_theta = theta1 + turn1_angle
    
    # Calculate the end point of the straight segment
    if angle_diff2 > 0:
        # Need to turn left at the end
        straight_end_x = x2 - turn1_radius * (math.sin(theta2) - math.sin(theta2 - angle_diff2))
        straight_end_y = y2 - turn1_radius * (math.cos(theta2 - angle_diff2) - math.cos(theta2))
    else:
        # Need to turn right at the end
        straight_end_x = x2 - turn1_radius * (math.sin(theta2 - angle_diff2) - math.sin(theta2))
        straight_end_y = y2 - turn1_radius * (math.cos(theta2) - math.cos(theta2 - angle_diff2))
    
    # Calculate the length of the straight segment
    straight_dx = straight_end_x - straight_start_x
    straight_dy = straight_end_y - straight_start_y
    straight_length = math.sqrt(straight_dx*straight_dx + straight_dy*straight_dy)
    
    # Generate points for the straight segment
    num_points2 = max(1, int(straight_length / step_size))
    for i in range(1, num_points2 + 1):
        t = i / num_points2
        x = straight_start_x + t * straight_dx
        y = straight_start_y + t * straight_dy
        path.append((x, y, straight_start_theta))
    
    # Second turn: adjust heading to match the goal
    # Calculate the start angle of the second turn
    if angle_diff2 > 0:
        # Turning left
        turn2_start_theta = straight_start_theta
    else:
        # Turning right
        turn2_start_theta = straight_start_theta
    
    # Generate points for the second turn
    num_points3 = max(1, int(abs(angle_diff2) * turn1_radius / step_size))
    for i in range(1, num_points3 + 1):
        t = i / num_points3
        angle = turn2_start_theta + t * angle_diff2
        
        # Calculate position along the arc
        if angle_diff2 > 0:
            # Turning left
            x = straight_end_x + turn1_radius * (math.sin(angle) - math.sin(turn2_start_theta))
            y = straight_end_y + turn1_radius * (math.cos(turn2_start_theta) - math.cos(angle))
        else:
            # Turning right
            x = straight_end_x + turn1_radius * (math.sin(turn2_start_theta) - math.sin(angle))
            y = straight_end_y + turn1_radius * (math.cos(angle) - math.cos(turn2_start_theta))
        
        path.append((x, y, angle))
    
    # Add the goal point
    path.append(goal)
    
    return path

def generate_backward_path(start, goal, turn_radius, step_size):
    """Generate a backward path from start to goal."""
    # For backward paths, we first need to turn to face away from the goal
    x1, y1, theta1 = start
    x2, y2, theta2 = goal
    
    # Calculate the angle between start and goal
    dx = x2 - x1
    dy = y2 - y1
    angle_to_goal = math.atan2(dy, dx)
    
    # Calculate the angle to turn to face away from the goal
    angle_diff1 = normalize_angle(angle_to_goal - theta1)
    
    # Determine which way to turn (left or right)
    if angle_diff1 > 0:
        # Turn left
        turn_angle = math.pi - angle_diff1
    else:
        # Turn right
        turn_angle = -(math.pi + angle_diff1)
    
    # Calculate the intermediate point after the first turn
    turn_radius_actual = min(turn_radius, abs(turn_angle) * turn_radius / math.pi)
    
    if turn_angle > 0:
        # Turning left
        intermediate_x = x1 + turn_radius_actual * (math.sin(theta1 + turn_angle) - math.sin(theta1))
        intermediate_y = y1 + turn_radius_actual * (math.cos(theta1) - math.cos(theta1 + turn_angle))
        intermediate_theta = theta1 + turn_angle
    else:
        # Turning right
        intermediate_x = x1 + turn_radius_actual * (math.sin(theta1) - math.sin(theta1 + turn_angle))
        intermediate_y = y1 + turn_radius_actual * (math.cos(theta1 + turn_angle) - math.cos(theta1))
        intermediate_theta = theta1 + turn_angle
    
    # Now generate a forward path from the intermediate point to the goal
    intermediate_path = generate_forward_path(
        (intermediate_x, intermediate_y, intermediate_theta),
        goal,
        turn_radius,
        step_size
    )
    
    # Generate the first turn
    first_turn_path = []
    num_points = max(1, int(abs(turn_angle) * turn_radius_actual / step_size))
    for i in range(num_points + 1):
        t = i / num_points
        angle = theta1 + t * turn_angle
        
        # Calculate position along the arc
        if turn_angle > 0:
            # Turning left
            x = x1 + turn_radius_actual * (math.sin(angle) - math.sin(theta1))
            y = y1 + turn_radius_actual * (math.cos(theta1) - math.cos(angle))
        else:
            # Turning right
            x = x1 + turn_radius_actual * (math.sin(theta1) - math.sin(angle))
            y = y1 + turn_radius_actual * (math.cos(angle) - math.cos(theta1))
        
        first_turn_path.append((x, y, angle))
    
    # Combine the paths
    return first_turn_path + intermediate_path[1:] 