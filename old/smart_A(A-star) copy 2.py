"""Smart Parking Assistance System with realistic car path planning."""
import cv2 as cv
import torch
import argparse
import math
import numpy as np
from enum import Enum
from model import DirectionalPointDetector
import config
from inference import detect_marking_points, inference_slots, plot_points, plot_slots
import matplotlib.pyplot as plt
import random
import copy

def setup_model(weights_path, use_cuda=False):
    """Setup the DMPR-PS model."""
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)
    
    # Initialize model
    model = DirectionalPointDetector(3, 32, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    return model, device

class Direction(Enum):
    FORWARD = 1
    REVERSE = -1

class CarParameters:
    def __init__(self):
        # Car dimensions in pixels (adjusted for better visualization)
        self.length = 60  # Length of the car (reduced from 100)
        self.width = 30   # Width of the car (reduced from 50)
        self.wheelbase = 40  # Distance between front and rear axles
        self.min_turn_radius = 80  # Minimum turning radius (reduced from 120)
        self.max_steering_angle = math.radians(35)  # Maximum steering angle in radians
        self.angle = 0  # Initialize angle attribute

class DraggableCar:
    def __init__(self, initial_pos, initial_angle):
        self.pos = initial_pos
        self.angle = initial_angle
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0

def get_slot_info(image, marking_points, slot):
    """Calculate and return detailed information about a parking slot."""
    height, width = image.shape[:2]
    point_a = marking_points[slot[0]]
    point_b = marking_points[slot[1]]
    
    # Get coordinates in pixels
    p0_x = int(round(width * point_a.x - 0.5))
    p0_y = int(round(height * point_a.y - 0.5))
    p1_x = int(round(width * point_b.x - 0.5))
    p1_y = int(round(height * point_b.y - 0.5))
    
    # Calculate angle in degrees
    angle = math.degrees(math.atan2(p1_y - p0_y, p1_x - p0_x))
    if angle < 0:
        angle += 360
        
    # Calculate center point
    center_x = (p0_x + p1_x) / 2
    center_y = (p0_y + p1_y) / 2
    
    return {
        'slot_id': len(slot),
        'center': (center_x, center_y),
        'angle': angle,
        'point1': (p0_x, p0_y),
        'point2': (p1_x, p1_y),
        'confidence': (point_a.shape, point_b.shape)
    }

def draw_car(image, position, car_params):
    """Draw a single red box representing the car at the given position."""
    x, y = position
    length = car_params.length
    width = car_params.width
    
    # Draw the car as a rectangle
    top_left = (int(x - length / 2), int(y - width / 2))
    bottom_right = (int(x + length / 2), int(y + width / 2))
    cv.rectangle(image, top_left, bottom_right, (0, 0, 255), -1)  # Red box

def calculate_arc_points(center, radius, start_angle, end_angle, direction, steps=20):
    """Calculate points along an arc."""
    points = []
    if direction == Direction.REVERSE:
        start_angle, end_angle = end_angle, start_angle
    
    for t in np.linspace(start_angle, end_angle, steps):
        x = center[0] + radius * math.cos(t)
        y = center[1] + radius * math.sin(t)
        points.append((int(x), int(y)))
    return points

class Node:
    def __init__(self, position, parent=None, angle=0):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to this node
        self.h = 0  # Heuristic cost to target
        self.f = 0  # Total cost
        self.angle = angle  # Angle of the car at this node

def astar(start, goal, car_params, grid):
    open_list = []
    closed_list = []

    start_node = Node(start)
    goal_node = Node(goal)
    open_list.append(start_node)

    while open_list:
        current_node = min(open_list, key=lambda o: o.f)
        open_list.remove(current_node)
        closed_list.append(current_node)

        if current_node.position == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        for new_position in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Up, Right, Down, Left
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if (0 <= node_position[0] < grid.shape[0]) and (0 <= node_position[1] < grid.shape[1]):
                if grid[node_position[0]][node_position[1]] == 1:  # Assuming 1 is an obstacle
                    continue

            # Calculate the angle for the new node based on the movement direction
            new_angle = current_node.angle  # Update this based on your turning logic

            child_node = Node(node_position, current_node, new_angle)

            # Calculate costs
            child_node.g = current_node.g + 1  # Assuming cost between nodes is 1
            child_node.h = (goal_node.position[0] - child_node.position[0]) ** 2 + (goal_node.position[1] - child_node.position[1]) ** 2
            child_node.f = child_node.g + child_node.h

            if child_node in closed_list:
                continue

            if add_to_open(open_list, child_node):
                open_list.append(child_node)

    return []  # Return an empty path if no path is found

def add_to_open(open_list, child_node):
    for node in open_list:
        if child_node.position == node.position and child_node.g >= node.g:
            return False
    return True

def calculate_parking_path(start_pos, target_pos, car_params, slot_angle=None):
    """
    Generate a realistic path from the car's position to the target parking slot
    considering the car's dimensions, angle, and turning constraints.
    
    Parameters:
    - start_pos: Tuple (x, y) of starting position
    - target_pos: Tuple (x, y) of target position (parking slot center)
    - car_params: CarParameters object with vehicle specs
    - slot_angle: Angle of the parking slot in radians (if None, will be calculated)
    
    Returns:
    - List of points representing the path
    """
    x1, y1 = start_pos
    x2, y2 = target_pos
    path_points = []
    
    # Calculate the angle to the target position
    target_angle = math.atan2(y2 - y1, x2 - x1)
    
    # Normalize the target angle
    target_angle = normalize_angle(target_angle)
    
    # Current car angle
    car_angle = car_params.angle
    
    # Print parameters
    print(f"Starting Position: {start_pos}, Starting Angle: {math.degrees(car_angle)}°")
    print(f"Target Position: {target_pos}, Target Angle: {math.degrees(target_angle)}°")
    
    # Calculate the angle difference
    angle_diff = normalize_angle(target_angle - car_angle)
    
    # Determine if we need to curve the path
    if abs(angle_diff) > math.radians(15):  # If the angle difference is significant
        return generate_curved_parking_path(start_pos, target_pos, car_angle, target_angle, car_params)
    else:
        return generate_direct_parking_path(start_pos, target_pos, car_angle, target_angle, car_params)

def normalize_angle(angle):
    """Normalize angle to range [-π, π]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def generate_direct_parking_path(start_pos, target_pos, start_angle, target_angle, car_params):
    """Generate a direct curved path to the parking slot"""
    path_points = []
    x1, y1 = start_pos
    x2, y2 = target_pos
    
    # Calculate distance to target
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Create a smoother approach by using a Bezier curve
    # Find a control point for the curve
    control_distance = distance * 0.7  # Control point at 70% of the distance
    
    # Control point in the direction of the car's current heading
    control_x = x1 + control_distance * math.cos(start_angle)
    control_y = y1 + control_distance * math.sin(start_angle)
    
    # Generate points along the Bezier curve
    num_steps = max(30, int(distance / 5))  # More steps for longer distances
    
    for t in np.linspace(0, 1, num_steps):
        # Quadratic Bezier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        # where P₀ is start, P₁ is control point, P₂ is end
        bx = (1-t)**2 * x1 + 2*(1-t)*t * control_x + t**2 * x2
        by = (1-t)**2 * y1 + 2*(1-t)*t * control_y + t**2 * y2
        
        # Calculate angle at this point (tangent to curve)
        if t < 1.0:
            next_t = min(1.0, t + 0.05)
            next_bx = (1-next_t)**2 * x1 + 2*(1-next_t)*next_t * control_x + next_t**2 * x2
            next_by = (1-next_t)**2 * y1 + 2*(1-next_t)*next_t * control_y + next_t**2 * y2
            point_angle = math.atan2(next_by - by, next_bx - bx)
        else:
            # At the end, use target angle
            point_angle = target_angle
        
        path_points.append((int(bx), int(by), point_angle))
    
    return path_points

def generate_three_point_parking_path(start_pos, target_pos, start_angle, target_angle, car_params):
    """Generate a three-point parking path for more complex maneuvers"""
    path_points = []
    x1, y1 = start_pos
    x2, y2 = target_pos
    
    # Calculate midpoint for the three-point turn
    # This is a simplified approach - for a real system, you would calculate
    # the exact intermediate positions based on car turning radius
    
    # First segment: Move to preparation position
    approach_distance = car_params.length * 2  # Distance to position before backing
    
    # Approach position (before starting to back into the spot)
    approach_x = x2 + approach_distance * math.cos(target_angle)
    approach_y = y2 + approach_distance * math.sin(target_angle)
    
    # Generate path to approach position (first segment)
    first_segment = generate_direct_parking_path(
        start_pos, 
        (approach_x, approach_y), 
        start_angle, 
        target_angle, 
        car_params
    )
    
    # Second segment: Back into the parking spot
    second_segment_steps = 15
    for i in range(second_segment_steps):
        t = i / (second_segment_steps - 1)
        bx = approach_x * (1-t) + x2 * t
        by = approach_y * (1-t) + y2 * t
        
        # When backing in, the car's angle is opposite to its movement
        backing_angle = normalize_angle(target_angle + math.pi)
        
        path_points.append((int(bx), int(by), backing_angle))
    
    # Combine segments, excluding duplicate point
    path_points = first_segment + path_points
    
    return path_points

def calculate_multiple_parking_paths(start_pos, target_pos, car_params, slot_angle=None, num_routes=3):
    """
    Generate multiple realistic parking paths with variations.
    
    Parameters:
    - start_pos: Tuple (x, y) of starting position
    - target_pos: Tuple (x, y) of target position (parking slot center)
    - car_params: CarParameters object with vehicle specs
    - slot_angle: Angle of the parking slot in radians
    - num_routes: Number of path variations to generate
    
    Returns:
    - List of paths, each path is a list of points
    """
    paths = []
    
    # Base path
    base_path = calculate_parking_path(start_pos, target_pos, car_params, slot_angle)
    paths.append(base_path)
    
    # Generate variations by adding slight angle variations and control point adjustments
    for i in range(1, num_routes):
        # Vary the car's initial angle slightly
        angle_variation = math.radians(5 * (i % 2 * 2 - 1))  # Alternate between +5° and -5°
        varied_car_params = copy.deepcopy(car_params)
        varied_car_params.angle = car_params.angle + angle_variation
        
        # Vary the approach distance slightly
        approach_variation = 0.9 + (i / num_routes) * 0.2  # 0.9 to 1.1
        
        # Calculate path with variations
        varied_path = calculate_parking_path(
            start_pos, 
            (target_pos[0], target_pos[1]), 
            varied_car_params, 
            slot_angle
        )
        
        paths.append(varied_path)
    
    return paths

def draw_parking_path(image, path_points, car_angle, color=(0, 255, 0), thickness=2):
    """Draw the parking path with direction indicators and better visualization."""
    if len(path_points) < 2:
        print("Warning: Not enough points to draw path")
        return
    
    print(f"Drawing path with {len(path_points)} points")
    
    # Create a copy of the path points for visualization
    vis_points = path_points.copy()
    
    # Draw the main path with gradient color for better depth perception
    for i in range(len(vis_points)-1):
        # Handle both (x,y) and (x,y,angle) formats
        pt1 = vis_points[i][:2]  # Get only x, y
        pt2 = vis_points[i+1][:2]  # Get only x, y
        
        # Ensure points are valid integers
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        
        # Create gradient color effect from start (lighter) to end (darker)
        progress = i / (len(vis_points) - 2)  # 0.0 to 1.0
        
        # Gradient from bright green to dark green
        g_value = min(255, int(255 - progress * 100))
        segment_color = (0, g_value, 0)
        
        # Draw thicker lines for better visibility
        segment_thickness = max(1, int(thickness * (1.0 - progress * 0.5)))
        cv.line(image, pt1, pt2, segment_color, segment_thickness)
    
    # Draw direction arrow indicators, reducing frequency for cleaner look
    for i in range(0, len(vis_points)-1, 8):
        pt1 = vis_points[i][:2]  # Get only x, y
        pt2 = vis_points[i+1][:2]  # Get only x, y
        
        # Draw arrow at middle of segment
        mid_point = ((pt1[0] + pt2[0])//2, (pt1[1] + pt2[1])//2)
        angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        arrow_length = 12
        arrow_point = (
            int(mid_point[0] + arrow_length * math.cos(angle)),
            int(mid_point[1] + arrow_length * math.sin(angle))
        )
        cv.arrowedLine(image, mid_point, arrow_point, (0, 200, 0), thickness+1)
    
    # Highlight starting and ending points with yellow lines
    if len(vis_points) > 0:
        # Start point (yellow line)
        start_pt = vis_points[0][:2]  # Get only x, y
        start_line_length = 30  # Length of the starting line
        start_line_end_x = start_pt[0] + start_line_length * math.cos(car_angle)  # Use the car's angle
        start_line_end_y = start_pt[1] + start_line_length * math.sin(car_angle)  # Use the car's angle
        cv.line(image, (int(start_pt[0]), int(start_pt[1])), (int(start_line_end_x), int(start_line_end_y)), (0, 255, 255), 2)  # Yellow starting line
        
        # End point (yellow line)
        end_pt = vis_points[-1][:2]  # Get only x, y
        end_line_length = 30  # Length of the ending line
        end_angle = vis_points[-1][2]  # Use the angle of the last point
        end_line_end_x = end_pt[0] + end_line_length * math.cos(end_angle)  # Use the angle of the last point
        end_line_end_y = end_pt[1] + end_line_length * math.sin(end_angle)  # Use the angle of the last point
        cv.line(image, (int(end_pt[0]), int(end_pt[1])), (int(end_line_end_x), int(end_line_end_y)), (0, 255, 255), 2)  # Yellow ending line

        # Print angles
        print(f"Starting Line Angle: {math.degrees(car_angle)}°")
        print(f"Ending Line Angle: {math.degrees(end_angle)}°")

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for car dragging."""
    car, image, car_params = param
    
    if event == cv.EVENT_LBUTTONDOWN:
        # Check if click is near car center
        car_x, car_y = car.pos
        length = car_params.length
        width = car_params.width
        if (car_x - length / 2 <= x <= car_x + length / 2) and (car_y - width / 2 <= y <= car_y + width / 2):
            car.dragging = True
            car.offset_x = car_x - x
            car.offset_y = car_y - y
    
    elif event == cv.EVENT_MOUSEMOVE:
        if car.dragging:
            # Update car position
            car.pos = (x + car.offset_x, y + car.offset_y)
            
            # Create a fresh copy of the image and redraw everything
            display_image = image.copy()
            draw_car(display_image, car.pos, car_params)  # Draw the car at the new position
            cv.imshow('Smart Parking Assistance', display_image)
    
    elif event == cv.EVENT_LBUTTONUP:
        car.dragging = False

def process_image(image_path, model, device, save_output=False):
    """Process image and provide parking assistance."""
    # Read image
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Initialize car parameters
    car_params = CarParameters()
    
    # Set initial car position at the bottom of the car
    height, width = image.shape[:2]
    initial_pos = (width // 2, height - car_params.length // 2)  # Adjusted to place at the bottom
    
    # Create draggable car instance
    car = DraggableCar(initial_pos, 0)  # Initialize with angle 0
    
    print(f"\nImage dimensions: {width}x{height}")
    print(f"Car starting position: {car.pos}")
    
    # Create window and set mouse callback
    cv.namedWindow('Smart Parking Assistance')
    cv.setMouseCallback('Smart Parking Assistance', mouse_callback, param=(car, image, car_params))
    
    # Draw the starting car at its initial position
    draw_car(image, car.pos, car_params)  # Draw the starting car

    # Main loop for interaction
    while True:
        display_image = image.copy()  # Create a fresh copy of the base image
        draw_car(display_image, car.pos, car_params)  # Draw the car
        cv.imshow('Smart Parking Assistance', display_image)
        
        # Handle keyboard input
        key = cv.waitKey(1) & 0xFF
        
        # Quit on 'q'
        if key == ord('q'):
            break
    
    cv.destroyAllWindows()

def generate_curved_parking_path(start_pos, target_pos, start_angle, target_angle, car_params):
    """Generate a curved path to the parking slot that is guided by the starting and ending lines."""
    path_points = []
    x1, y1 = start_pos
    x2, y2 = target_pos
    
    # Calculate distance to target
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Create control points for the curve
    control_distance = distance * 0.5  # Control point at 50% of the distance
    
    # Calculate tangent points based on the car's angle
    tangent_start_x = x1 + (car_params.length / 2) * math.cos(start_angle)
    tangent_start_y = y1 + (car_params.length / 2) * math.sin(start_angle)
    
    tangent_end_x = x2 - (car_params.length / 2) * math.cos(target_angle)
    tangent_end_y = y2 - (car_params.length / 2) * math.sin(target_angle)
    
    # Control point in the direction of the car's current heading
    control_x = (tangent_start_x + tangent_end_x) / 2
    control_y = (tangent_start_y + tangent_end_y) / 2 + control_distance * 0.5  # Adjust for curve height
    
    # Generate points along the Bezier curve
    num_steps = max(30, int(distance / 5))  # More steps for longer distances
    
    for t in np.linspace(0, 1, num_steps):
        # Quadratic Bezier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        bx = (1-t)**2 * tangent_start_x + 2*(1-t)*t * control_x + t**2 * tangent_end_x
        by = (1-t)**2 * tangent_start_y + 2*(1-t)*t * control_y + t**2 * tangent_end_y
        
        # Calculate angle at this point (tangent to curve)
        if t < 1.0:
            next_t = min(1.0, t + 0.05)
            next_bx = (1-next_t)**2 * tangent_start_x + 2*(1-next_t)*next_t * control_x + next_t**2 * tangent_end_x
            next_by = (1-next_t)**2 * tangent_start_y + 2*(1-next_t)*next_t * control_y + next_t**2 * tangent_end_y
            point_angle = math.atan2(next_by - by, next_bx - bx)
        else:
            point_angle = target_angle
        
        path_points.append((int(bx), int(by), point_angle))
    
    return path_points

def main():
    parser = argparse.ArgumentParser(description='Smart Parking Assistance System')
    parser.add_argument('--weights', default='dmpr_pretrained_weights.pth',
                      help='Path to model weights')
    parser.add_argument('--image', default='data/images/ps2.0/testing/outdoor-shadow/0035.jpg',
                      help='Path to input image')
    parser.add_argument('--save', action='store_true',
                      help='Save the output image')
    parser.add_argument('--cuda', action='store_true',
                      help='Use CUDA if available')
    
    args = parser.parse_args()
    
    # Setup model
    print("Loading model...")
    model, device = setup_model(args.weights, args.cuda)
    
    # Process image
    print(f"Processing image: {args.image}")
    process_image(args.image, model, device, args.save)

if __name__ == '__main__':
    main() 