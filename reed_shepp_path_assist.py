import cv2 as cv
import csv
from datetime import datetime
import argparse
import numpy as np
import math
import os
import json
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib

# Global configuration parameters for Reed-Shepp path generation
CONFIG = {
    # Turning radius: smaller value = tighter turns
    'TURNING_RADIUS': 30.0,
    # Distance scaling: how turning radius relates to distance between points
    'DISTANCE_SCALING': 4.0,  # Increased from 3.0 for better reversing maneuvers
    # Minimum turning radius to maintain vehicle constraints
    'MIN_TURNING_RADIUS': 25.0,  # Increased from 15.0 for more realistic car behavior
    # Maximum turning radius to prevent too large curves
    'MAX_TURNING_RADIUS': 50.0,
    # Number of path points generated along the path
    'PATH_POINTS': 80,  # Increased from 50 for smoother path visualization
    # Path color in BGR
    'PATH_COLOR': (0, 255, 0),  # Bright Green for Forward
    # Reverse Path color in BGR
    'REVERSE_COLOR': (0, 0, 255), # Bright Red for Reverse
    # Line thickness for path
    'PATH_THICKNESS': 2,
    # Color for steering indicators
    'INDICATOR_COLOR': (50, 200, 50),
    # Number of indicators to show
    'MAX_INDICATORS': 5,
    # Indicator line length
    'INDICATOR_LENGTH': 10,
    # Toggle debug visualization (additional path information)
    'DEBUG_VISUALIZATION': False,
    # Path smoothing factor (0-1): higher value = smoother path
    'PATH_SMOOTHING': 0.1,  # Reduced from 0.2 for better endpoint accuracy
    # Prefer reversing maneuvers when appropriate
    'FAVOR_REVERSING': True,
    # Show grid
    'SHOW_GRID': False,
    # Grid size in pixels
    'GRID_SIZE': 50,
    # Grid color
    'GRID_COLOR': (100, 100, 100),
}

# Function to append box parameters to a CSV file with a timestamp
def save_to_csv(red_box, green_box, filename='box_parameters.csv'):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a header if the file is empty
        if file.tell() == 0:
            writer.writerow(['Timestamp', 'Box', 'Top Left X', 'Top Left Y', 'Width', 'Height', 'Angle', 'Label'])
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Write parameters with timestamp and label
        writer.writerow([timestamp, 'Red', red_box.top_left[0], red_box.top_left[1], red_box.width, red_box.height, red_box.angle, red_box.label])
        writer.writerow([timestamp, 'Green', green_box.top_left[0], green_box.top_left[1], green_box.width, green_box.height, green_box.angle, green_box.label])

class DraggableBox:
    def __init__(self, top_left, color, width, height, label):
        self.top_left = top_left
        self.color = color
        self.width = width
        self.height = height
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0
        self.angle = 0  # Initialize angle
        self.label = label  # Add label attribute

    def draw(self, image):
        # Calculate the center of the box
        center_x = self.top_left[0] + self.width // 2
        center_y = self.top_left[1] + self.height // 2

        # Calculate rotation matrix
        rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), self.angle, 1.0)

        # Define the box corners
        corners = np.array([
            [self.top_left[0], self.top_left[1]],
            [self.top_left[0] + self.width, self.top_left[1]],
            [self.top_left[0] + self.width, self.top_left[1] + self.height],
            [self.top_left[0], self.top_left[1] + self.height]
        ])

        # Rotate the corners
        rotated_corners = cv.transform(np.array([corners]), rotation_matrix)[0]

        # Draw the rotated box
        cv.polylines(image, [np.int32(rotated_corners)], isClosed=True, color=self.color, thickness=2)

        # Calculate front and end coordinates at the center of the x-axis
        front_coord = ((rotated_corners[0][0] + rotated_corners[1][0]) // 2, (rotated_corners[0][1] + rotated_corners[1][1]) // 2)
        end_coord = ((rotated_corners[2][0] + rotated_corners[3][0]) // 2, (rotated_corners[2][1] + rotated_corners[3][1]) // 2)

        # Display coordinates, angle, and label
        text = f"Center: ({center_x}, {center_y}), Angle: {self.angle}°, {self.label}"
        cv.putText(image, text, (self.top_left[0] + 5, self.top_left[1] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)

        # Display front and end coordinates
        front_text = f"Front: ({int(front_coord[0])}, {int(front_coord[1])})"
        end_text = f"End: ({int(end_coord[0])}, {int(end_coord[1])})"
        cv.putText(image, front_text, (self.top_left[0] + 5, self.top_left[1] + 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)
        cv.putText(image, end_text, (self.top_left[0] + 5, self.top_left[1] + 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)

        # Draw a dot at the end coordinate
        cv.circle(image, (int(end_coord[0]), int(end_coord[1])), 5, self.color, -1)
    
    def get_center(self):
        return (self.top_left[0] + self.width // 2, self.top_left[1] + self.height // 2)
    
    def get_end_coordinates(self):
        center_x, center_y = self.get_center()
        angle_rad = math.radians(self.angle)
        rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), self.angle, 1.0)
        
        # Define the box corners
        corners = np.array([
            [self.top_left[0], self.top_left[1]],
            [self.top_left[0] + self.width, self.top_left[1]],
            [self.top_left[0] + self.width, self.top_left[1] + self.height],
            [self.top_left[0], self.top_left[1] + self.height]
        ])
        
        # Rotate the corners
        rotated_corners = cv.transform(np.array([corners]), rotation_matrix)[0]
        
        # Calculate front and end coordinates
        front_coord = ((rotated_corners[0][0] + rotated_corners[1][0]) // 2, (rotated_corners[0][1] + rotated_corners[1][1]) // 2)
        end_coord = ((rotated_corners[2][0] + rotated_corners[3][0]) // 2, (rotated_corners[2][1] + rotated_corners[3][1]) // 2)
        
        return front_coord, end_coord

    def is_inside(self, x, y):
        return self.top_left[0] <= x <= self.top_left[0] + self.width and self.top_left[1] <= y <= self.top_left[1] + self.height

    def start_drag(self, x, y):
        self.dragging = True
        self.offset_x = self.top_left[0] - x
        self.offset_y = self.top_left[1] - y

    def update_position(self, x, y):
        if self.dragging:
            self.top_left = (x + self.offset_x, y + self.offset_y)

    def stop_drag(self):
        self.dragging = False

    def rotate(self, delta_angle):
        self.angle = (self.angle + delta_angle) % 360


class DraggableControlPoint:
    def __init__(self, position, color=(255, 0, 255), radius=5):
        self.position = position
        self.color = color
        self.radius = radius
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0
    
    def draw(self, image):
        cv.circle(image, (int(self.position[0]), int(self.position[1])), self.radius, self.color, -1)
    
    def is_inside(self, x, y):
        return math.sqrt((self.position[0] - x)**2 + (self.position[1] - y)**2) <= self.radius * 2
    
    def start_drag(self, x, y):
        self.dragging = True
        self.offset_x = self.position[0] - x
        self.offset_y = self.position[1] - y
    
    def update_position(self, x, y):
        if self.dragging:
            self.position = (x + self.offset_x, y + self.offset_y)
    
    def stop_drag(self):
        self.dragging = False


class PathTrainer:
    def __init__(self, model_path="training_data/path_model.joblib"):
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        """Load a trained model if available."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                return True
            except:
                print("Error loading model. Training a new one may be required.")
        return False
    
    def train(self, data_path="training_data/path_data.json"):
        """Train a model on path data."""
        # Check if data exists
        if not os.path.exists(data_path):
            print("No training data found!")
            return False
        
        try:
            # Load data
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Filter for correct paths only
            correct_paths = [item for item in data if item['is_correct']]
            
            if len(correct_paths) < 5:  # Minimum threshold for training
                print(f"Not enough correct paths ({len(correct_paths)}/5 needed). Generate more paths and mark them as correct.")
                return False
            
            # Prepare features (car positions and angles) and targets (control points)
            X = []
            y = []
            
            for item in correct_paths:
                # Features: start position, angle, target position, angle
                green_box = item['green_box']
                red_box = item['red_box']
                
                # Skip if no control points
                if 'control_points' not in item or len(item['control_points']) != 2:
                    continue
                
                # Extract features
                feature = [
                    green_box['top_left'][0], green_box['top_left'][1], 
                    green_box['angle'], 
                    red_box['top_left'][0], red_box['top_left'][1], 
                    red_box['angle']
                ]
                
                # Extract targets (control points)
                target = [
                    item['control_points'][0]['x'], item['control_points'][0]['y'],
                    item['control_points'][1]['x'], item['control_points'][1]['y']
                ]
                
                X.append(feature)
                y.append(target)
            
            if len(X) < 5:
                print(f"Not enough complete paths with control points ({len(X)}/5 needed).")
                return False
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            # Save the model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            
            print(f"Model trained on {len(X)} examples and saved to {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_control_points(self, green_box, red_box):
        """Predict control points for path generation based on car positions."""
        if self.model is None:
            return None, None
        
        # Prepare features
        feature = [
            green_box.top_left[0], green_box.top_left[1], 
            green_box.angle, 
            red_box.top_left[0], red_box.top_left[1], 
            red_box.angle
        ]
        
        # Make prediction
        try:
            prediction = self.model.predict([feature])[0]
            
            # Create control points
            control_point1 = DraggableControlPoint((prediction[0], prediction[1]))
            control_point2 = DraggableControlPoint((prediction[2], prediction[3]))
            
            return control_point1, control_point2
        except:
            # Fall back to manual if prediction fails
            return None, None


def generate_path(green_box, red_box, image, control_point1=None, control_point2=None, use_ml=False, path_trainer=None, rotation_offset=0):
    """Generate a realistic car path from current position (green) to target position (red) using Reed-Shepp curves."""
    # Get the centers and end coordinates of both boxes
    _, green_end = green_box.get_end_coordinates()
    _, red_end = red_box.get_end_coordinates()
    
    # Extract points and orientation angles
    start_x, start_y = int(green_end[0]), int(green_end[1])
    target_x, target_y = int(red_end[0]), int(red_end[1])
    
    # Convert angles to radians (note: may need to adjust for coordinate system)
    start_angle = math.radians(green_box.angle % 360)
    goal_angle = math.radians(red_box.angle % 360)
    
    # Calculate distance between points
    distance = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
    
    # Reed-Shepp parameters
    # Turning radius (can be adjusted based on vehicle constraints)
    turning_radius = min(CONFIG['MAX_TURNING_RADIUS'], 
                         max(CONFIG['MIN_TURNING_RADIUS'], 
                             distance / CONFIG['DISTANCE_SCALING']))
    
    # Compute path using the rotation_offset
    path_points, path_segments = compute_reed_shepp_path(
        start_x, start_y, start_angle,
        target_x, target_y, goal_angle,
        turning_radius,
        rotation_offset
    )
    
    # Get the indices for different path segments
    fixed_segment_points = 10  # Same as in compute_reed_shepp_path
    remaining_points = CONFIG['PATH_POINTS'] - fixed_segment_points
    
    # Calculate forward segment points (same calculation as in compute_reed_shepp_path)
    if distance < 10:
        forward_ratio = 0.3
    else:
        forward_ratio = 0.4
    
    forward_points = int(remaining_points * forward_ratio)
    segment1_end_idx = fixed_segment_points
    segment2_end_idx = fixed_segment_points + forward_points
    
    # Draw the path with different colors for each segment
    # First segment (fixed start) - Green
    segment1_color = (0, 255, 0)  # Green in BGR
    for i in range(1, segment1_end_idx):
        pt1 = (int(path_points[i-1][0]), int(path_points[i-1][1]))
        pt2 = (int(path_points[i][0]), int(path_points[i][1]))
        cv.line(image, pt1, pt2, segment1_color, CONFIG['PATH_THICKNESS'])
    
    # Second segment (forward curve) - Blue
    segment2_color = (255, 165, 0)  # Blue in BGR
    for i in range(segment1_end_idx, segment2_end_idx):
        pt1 = (int(path_points[i-1][0]), int(path_points[i-1][1]))
        pt2 = (int(path_points[i][0]), int(path_points[i][1]))
        cv.line(image, pt1, pt2, segment2_color, CONFIG['PATH_THICKNESS'])
    
    # Third segment (reverse motion) - Red
    segment3_color = (0, 0, 255)  # Red in BGR
    for i in range(segment2_end_idx, len(path_points)):
        if i > 0:  # Skip first point which is already drawn
            pt1 = (int(path_points[i-1][0]), int(path_points[i-1][1]))
            pt2 = (int(path_points[i][0]), int(path_points[i][1]))
            cv.line(image, pt1, pt2, segment3_color, CONFIG['PATH_THICKNESS'])
    
    # Draw small circles at segment transition points
    cv.circle(image, (int(path_points[segment1_end_idx-1][0]), int(path_points[segment1_end_idx-1][1])), 
             5, (0, 255, 255), -1)  # Yellow dot at end of segment 1
    cv.circle(image, (int(path_points[segment2_end_idx-1][0]), int(path_points[segment2_end_idx-1][1])), 
             5, (0, 255, 255), -1)  # Yellow dot at end of segment 2
    
    # Draw small circles at key points to visualize the curve better
    for i in range(0, len(path_points), max(1, len(path_points) // 10)):
        pt = (int(path_points[i][0]), int(path_points[i][1]))
        cv.circle(image, pt, 3, (0, 165, 255), -1)  # Orange dots along path
    
    # Draw the final connection to the target explicitly
    if len(path_points) >= 2:
        # Draw a line from the last computed point to the exact target position
        last_pt = (int(path_points[-2][0]), int(path_points[-2][1]))
        target_pt = (target_x, target_y)
        cv.line(image, last_pt, target_pt, (0, 255, 255), CONFIG['PATH_THICKNESS'])

    # Draw path info
    cv.putText(image, f"Path Length: {distance:.1f} pixels, Turning Radius: {turning_radius:.1f}", 
              (10, image.shape[0] - 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw segment info legend
    cv.putText(image, "Segments: Green=Start, Blue=Forward, Red=Reverse", 
              (10, image.shape[0] - 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw instruction text for feedback
    cv.putText(image, "Press 'a' to approve path, 'r' to reject, '+/-' to adjust turning radius", 
              (10, image.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Return without control points (they're no longer used)
    return image, path_points, None, None


def path_length(path):
    """Calculate the total length of a path."""
    return sum(abs(seg[1]) for seg in path)


def path_score(path, favor_reversing=True):
    """
    Score a path based on length and complexity.
    
    Args:
        path: List of (segment_type, length) tuples
        favor_reversing: Whether to slightly prefer reversing paths when appropriate
    
    Returns:
        Score value (lower is better)
    """
    # Calculate total path length
    total_length = sum(abs(seg[1]) for seg in path)
    
    # Count segments that are straight lines
    straight_segments = sum(1 for seg in path if seg[0] == 'S')
    
    # Prefer paths with straight segments (more direct)
    straight_bonus = -0.2 if straight_segments > 0 else 0
    
    # Add a small bonus for shorter paths
    return total_length + straight_bonus


def compute_reed_shepp_path(start_x, start_y, start_angle, goal_x, goal_y, goal_angle, turning_radius, rotation_offset=0):
    """
    Create a realistic S-curve path specifically designed for reverse parking.
    The second segment (reverse) approaches at a -90 degree angle to the red car.
    Green car starts with a -90 degree angle adjustment from its orientation.
    Includes a fixed straight segment at the beginning and multiple straight segments during reverse.
    """
    # Calculate the distance between start and goal
    dx = goal_x - start_x
    dy = goal_y - start_y
    distance = math.sqrt(dx*dx + dy*dy)
    
    # Calculate angles - use exact angles from cars without modifications
    start_direction_angle = start_angle
    goal_direction_angle = goal_angle
    
    # Always print the angle information to the console/serial monitor
    print("\nPath Generation Angle Information:")
    print(f"  Green car angle: {math.degrees(start_angle):.1f}°") 
    print(f"  Red car angle: {math.degrees(goal_angle):.1f}°")
    
    # To maintain true perpendicularity:
    # 1. Get the car's angle in radians (including any rotation)
    car_angle_rad = start_angle
    if rotation_offset != 0:
        car_angle_rad += math.radians(rotation_offset)
        print(f"  Applied rotation offset: {rotation_offset}°")
    
    # 2. Calculate perpendicular angle by adding π/2 (90 degrees)
    # For clockwise perpendicular, subtract π/2
    adjusted_start_angle = car_angle_rad - math.pi/2
    
    print(f"  Car angle after rotation: {math.degrees(car_angle_rad):.1f}°")
    print(f"  Final path angle (perpendicular): {math.degrees(adjusted_start_angle):.1f}°")
    
    # The distance to place the intermediate control points
    intermediate_dist = min(distance * 0.4, 100)  # Reduced from 0.7 to 0.4, and max from 200 to 100

    # Add a fixed distance straight segment at the start
    fixed_start_distance = 50  # Fixed initial segment
    start_segment_x = start_x + fixed_start_distance * math.cos(adjusted_start_angle)
    start_segment_y = start_y + fixed_start_distance * math.sin(adjusted_start_angle)
    print(f"  Added fixed start segment: {fixed_start_distance} pixels at {math.degrees(adjusted_start_angle):.1f}°")
    
    # Adjust the approach angle for reverse parking
    approach_angle = goal_angle + math.radians(90)  
    print(f"  Approach angle: {math.degrees(approach_angle):.1f}° (red car angle + 90°)")
    
    # Calculate position to approach from using the adjusted angle
    if distance < 208:
        approach_distance = min(200, distance * 0.5)  # Increased from 100 to 200, and ratio from 0.3 to 0.5
        print(f"  Reduced approach distance: {approach_distance:.1f} (cars are close)")
    else:
        approach_distance = 208  # 208 pixels is the distance between the two cars in the image
    
    # Calculate multiple reverse segments
    num_reverse_segments = 10  # Number of reverse segments
    reverse_segment_length = approach_distance / num_reverse_segments
    
    # Calculate points for reverse segments
    reverse_points = []
    for i in range(num_reverse_segments + 1):  # +1 for the final point
        t = i / num_reverse_segments
        segment_x = goal_x + (approach_distance * (1 - t)) * math.cos(approach_angle)
        segment_y = goal_y + (approach_distance * (1 - t)) * math.sin(approach_angle)
        reverse_points.append((segment_x, segment_y))
    
    # Point for leaving the starting position - now starting from the end of the fixed segment
    leaving_start_x = start_segment_x + (intermediate_dist - fixed_start_distance) * math.cos(adjusted_start_angle)
    leaving_start_y = start_segment_y + (intermediate_dist - fixed_start_distance) * math.sin(adjusted_start_angle)
    
    # Calculate a good middle control point that creates a smooth S-curve
    mid_x = (leaving_start_x + reverse_points[0][0]) / 2
    mid_y = (leaving_start_y + reverse_points[0][1]) / 2
    
    # Add a slight offset to create a nice curve
    perpendicular_angle = adjusted_start_angle + math.pi/2
    curve_offset = min(distance * 0.15, 30)  # Reduced from 0.2 to 0.15, and max from 40 to 30
    mid_x += curve_offset * math.cos(perpendicular_angle)
    mid_y += curve_offset * math.sin(perpendicular_angle)
    
    # Generate points for the path
    points = []
    num_points = CONFIG['PATH_POINTS']
    
    # Add the fixed initial straight segment
    fixed_segment_points = 10
    for i in range(fixed_segment_points):
        t = i / (fixed_segment_points - 1)
        x = start_x + t * (start_segment_x - start_x)
        y = start_y + t * (start_segment_y - start_y)
        points.append((x, y))
    
    # Allocate remaining points
    remaining_points = num_points - fixed_segment_points
    forward_ratio = 0.25 if distance >= 10 else 0.2  # Reduced from 0.4/0.3 to 0.25/0.2
    forward_points = int(remaining_points * forward_ratio)
    reverse_points_count = remaining_points - forward_points
    
    # Add forward curve points
    for i in range(forward_points):
        t = i / (forward_points - 1)
        x = (1-t)*(1-t) * start_segment_x + 2*(1-t)*t * mid_x + t*t * reverse_points[0][0]
        y = (1-t)*(1-t) * start_segment_y + 2*(1-t)*t * mid_y + t*t * reverse_points[0][1]
        points.append((x, y))
    
    # Add reverse segment points
    points_per_segment = reverse_points_count // num_reverse_segments
    for i in range(len(reverse_points) - 1):
        start_pt = reverse_points[i]
        end_pt = reverse_points[i + 1]
        for j in range(points_per_segment):
            t = j / points_per_segment
            x = start_pt[0] + t * (end_pt[0] - start_pt[0])
            y = start_pt[1] + t * (end_pt[1] - start_pt[1])
            points.append((x, y))
    
    # Add final approach to goal
    points.append((goal_x, goal_y))
    
    # Create segments list for compatibility
    segments = [('S', distance)]
    
    return points, segments


def smooth_path(points, smoothing_factor=0.2, iterations=5):
    """
    Apply an iterative smoothing filter to the path points, keeping endpoints fixed.
    
    Args:
        points: List of (x, y) tuples
        smoothing_factor: How much to smooth (0-1), often called alpha.
        iterations: Number of smoothing passes.
    
    Returns:
        Smoothed list of points
    """
    if len(points) <= 2 or smoothing_factor <= 0 or iterations <= 0:
        return points
    
    # Limit smoothing factor
    alpha = max(0, min(1, smoothing_factor))
    # Beta controls how much the original position is retained
    beta = 1.0 - alpha 
    
    smoothed = list(points) # Make a copy to modify

    for _ in range(iterations):
        # Create a temporary list to store results of this iteration
        iteration_smoothed = [smoothed[0]] # Keep first point fixed
        
        # Apply smoothing filter to internal points
        for i in range(1, len(points) - 1):
            prev_x, prev_y = smoothed[i-1]
            curr_x, curr_y = smoothed[i]
            next_x, next_y = smoothed[i+1]
            
            # Weighted average: pull point towards the average of its neighbors
            smooth_x = beta * curr_x + alpha * (prev_x + next_x) / 2
            smooth_y = beta * curr_y + alpha * (prev_y + next_y) / 2
            
            iteration_smoothed.append((smooth_x, smooth_y))
        
        iteration_smoothed.append(smoothed[-1]) # Keep last point fixed
        smoothed = iteration_smoothed # Update path for next iteration
    
    return smoothed


def compute_lsl_path(x, y, phi):
    """Compute Left-Straight-Left path."""
    # Compute parameters for LSL
    u, t = x + math.sin(phi), y - 1 - math.cos(phi)
    
    # Compute the central straight segment length
    v = math.sqrt(u*u + t*t)
    if v < 2:  # Not feasible
        return None
    
    # Compute angles
    alpha = math.atan2(t, u)
    beta = math.acos(2 / v)
    
    # Calculate segment lengths
    t1 = (alpha - beta) % (2 * math.pi)
    q = v * math.sin(beta)  # Straight length
    t2 = (phi - alpha - beta) % (2 * math.pi)
    
    # Ensure t1, t2 are in [-pi, pi]
    t1 = (t1 + math.pi) % (2 * math.pi) - math.pi
    t2 = (t2 + math.pi) % (2 * math.pi) - math.pi
    
    return [('L', t1), ('S', q), ('L', t2)]


def compute_rsr_path(x, y, phi):
    """Compute Right-Straight-Right path."""
    # For RSR, reflect the problem
    x_r, y_r = x, -y
    phi_r = -phi
    
    # Use LSL solution with the reflected problem
    path = compute_lsl_path(x_r, y_r, phi_r)
    
    if not path:
        return None
    
    # Convert back to RSR by changing L to R and negating angles
    return [(('R' if seg[0] == 'L' else seg[0]), 
             (-seg[1] if seg[0] == 'L' else seg[1])) 
            for seg in path]


def compute_lsr_path(x, y, phi):
    """Compute Left-Straight-Right path."""
    # Compute parameters for LSR
    u, t = x - math.sin(phi), y - 1 + math.cos(phi)
    
    # Compute central segment length
    v = math.sqrt(u*u + t*t)
    if v < 2:  # Not feasible
        return None
    
    # Compute angles
    alpha = math.atan2(t, u)
    beta = math.acos(2 / v)
    
    # Calculate segment lengths
    t1 = (alpha + beta) % (2 * math.pi)
    q = v * math.sin(beta)  # Straight length
    t2 = (phi - alpha + beta) % (2 * math.pi)
    
    # Ensure t1, t2 are in [-pi, pi]
    t1 = (t1 + math.pi) % (2 * math.pi) - math.pi
    t2 = (t2 + math.pi) % (2 * math.pi) - math.pi
    
    return [('L', t1), ('S', q), ('R', -t2)]


def compute_rsl_path(x, y, phi):
    """Compute Right-Straight-Left path."""
    # For RSL, reflect the problem
    x_r, y_r = x, -y
    phi_r = -phi
    
    # Use LSR solution with the reflected problem
    path = compute_lsr_path(x_r, y_r, phi_r)
    
    if not path:
        return None
    
    # Convert back to RSL by changing L to R and R to L and negating angles
    return [(('R' if seg[0] == 'L' else 'L' if seg[0] == 'R' else seg[0]), 
             (-seg[1] if seg[0] != 'S' else seg[1])) 
            for seg in path]


def compute_ccc_path(x, y, phi):
    """Compute CCC (Curve-Curve-Curve) paths with all variants."""
    # CCC has 4 possible types: LRL, RLR, LRL (backwards), RLR (backwards)
    
    # Calculate distance to goal
    d = math.sqrt(x*x + y*y)
    
    # Compute LRL path
    lrl = compute_lrl_path(x, y, phi)
    
    # Compute RLR path
    rlr = compute_rlr_path(x, y, phi)
    
    # Find the shortest valid path
    paths = [p for p in [lrl, rlr] if p]
    if not paths:
        return None
    
    return min(paths, key=lambda p: path_length(p))


def compute_lrl_path(x, y, phi):
    """Compute Left-Right-Left path."""
    # Compute parameters for LRL
    u, v = x - math.sin(phi), y - 1 + math.cos(phi)
    
    # Calculate distance for feasibility check
    d = u*u + v*v
    if d < 4:  # Feasibility threshold
        # Find the middle turning angle
        A = math.atan2(v, u)
        B = math.acos((d - 4) / 4)  # The acos argument must be in [-1, 1]
        if abs((d - 4) / 4) > 1:
            return None
        
        # Calculate segment lengths
        t1 = (A + B) % (2 * math.pi)
        p = 2 * B  # Middle turn angle
        t2 = (phi - t1 + p) % (2 * math.pi)
        
        # Ensure angles are in [-pi, pi]
        t1 = (t1 + math.pi) % (2 * math.pi) - math.pi
        p = (p + math.pi) % (2 * math.pi) - math.pi
        t2 = (t2 + math.pi) % (2 * math.pi) - math.pi
        
        return [('L', t1), ('R', -p), ('L', t2)]
    
    return None


def compute_rlr_path(x, y, phi):
    """Compute Right-Left-Right path."""
    # For RLR, reflect the problem
    x_r, y_r = x, -y
    phi_r = -phi
    
    # Use LRL solution with the reflected problem
    path = compute_lrl_path(x_r, y_r, phi_r)
    
    if not path:
        return None
    
    # Convert back to RLR by changing L to R and R to L and negating angles
    return [(('R' if seg[0] == 'L' else 'L' if seg[0] == 'R' else seg[0]), 
             (-seg[1])) 
            for seg in path]


def compute_ccsc_path(x, y, phi):
    """Compute CCSC (Curve-Curve-Straight-Curve) path (simplified)."""
    # This is a placeholder for the complex CCSC path computation
    # In a complete implementation, we would check various configurations
    
    # For now, return None to indicate this path type isn't computed
    return None


def compute_cscc_path(x, y, phi):
    """Compute CSCC (Curve-Straight-Curve-Curve) path (simplified)."""
    # This is a placeholder for the complex CSCC path computation
    # In a complete implementation, we would check various configurations
    
    # For now, return None to indicate this path type isn't computed
    return None


def compute_ccscc_path(x, y, phi):
    """Compute CCSCC (Curve-Curve-Straight-Curve-Curve) path (simplified)."""
    # This is a placeholder for the complex CCSCC path computation
    # In a complete implementation, we would check various configurations
    
    # For now, return None to indicate this path type isn't computed
    return None


def simple_dubins_path(start_x, start_y, start_angle, goal_x, goal_y, goal_angle, turning_radius):
    """
    Fallback path generator: returns a straight line from start to goal.
    Indicates Reed-Shepp failure.
    """
    print("WARNING: Reed-Shepp path computation failed. Falling back to straight line.")
    # Return only start and end points to signify failure clearly
    return [(start_x, start_y), (goal_x, goal_y)]


def generate_points_along_path(path_segments, start_x, start_y, start_angle, turning_radius):
    """Generate a sequence of points along the computed Reed-Shepp path."""
    total_points = CONFIG['PATH_POINTS']
    
    # Calculate total actual path length for point distribution
    total_actual_length = 0
    for segment_type, length in path_segments:
        # Length is scaled (angle for turns, distance/radius for straights)
        if segment_type == 'S':
            total_actual_length += abs(length) * turning_radius # Actual distance
        else: # L or R
            total_actual_length += abs(length) * turning_radius # Arc length = angle * radius

    # Handle zero-length path case
    if total_actual_length < 1e-6:
        return [(start_x, start_y)]

    x, y, theta = start_x, start_y, start_angle
    points = [(x, y)]
    points_generated = 1

    for segment_type, length in path_segments:
        # Calculate actual length of this segment
        actual_segment_length = 0
        if segment_type == 'S':
            actual_segment_length = abs(length) * turning_radius
        else: # L or R
            actual_segment_length = abs(length) * turning_radius # Angle * Radius

        if actual_segment_length < 1e-6: # Skip tiny segments
             continue

        # Number of points for this segment (proportional to its length contribution)
        # Ensure at least 2 points for any non-zero segment
        # Distribute remaining points proportionally, ensuring we don't exceed total_points
        if total_actual_length > 1e-6:
            points_for_segment = (actual_segment_length / total_actual_length) * (total_points - points_generated)
        else:
            points_for_segment = 0 # Avoid division by zero if total length is somehow near zero
            
        segment_points_count = max(2, int(points_for_segment)) if points_generated < total_points else 0
        
        # Avoid generating more points than needed overall
        segment_points_count = min(segment_points_count, total_points - points_generated)


        direction = 1 if length >= 0 else -1
        segment_length_scaled = abs(length) # Scaled length (distance/radius or angle)

        # Segment start state
        segment_start_x, segment_start_y = x, y 
        segment_start_theta = theta 

        for i in range(1, segment_points_count + 1):
            interp_factor = i / segment_points_count # Interpolation factor along segment: 0 to 1

            # Calculate state at interp_factor * segment_length
            current_segment_progress = direction * segment_length_scaled * interp_factor

            if segment_type == 'S':
                # Straight segment: move along segment_start_theta
                dist = current_segment_progress * turning_radius
                x = segment_start_x + dist * math.cos(segment_start_theta)
                y = segment_start_y + dist * math.sin(segment_start_theta)
                theta = segment_start_theta # Angle doesn't change
            
            elif segment_type == 'L':
                # Left turn: Angle changes by current_segment_progress (which is an angle)
                delta_theta = current_segment_progress
                # Calculate position using standard circle equations relative to segment start
                # Center of rotation is R distance perpendicular left to the starting direction
                center_x = segment_start_x - turning_radius * math.sin(segment_start_theta)
                center_y = segment_start_y + turning_radius * math.cos(segment_start_theta)
                # New position relative to center
                x = center_x + turning_radius * math.sin(segment_start_theta + delta_theta)
                y = center_y - turning_radius * math.cos(segment_start_theta + delta_theta)
                theta = segment_start_theta + delta_theta # Update angle

            elif segment_type == 'R':
                # Right turn: Angle changes by -current_segment_progress
                delta_theta = -current_segment_progress # Note the minus sign for right turn angle change
                # Center of rotation is R distance perpendicular right to the starting direction
                center_x = segment_start_x + turning_radius * math.sin(segment_start_theta)
                center_y = segment_start_y - turning_radius * math.cos(segment_start_theta)
                 # New position relative to center
                x = center_x - turning_radius * math.sin(segment_start_theta + delta_theta)
                y = center_y + turning_radius * math.cos(segment_start_theta + delta_theta)
                theta = segment_start_theta + delta_theta # Update angle

            # Ensure angle stays within [-pi, pi] or [0, 2pi] if needed
            # theta = (theta + math.pi) % (2 * math.pi) - math.pi 
                
                points.append((x, y))
            points_generated += 1


        # Update state (x, y, theta) for the start of the next segment 
        # based on the *precise* end state of the current segment, 
        # avoiding reliance on the last interpolated point which might have small errors.
        final_segment_progress = direction * segment_length_scaled
        if segment_type == 'S':
             dist = final_segment_progress * turning_radius
             x = segment_start_x + dist * math.cos(segment_start_theta)
             y = segment_start_y + dist * math.sin(segment_start_theta)
             theta = segment_start_theta
        elif segment_type == 'L':
             delta_theta = final_segment_progress
             center_x = segment_start_x - turning_radius * math.sin(segment_start_theta)
             center_y = segment_start_y + turning_radius * math.cos(segment_start_theta)
             x = center_x + turning_radius * math.sin(segment_start_theta + delta_theta)
             y = center_y - turning_radius * math.cos(segment_start_theta + delta_theta)
             theta = segment_start_theta + delta_theta
        elif segment_type == 'R':
             delta_theta = -final_segment_progress
             center_x = segment_start_x + turning_radius * math.sin(segment_start_theta)
             center_y = segment_start_y - turning_radius * math.cos(segment_start_theta)
             x = center_x - turning_radius * math.sin(segment_start_theta + delta_theta)
             y = center_y + turning_radius * math.cos(segment_start_theta + delta_theta)
             theta = segment_start_theta + delta_theta

        # Update the last point generated for this segment accurately
        if i >= 1 and len(points) > 0: # Check if any points were added for this segment
             points[-1] = (x, y)


        if points_generated >= total_points:
            break # Stop if we have generated enough points

    # The compute_reed_shepp_path function will handle final endpoint fixing after smoothing
    return points


def save_path_data(green_box, red_box, path_points, is_correct, filename='path_data.json'):
    """Save path data for training."""
    # Create data directory if it doesn't exist
    data_dir = 'training_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, filename)
    
    # Prepare data to save
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'is_correct': is_correct,
        'green_box': {
            'top_left': green_box.top_left,
            'angle': green_box.angle,
            'width': green_box.width,
            'height': green_box.height
        },
        'red_box': {
            'top_left': red_box.top_left,
            'angle': red_box.angle,
            'width': red_box.width,
            'height': red_box.height
        },
        'path_points': path_points,
        'path_type': 'reed_shepp'  # Add path type information
    }
    
    # Load existing data if file exists
    existing_data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            # If file is corrupted, start fresh
            existing_data = []
    
    # Append new data
    existing_data.append(data)
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    status = "correct" if is_correct else "incorrect"
    print(f"Reed-Shepp path data saved as {status} example at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def draw_grid(image):
    """Draw a grid overlay on the image."""
    if not CONFIG['SHOW_GRID']:
        return image
        
    grid_image = image.copy()
    h, w = grid_image.shape[:2]
    grid_size = CONFIG['GRID_SIZE']
    grid_color = CONFIG['GRID_COLOR']
    
    # Draw vertical lines
    for x in range(0, w, grid_size):
        cv.line(grid_image, (x, 0), (x, h), grid_color, 1)
        # Add coordinates every 100 pixels
        if x % 100 == 0:
            cv.putText(grid_image, str(x), (x + 5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw horizontal lines
    for y in range(0, h, grid_size):
        cv.line(grid_image, (0, y), (w, y), grid_color, 1)
        # Add coordinates every 100 pixels
        if y % 100 == 0:
            cv.putText(grid_image, str(y), (5, y + 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return grid_image


def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for dragging boxes."""
    red_box, green_box, image, _, edit_control_points = param
    
    if event == cv.EVENT_LBUTTONDOWN:
        # Check if a box is clicked
        if red_box.is_inside(x, y):
            red_box.start_drag(x, y)
        elif green_box.is_inside(x, y):
            green_box.start_drag(x, y)
    
    elif event == cv.EVENT_MOUSEMOVE:
        # Update box positions
        red_box.update_position(x, y)
        green_box.update_position(x, y)
    
    elif event == cv.EVENT_LBUTTONUP:
        # Stop dragging boxes
        red_box.stop_drag()
        green_box.stop_drag()
    
    # Redraw the image
    display_image = image.copy()
    
    # Apply grid if enabled
    if CONFIG['SHOW_GRID']:
        display_image = draw_grid(display_image)
    
    # Generate and draw the Reed-Shepp path
    display_image, _, _, _ = generate_path(green_box, red_box, display_image, rotation_offset=0)
    
    # Draw boxes
    red_box.draw(display_image)
    green_box.draw(display_image)
    
    cv.imshow('Image Display', display_image)


def process_image(image_path):
    """Load and display the image with draggable and rotatable boxes."""
    # Read image
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Define box dimensions and colors
    box_width, box_height = 116, 208
    red_color = (0, 0, 255)  # Red in BGR
    green_color = (0, 255, 0)  # Green in BGR

    # Initialize draggable boxes with labels
    # Set red_box initial position and angle based on user provided CSV data
    red_box_start_pos = (503, 378)
    red_box_start_angle = 90
    red_box = DraggableBox(red_box_start_pos, red_color, box_width, box_height, "Target Position")
    red_box.angle = red_box_start_angle # Set the initial angle
    
    # Set green_box initial position based on the user provided CSV data
    green_box_start_pos = (242, 180)
    green_box = DraggableBox(green_box_start_pos, green_color, box_width, box_height, "Current Position")

    # Path visualization variables
    show_path = False
    path_points = []
    
    # Track rotation offset for the green car
    # When the car is manually rotated, this will be set to 5, otherwise 0
    rotation_offset = 0
    
    # Set up window and mouse callback
    cv.namedWindow('Image Display')
    cv.setMouseCallback('Image Display', mouse_callback, 
                      param=(red_box, green_box, image.copy(), None, False))

    # Display the image
    display_image = image.copy()
    if CONFIG['SHOW_GRID']:
        display_image = draw_grid(display_image)
    red_box.draw(display_image)
    green_box.draw(display_image)
    cv.imshow('Image Display', display_image)
    
    # Show instructions
    print("\nReed-Shepp Path Assist Controls:")
    print("- 'z'/'x': Rotate green box counter-clockwise/clockwise")
    print("- 'c'/'v': Rotate red box counter-clockwise/clockwise")
    print("- 'p': Generate Reed-Shepp path")
    print("- 'a': Approve path as correct")
    print("- 'r': Reject path as incorrect")
    print("- '+'/'-': Increase/decrease turning radius")
    print("- 'd': Toggle debug visualization")
    print("- 'g': Toggle grid display")
    print("- Space: Save box parameters to CSV")
    print("- 'q': Quit\n")
    
    # Wait for 'q' to quit
    while True:
        key = cv.waitKey(1) & 0xFF
        
        # Rotate red box with 'c' and 'v' keys
        if key == ord('c'):
            red_box.rotate(-5)
            show_path = False
        elif key == ord('v'):
            red_box.rotate(5)
            show_path = False

        # Rotate green box with 'z' and 'x' keys
        elif key == ord('z'):
            green_box.rotate(-5)
            rotation_offset = 5  # Keep same direction as box rotation
            show_path = False
            print("Rotating green box -5°, path adjusting to maintain perpendicularity")
        elif key == ord('x'):
            green_box.rotate(5)
            rotation_offset = -5  # Keep same direction as box rotation
            show_path = False
            print("Rotating green box +5°, path adjusting to maintain perpendicularity")

        # Save parameters to CSV when spacebar is pressed
        elif key == ord(' '):
            save_to_csv(red_box, green_box)
            print(f"Parameters saved to box_parameters.csv at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generate and show path when 'p' is pressed
        elif key == ord('p'):
            show_path = True
            display_image = image.copy()
            # Pass the rotation_offset to the generate_path function
            display_image, path_points, _, _ = generate_path(green_box, red_box, display_image, rotation_offset=rotation_offset)
            red_box.draw(display_image)
            green_box.draw(display_image)
            cv.imshow('Image Display', display_image)
            print("Generating Reed-Shepp path...")
        
        # Approve the path
        elif key == ord('a') and show_path:
            if path_points:
                save_path_data(green_box, red_box, path_points, True)
                print("Path approved and saved as a correct example.")
        
        # Reject the path
        elif key == ord('r') and show_path:
            if path_points:
                save_path_data(green_box, red_box, path_points, False)
                print("Path rejected and saved as an incorrect example.")
        
        # Increase turning radius
        elif key == ord('+') or key == ord('='):  # '+' is often Shift+= on keyboards
            CONFIG['MIN_TURNING_RADIUS'] += 5
            CONFIG['MAX_TURNING_RADIUS'] += 5
            print(f"Increased turning radius range: {CONFIG['MIN_TURNING_RADIUS']}-{CONFIG['MAX_TURNING_RADIUS']}")
            if show_path:
                display_image = image.copy()
                display_image, path_points, _, _ = generate_path(green_box, red_box, display_image, rotation_offset=rotation_offset)
                red_box.draw(display_image)
                green_box.draw(display_image)
                cv.imshow('Image Display', display_image)
        
        # Decrease turning radius
        elif key == ord('-') or key == ord('_'):  # '_' is often Shift+- on keyboards
            CONFIG['MIN_TURNING_RADIUS'] = max(5, CONFIG['MIN_TURNING_RADIUS'] - 5)
            CONFIG['MAX_TURNING_RADIUS'] = max(10, CONFIG['MAX_TURNING_RADIUS'] - 5)
            print(f"Decreased turning radius range: {CONFIG['MIN_TURNING_RADIUS']}-{CONFIG['MAX_TURNING_RADIUS']}")
            if show_path:
                display_image = image.copy()
                display_image, path_points, _, _ = generate_path(green_box, red_box, display_image, rotation_offset=rotation_offset)
                red_box.draw(display_image)
                green_box.draw(display_image)
                cv.imshow('Image Display', display_image)
        
        # Toggle debug visualization
        elif key == ord('d'):
            CONFIG['DEBUG_VISUALIZATION'] = not CONFIG['DEBUG_VISUALIZATION']
            print(f"Debug visualization: {'ON' if CONFIG['DEBUG_VISUALIZATION'] else 'OFF'}")
            if show_path:
                display_image = image.copy()
                if CONFIG['SHOW_GRID']:
                    display_image = draw_grid(display_image)
                display_image, path_points, _, _ = generate_path(green_box, red_box, display_image, rotation_offset=rotation_offset)
                red_box.draw(display_image)
                green_box.draw(display_image)
                cv.imshow('Image Display', display_image)
        
        # Toggle grid display
        elif key == ord('g'):
            CONFIG['SHOW_GRID'] = not CONFIG['SHOW_GRID']
            print(f"Grid display: {'ON' if CONFIG['SHOW_GRID'] else 'OFF'}")
            display_image = image.copy()
            if CONFIG['SHOW_GRID']:
                display_image = draw_grid(display_image)
            if show_path:
                display_image, path_points, _, _ = generate_path(green_box, red_box, display_image, rotation_offset=rotation_offset)
                red_box.draw(display_image)
                green_box.draw(display_image)
                cv.imshow('Image Display', display_image)
                
        # Update display when anything changes
        elif show_path:
            display_image = image.copy()
            if CONFIG['SHOW_GRID']:
                display_image = draw_grid(display_image)
            display_image, path_points, _, _ = generate_path(green_box, red_box, display_image, rotation_offset=rotation_offset)
            red_box.draw(display_image)
            green_box.draw(display_image)
            cv.imshow('Image Display', display_image)
        
        # Quit on 'q'
        elif key == ord('q'):
            break

    cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Smart Parking Path Assistant')
    parser.add_argument('--image', default='data/images/ps2.0/testing/outdoor-normal daylight/193.jpg',
                      help='Path to input image')
    
    args = parser.parse_args()
    
    # Process image
    print(f"Loading image: {args.image}")
    process_image(args.image)

if __name__ == '__main__':
    main() 