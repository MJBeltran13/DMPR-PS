import cv2 as cv
import csv
from datetime import datetime
import argparse
import numpy as np
import math
import os
import json

# Global configuration parameters
CONFIG = {
    'PATH_POINTS': 80,  # Number of path points generated along the path
    'PATH_COLOR': (0, 255, 0),  # Bright Green for Forward
    'REVERSE_COLOR': (0, 0, 255), # Bright Red for Reverse
    'PATH_THICKNESS': 2,
    'INDICATOR_COLOR': (50, 200, 50),
    'MAX_INDICATORS': 5,
    'INDICATOR_LENGTH': 10,
    'DEBUG_VISUALIZATION': False,
    'SHOW_GRID': False,
    'GRID_SIZE': 50,
    'GRID_COLOR': (100, 100, 100),
}

class PathGenerator:
    def __init__(self):
        self.turning_radius = 100  # Minimum turning radius in pixels
        self.distance_scaling = 1.0  # Scale factor for path distances
        self.path_points = 80  # Number of points to generate along the path
        self.forward_color = (0, 255, 0)  # Green for forward movement
        self.reverse_color = (0, 0, 255)  # Red for reverse movement
        self.indicator_color = (50, 200, 50)  # Light green for direction indicators
        self.max_indicators = 5  # Maximum number of direction indicators to show
        self.indicator_length = 10  # Length of direction indicators in pixels
        self.parking_type = "reverse"  # Default to reverse parking

    def generate_parking_path(self, current_box, target_box):
        """Generate a realistic reverse parking path from current position to target position."""
        # Extract positions and angles
        start_pos = current_box['center']
        end_pos = target_box['center']
        start_angle = current_box['angle']
        target_angle = target_box['angle']

        # Calculate the direct distance between start and end
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        direct_distance = math.sqrt(dx*dx + dy*dy)

        if self.parking_type == "reverse":
            return self._generate_reverse_parking_path(start_pos, end_pos, start_angle, target_angle, direct_distance)
        else:
            return self._generate_forward_parking_path(start_pos, end_pos, start_angle, target_angle, direct_distance)

    def _generate_reverse_parking_path(self, start_pos, end_pos, start_angle, target_angle, direct_distance):
        """Generate a specialized reverse parking path with multiple segments."""
        path_points = []
        num_points = self.path_points
        
        # Calculate angle difference
        angle_diff = target_angle - start_angle
        # Normalize angle difference to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
            
        # For reverse parking, we'll create a path with distinct segments:
        # 1. Initial positioning (forward)
        # 2. Turning to align with parking space (forward)
        # 3. Reverse movement into the parking space
        # 4. Final adjustments
        
        # Calculate approach position - a point offset from target
        # This is where the car will start reversing from
        approach_distance = direct_distance * 0.6
        approach_angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        
        # Calculate approach position (90 degrees to target angle)
        target_angle_rad = math.radians(target_angle)
        perpendicular_angle = target_angle_rad + math.pi/2
        
        # Position the approach point to the side and in front of the target
        approach_x = end_pos[0] + math.cos(perpendicular_angle) * self.turning_radius * 1.5
        approach_y = end_pos[1] + math.sin(perpendicular_angle) * self.turning_radius * 1.5
        
        # Make the approach point closer to start for a more natural path
        approach_x = (approach_x + start_pos[0]) / 2
        approach_y = (approach_y + start_pos[1]) / 2
        
        # Segment 1: Start to approach position (forward movement)
        segment1_points = int(num_points * 0.3)
        for i in range(segment1_points):
            t = i / (segment1_points - 1)
            # Linear interpolation from start to approach
            x = start_pos[0] * (1-t) + approach_x * t
            y = start_pos[1] * (1-t) + approach_y * t
            
            # Gradually change angle to face the approach angle
            angle = start_angle * (1-t) + (target_angle + 90) * t
            
            path_points.append({
                'x': x,
                'y': y,
                'angle': angle,
                'action': 'forward'
            })
        
        # Segment 2: Curve into position for reversing
        # This is a quarter circle to align the car for reverse
        segment2_points = int(num_points * 0.2)
        last_point = path_points[-1]
        center_of_turn_x = approach_x - math.cos(target_angle_rad) * self.turning_radius
        center_of_turn_y = approach_y - math.sin(target_angle_rad) * self.turning_radius
        
        for i in range(segment2_points):
            t = i / (segment2_points - 1)
            angle_radians = math.radians(target_angle + 90) - t * math.pi/2
            
            # Calculate position on the arc
            x = center_of_turn_x + math.cos(angle_radians) * self.turning_radius
            y = center_of_turn_y + math.sin(angle_radians) * self.turning_radius
            
            # Calculate car angle (tangent to the circle)
            angle = target_angle + 90 - t * 90
            
            path_points.append({
                'x': x,
                'y': y,
                'angle': angle,
                'action': 'forward'
            })
        
        # Segment 3: Reverse into parking space
        segment3_points = int(num_points * 0.5)
        last_point = path_points[-1]
        
        for i in range(segment3_points):
            t = i / (segment3_points - 1)
            
            # Linear interpolation from last position to target
            x = last_point['x'] * (1-t) + end_pos[0] * t
            y = last_point['y'] * (1-t) + end_pos[1] * t
            
            # Gradually change angle to target angle
            angle = last_point['angle'] * (1-t) + target_angle * t
            
            path_points.append({
                'x': x,
                'y': y,
                'angle': angle,
                'action': 'reverse'
            })
        
        return path_points

    def _generate_forward_parking_path(self, start_pos, end_pos, start_angle, target_angle, direct_distance):
        """Generate a smooth forward parking path."""
        path_points = []
        num_points = self.path_points
        
        # Calculate angle difference
        angle_diff = target_angle - start_angle
        # Normalize angle difference to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
            
        # Generate smooth path
        for i in range(num_points):
            t = i / (num_points - 1)  # Parameter from 0 to 1
            
            # Smooth position interpolation
            x = start_pos[0] + (end_pos[0] - start_pos[0]) * t
            y = start_pos[1] + (end_pos[1] - start_pos[1]) * t
            
            # Smooth angle interpolation
            angle = start_angle + angle_diff * t
            
            # Add some curvature to make it more realistic
            # Use sine function to create a smooth curve
            curve_factor = math.sin(t * math.pi) * 0.2  # 0.2 controls the amount of curvature
            x += curve_factor * direct_distance * math.cos(math.radians(angle + 90))
            y += curve_factor * direct_distance * math.sin(math.radians(angle + 90))
            
            # Determine if this segment is forward or reverse
            # Generally forward if moving towards target, reverse if moving away
            is_forward = True
            if i > 0:
                prev_point = path_points[-1]
                prev_dist = math.sqrt((prev_point['x'] - end_pos[0])**2 + (prev_point['y'] - end_pos[1])**2)
                curr_dist = math.sqrt((x - end_pos[0])**2 + (y - end_pos[1])**2)
                is_forward = curr_dist <= prev_dist
            
            path_points.append({
                'x': x,
                'y': y,
                'angle': angle,
                'action': 'forward' if is_forward else 'reverse'
            })
            
        return path_points

    def draw_path(self, image, path_points):
        """Draw the generated path on the image."""
        if not path_points:
            return image

        # Create a copy of the image to draw on
        result = image.copy()
        
        # Draw the path segments
        for i in range(len(path_points) - 1):
            pt1 = (int(path_points[i]['x']), int(path_points[i]['y']))
            pt2 = (int(path_points[i+1]['x']), int(path_points[i+1]['y']))
            
            # Choose color based on action
            color = self.forward_color if path_points[i]['action'] == 'forward' else self.reverse_color
            
            # Draw the path segment
            cv.line(result, pt1, pt2, color, 2)
            
            # Draw direction indicators at regular intervals
            if i % (len(path_points) // self.max_indicators) == 0:
                # Calculate the angle of the segment
                angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
                
                # Calculate the midpoint
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                
                # Calculate the indicator endpoints
                indicator_angle = angle + (math.pi/2 if path_points[i]['action'] == 'forward' else -math.pi/2)
                end_x = int(mid_x + self.indicator_length * math.cos(indicator_angle))
                end_y = int(mid_y + self.indicator_length * math.sin(indicator_angle))
                
                # Draw the indicator
                cv.line(result, (mid_x, mid_y), (end_x, end_y), self.indicator_color, 2)

        return result

    def generate_parking_instructions(self, path_points, target_box):
        """Generate step-by-step parking instructions based on the path."""
        if not path_points:
            return {"instructions": ["No path available"], "difficulty": "Unknown"}

        instructions = []
        total_distance = 0
        total_angle_change = 0
        max_angle_change = 0
        current_angle = path_points[0]['angle']
        
        # Calculate total path metrics
        for i in range(len(path_points) - 1):
            pt1 = path_points[i]
            pt2 = path_points[i + 1]
            
            # Calculate distance
            dx = pt2['x'] - pt1['x']
            dy = pt2['y'] - pt1['y']
            segment_distance = math.sqrt(dx*dx + dy*dy)
            total_distance += segment_distance
            
            # Calculate angle change
            angle_change = abs(pt2['angle'] - pt1['angle'])
            total_angle_change += angle_change
            max_angle_change = max(max_angle_change, angle_change)
        
        # Calculate angle difference to target
        angle_diff = target_box['angle'] - current_angle
        # Normalize to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
            
        # Direction of turn
        turn_direction = "left" if angle_diff > 0 else "right"
        abs_angle_diff = abs(angle_diff)

        # Identify the segments (forward, then reverse)
        forward_segments = []
        reverse_segments = []
        
        current_action = path_points[0]['action']
        segment_start = 0
        
        for i in range(1, len(path_points)):
            if path_points[i]['action'] != current_action:
                if current_action == 'forward':
                    forward_segments.append((segment_start, i-1))
                else:
                    reverse_segments.append((segment_start, i-1))
                segment_start = i
                current_action = path_points[i]['action']
        
        # Add the last segment
        if current_action == 'forward':
            forward_segments.append((segment_start, len(path_points)-1))
        else:
            reverse_segments.append((segment_start, len(path_points)-1))
        
        # Generate detailed reverse parking instructions
        instructions.append(f"1. Start from your current position")
        
        # Forward movement to position for reverse
        if forward_segments:
            instructions.append(f"2. Drive forward to position for reverse parking")
            
            # Add specific turning instructions for positioning
            if abs_angle_diff > 5:
                instructions.append(f"3. Turn steering wheel {turn_direction} {abs_angle_diff:.1f}° while moving forward")
        
        # Reverse into parking space
        if reverse_segments:
            instructions.append(f"4. STOP and shift to REVERSE")
            instructions.append(f"5. Turn steering wheel in opposite direction")
            
            # Use target angle for final alignment
            reverse_target_diff = abs(target_box['angle'] - path_points[reverse_segments[0][0]]['angle'])
            reverse_direction = "left" if (target_box['angle'] - path_points[reverse_segments[0][0]]['angle']) > 0 else "right"
            
            instructions.append(f"6. Turn steering wheel {reverse_direction} {reverse_target_diff:.1f}° while reversing")
            instructions.append(f"7. Reverse slowly into the parking space")
            instructions.append(f"8. Straighten wheels when properly aligned")
        
        # Final adjustment
        instructions.append(f"9. Make final adjustments if needed")
        instructions.append(f"10. STOP when properly parked")

        # Assess difficulty
        difficulty = "Easy"
        if total_angle_change > 90 or abs_angle_diff > 45:
            difficulty = "Hard"
        elif total_angle_change > 45 or abs_angle_diff > 20:
            difficulty = "Medium"
        if total_distance > 200:
            difficulty = "Hard" if difficulty != "Easy" else "Medium"

        return {
            "instructions": instructions,
            "difficulty": difficulty,
            "turn_angle": abs_angle_diff,
            "turn_direction": turn_direction,
            "distance": total_distance
        }

    def toggle_parking_type(self):
        """Toggle between reverse and forward parking."""
        self.parking_type = "forward" if self.parking_type == "reverse" else "reverse"
        return self.parking_type

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

        # Calculate x and y axes of the box
        x_axis_angle = math.degrees(math.atan2(rotated_corners[1][1] - rotated_corners[0][1], 
                                             rotated_corners[1][0] - rotated_corners[0][0]))
        y_axis_angle = math.degrees(math.atan2(rotated_corners[3][1] - rotated_corners[0][1], 
                                             rotated_corners[3][0] - rotated_corners[0][0]))
        
        # Display coordinates, angle, and label
        text = f"Center: ({center_x}, {center_y}), Angle: {self.angle}°, {self.label}"
        cv.putText(image, text, (self.top_left[0] + 5, self.top_left[1] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)

        # Display front and end coordinates
        front_text = f"Front: ({int(front_coord[0])}, {int(front_coord[1])})"
        end_text = f"End: ({int(end_coord[0])}, {int(end_coord[1])})"
        cv.putText(image, front_text, (self.top_left[0] + 5, self.top_left[1] + 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)
        cv.putText(image, end_text, (self.top_left[0] + 5, self.top_left[1] + 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)
        
        # Display box axes angles
        axes_text = f"X-axis: {x_axis_angle:.1f}°, Y-axis: {y_axis_angle:.1f}°"
        cv.putText(image, axes_text, (self.top_left[0] + 5, self.top_left[1] + 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)

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

def generate_path(green_box, red_box, image, control_point1=None, control_point2=None, use_ml=False, path_trainer=None, rotation_offset=0, path_gen=None):
    """Generate a realistic car path from current position (green) to target position (red) using the PathGenerator class."""
    # Get the centers and end coordinates of both boxes
    green_center = green_box.get_center()
    red_center = red_box.get_center()
    
    # Create dictionaries for the PathGenerator
    current_box = {
        'center': green_center,
        'angle': green_box.angle
    }
    
    target_box = {
        'center': red_center,
        'angle': red_box.angle
    }
    
    # Create a PathGenerator instance if not provided
    if path_gen is None:
        path_gen = PathGenerator()
    
    # Generate the path
    path_points = path_gen.generate_parking_path(current_box, target_box)
    
    # Draw the path on the image
    image_with_path = path_gen.draw_path(image, path_points)

    # Draw path info
    cv.putText(image_with_path, f"Path Length: {len(path_points)} points", 
              (10, image.shape[0] - 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw instruction text for feedback
    cv.putText(image_with_path, "Press 'a' to approve path, 'r' to reject", 
              (10, image.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Convert path_points to the format expected by the rest of the code
    # The original format is a list of (x, y) tuples
    converted_path_points = [(pt['x'], pt['y']) for pt in path_points]
    
    # Return the image with the path drawn, the path points, and None for control points
    return image_with_path, converted_path_points, None, None

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
        'path_type': 'path_generator'  # Update path type information
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
    print(f"Path data saved as {status} example at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

def check_boxes_meet(red_box, green_box):
    """Check if the two boxes meet (are close enough to be considered parked)."""
    # Get the centers of both boxes
    red_center = red_box.get_center()
    green_center = green_box.get_center()
    
    # Calculate the distance between centers
    distance = math.sqrt((red_center[0] - green_center[0])**2 + (red_center[1] - green_center[1])**2)
    
    # Define a threshold for "meeting" - if boxes are within this distance, they're considered to meet
    # This threshold is based on the box dimensions
    threshold = (red_box.width + green_box.width) / 4  # Quarter of the combined width
    
    # Check if the angles are similar (within 10 degrees)
    angle_diff = abs(red_box.angle - green_box.angle) % 360
    angle_diff = min(angle_diff, 360 - angle_diff)  # Get the smaller angle difference
    
    # Return True if boxes are close enough and angles are similar
    return distance < threshold and angle_diff < 10

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for dragging boxes."""
    red_box, green_box, image, path_gen, edit_control_points = param
    
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
    
    # Generate and draw the path with the current rotation offset
    # Get rotation_offset from the global state in process_image
    global current_rotation_offset
    display_image, path_points, _, _ = generate_path(green_box, red_box, display_image, 
                                                   rotation_offset=current_rotation_offset,
                                                   path_gen=path_gen)
    
    # Draw boxes
    red_box.draw(display_image)
    green_box.draw(display_image)
    
    # Generate and display parking instructions
    if path_points:
        # Convert path_points to the format expected by generate_parking_instructions
        path_points_for_instructions = []
        for pt in path_points:
            path_points_for_instructions.append({
                'x': pt[0],
                'y': pt[1],
                'angle': green_box.angle,  # Use green box angle for better instructions
                'action': 'move'  # Default action
            })
        
        # Create target box for instructions
        target_box_for_instructions = {
            'angle': red_box.angle
        }
        
        # Generate instructions
        result = path_gen.generate_parking_instructions(path_points_for_instructions, target_box_for_instructions)
        
        # Display instructions on the image
        y_offset = 30
        for i, instruction in enumerate(result["instructions"]):
            cv.putText(display_image, instruction, (10, y_offset + i*25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display difficulty
        cv.putText(display_image, f"Difficulty: {result['difficulty']}", 
                  (10, y_offset + len(result["instructions"])*25 + 10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display specific turning instructions
        if 'turn_angle' in result:
            turn_text = f"Turn {result['turn_direction']} {result['turn_angle']:.1f}°"
            cv.putText(display_image, turn_text, 
                      (10, y_offset + len(result["instructions"])*25 + 35), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                      
        # Display parking type
        parking_type_text = f"Parking Mode: {path_gen.parking_type.upper()}"
        cv.putText(display_image, parking_type_text, 
                  (10, y_offset + len(result["instructions"])*25 + 60), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    
    # Check if boxes meet and display congratulatory message
    if check_boxes_meet(red_box, green_box):
        # Add a semi-transparent overlay for the congratulatory message
        overlay = display_image.copy()
        cv.rectangle(overlay, (0, 0), (display_image.shape[1], display_image.shape[0]), (0, 255, 0), -1)
        cv.addWeighted(overlay, 0.3, display_image, 0.7, 0, display_image)
        
        # Add congratulatory text
        text = "CONGRATULATIONS! The car is parked!"
        
        # Calculate appropriate font size based on screen dimensions
        height, width = display_image.shape[:2]
        # Start with a large font size
        font_scale = 2.0
        font_thickness = 3
        
        # Get text size
        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Reduce font size until text fits within 80% of screen width
        while text_size[0] > width * 0.8 and font_scale > 0.5:
            font_scale -= 0.1
            text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Adjust thickness based on font size
        font_thickness = max(1, int(font_scale * 1.5))
        
        # Calculate text position to center it
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # Draw text with outline for better visibility
        cv.putText(display_image, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2)
        cv.putText(display_image, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
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
    show_path = True  # Always show path by default
    path_points = []
    
    # Create a PathGenerator instance for generating instructions
    path_gen = PathGenerator()
    
    # Track rotation offset for the green car as a global to share with callback
    global current_rotation_offset
    current_rotation_offset = 0
    
    # Set up window and mouse callback
    cv.namedWindow('Image Display')
    cv.setMouseCallback('Image Display', mouse_callback, 
                      param=(red_box, green_box, image.copy(), path_gen, False))

    # Display the image
    display_image = image.copy()
    if CONFIG['SHOW_GRID']:
        display_image = draw_grid(display_image)
    
    # Generate initial path
    display_image, path_points, _, _ = generate_path(green_box, red_box, display_image, 
                                                   rotation_offset=current_rotation_offset,
                                                   path_gen=path_gen)
    
    # Draw boxes
    red_box.draw(display_image)
    green_box.draw(display_image)
    
    # Generate and display initial parking instructions
    if path_points:
        # Convert path_points to the format expected by generate_parking_instructions
        path_points_for_instructions = []
        for pt in path_points:
            path_points_for_instructions.append({
                'x': pt[0],
                'y': pt[1],
                'angle': green_box.angle,  # Use green box angle for better instructions
                'action': 'move'  # Default action
            })
        
        # Create target box for instructions
        target_box_for_instructions = {
            'angle': red_box.angle
        }
        
        # Generate instructions
        result = path_gen.generate_parking_instructions(path_points_for_instructions, target_box_for_instructions)
        
        # Display instructions on the image
        y_offset = 30
        for i, instruction in enumerate(result["instructions"]):
            cv.putText(display_image, instruction, (10, y_offset + i*25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display difficulty
        cv.putText(display_image, f"Difficulty: {result['difficulty']}", 
                  (10, y_offset + len(result["instructions"])*25 + 10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display specific turning instructions
        if 'turn_angle' in result:
            turn_text = f"Turn {result['turn_direction']} {result['turn_angle']:.1f}°"
            cv.putText(display_image, turn_text, 
                      (10, y_offset + len(result["instructions"])*25 + 35), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Display parking type
        parking_type_text = f"Parking Mode: {path_gen.parking_type.upper()}"
        cv.putText(display_image, parking_type_text, 
                  (10, y_offset + len(result["instructions"])*25 + 60), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    
    cv.imshow('Image Display', display_image)
    
    # Show instructions
    print("\nPath Generator Controls:")
    print("- 'z'/'x': Rotate green box counter-clockwise/clockwise")
    print("- 'c'/'v': Rotate red box counter-clockwise/clockwise")
    print("- 'p': Toggle path display")
    print("- 'a': Approve path as correct")
    print("- 'r': Reject path as incorrect")
    print("- 'i': Show parking instructions in console")
    print("- 'd': Toggle debug visualization")
    print("- 'g': Toggle grid display")
    print("- 'm': Toggle parking mode (forward/reverse)")
    print("- Space: Save box parameters to CSV")
    print("- 'q': Quit\n")
    
    print(f"Current parking mode: {path_gen.parking_type.upper()}")
    
    # Wait for 'q' to quit
    while True:
        key = cv.waitKey(100) & 0xFF  # Reduced wait time to check more frequently
        
        # Rotate red box with 'c' and 'v' keys
        if key == ord('c'):
            red_box.rotate(-5)
            show_path = True
        elif key == ord('v'):
            red_box.rotate(5)
            show_path = True

        # Rotate green box with 'z' and 'x' keys
        elif key == ord('z'):
            green_box.rotate(-5)
            current_rotation_offset = 5  # Invert: when box rotates counterclockwise, use clockwise offset
            show_path = True
            
        elif key == ord('x'):
            green_box.rotate(5)
            current_rotation_offset = -5  # Invert: when box rotates clockwise, use counterclockwise offset
            show_path = True
          
        # Toggle path display
        elif key == ord('p'):
            show_path = not show_path
            
        # Toggle parking mode
        elif key == ord('m'):
            new_mode = path_gen.toggle_parking_type()
            print(f"Switched to {new_mode.upper()} parking mode")
            show_path = True

        # Save parameters to CSV when spacebar is pressed
        elif key == ord(' '):
            save_to_csv(red_box, green_box)
            print(f"Parameters saved to box_parameters.csv at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
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
        
        # Show parking instructions in console
        elif key == ord('i') and show_path:
            if path_points:
                # Convert path_points to the format expected by generate_parking_instructions
                path_points_for_instructions = []
                for pt in path_points:
                    path_points_for_instructions.append({
                        'x': pt[0],
                        'y': pt[1],
                        'angle': green_box.angle,  # Use green box angle for better instructions
                        'action': 'move'  # Default action
                    })
                
                # Create target box for instructions
                target_box_for_instructions = {
                    'angle': red_box.angle
                }
                
                # Generate instructions
                result = path_gen.generate_parking_instructions(path_points_for_instructions, target_box_for_instructions)
                
                # Print instructions
                print("\n=== Parking Instructions ===")
                print(f"Parking Mode: {path_gen.parking_type.upper()}")
                for instruction in result["instructions"]:
                    print(instruction)
                print(f"\nParking Difficulty: {result['difficulty']}")
                if 'turn_angle' in result:
                    print(f"Turn {result['turn_direction']} {result['turn_angle']:.1f}°")
                if 'distance' in result:
                    print(f"Distance to travel: {result['distance']/100:.1f}m")
        
        # Toggle debug visualization
        elif key == ord('d'):
            CONFIG['DEBUG_VISUALIZATION'] = not CONFIG['DEBUG_VISUALIZATION']
            print(f"Debug visualization: {'ON' if CONFIG['DEBUG_VISUALIZATION'] else 'OFF'}")
        
        # Toggle grid display
        elif key == ord('g'):
            CONFIG['SHOW_GRID'] = not CONFIG['SHOW_GRID']
            print(f"Grid display: {'ON' if CONFIG['SHOW_GRID'] else 'OFF'}")
        
        # Update display when anything changes
        if show_path or key != 255:  # 255 means no key pressed
            display_image = image.copy()
            if CONFIG['SHOW_GRID']:
                display_image = draw_grid(display_image)
            if show_path:
                display_image, path_points, _, _ = generate_path(green_box, red_box, display_image, 
                                                              rotation_offset=current_rotation_offset,
                                                              path_gen=path_gen)
            red_box.draw(display_image)
            green_box.draw(display_image)
            
            # Generate and display parking instructions
            if show_path and path_points:
                # Convert path_points to the format expected by generate_parking_instructions
                path_points_for_instructions = []
                for pt in path_points:
                    path_points_for_instructions.append({
                        'x': pt[0],
                        'y': pt[1],
                        'angle': green_box.angle,  # Use green box angle for better instructions
                        'action': 'move'  # Default action
                    })
                
                # Create target box for instructions
                target_box_for_instructions = {
                    'angle': red_box.angle
                }
                
                # Generate instructions
                result = path_gen.generate_parking_instructions(path_points_for_instructions, target_box_for_instructions)
                
                # Display instructions on the image
                y_offset = 30
                for i, instruction in enumerate(result["instructions"]):
                    cv.putText(display_image, instruction, (10, y_offset + i*25), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display difficulty
                cv.putText(display_image, f"Difficulty: {result['difficulty']}", 
                          (10, y_offset + len(result["instructions"])*25 + 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display specific turning instructions
                if 'turn_angle' in result:
                    turn_text = f"Turn {result['turn_direction']} {result['turn_angle']:.1f}°"
                    cv.putText(display_image, turn_text, 
                              (10, y_offset + len(result["instructions"])*25 + 35), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                              
                # Display parking type
                parking_type_text = f"Parking Mode: {path_gen.parking_type.upper()}"
                cv.putText(display_image, parking_type_text, 
                          (10, y_offset + len(result["instructions"])*25 + 60), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            
            # Check if boxes meet and display congratulatory message
            if check_boxes_meet(red_box, green_box):
                # Add a semi-transparent overlay for the congratulatory message
                overlay = display_image.copy()
                cv.rectangle(overlay, (0, 0), (display_image.shape[1], display_image.shape[0]), (0, 255, 0), -1)
                cv.addWeighted(overlay, 0.3, display_image, 0.7, 0, display_image)
                
                # Add congratulatory text
                text = "CONGRATULATIONS! The car is parked!"
                
                # Calculate appropriate font size based on screen dimensions
                height, width = display_image.shape[:2]
                # Start with a large font size
                font_scale = 2.0
                font_thickness = 3
                
                # Get text size
                text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                
                # Reduce font size until text fits within 80% of screen width
                while text_size[0] > width * 0.8 and font_scale > 0.5:
                    font_scale -= 0.1
                    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                
                # Adjust thickness based on font size
                font_thickness = max(1, int(font_scale * 1.5))
                
                # Calculate text position to center it
                text_x = (width - text_size[0]) // 2
                text_y = (height + text_size[1]) // 2
                
                # Draw text with outline for better visibility
                cv.putText(display_image, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2)
                cv.putText(display_image, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
            cv.imshow('Image Display', display_image)
        
        # Quit on 'q'
        if key == ord('q'):
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