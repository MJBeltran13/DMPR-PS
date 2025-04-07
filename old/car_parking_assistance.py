"""Car Parking Assistance System with realistic turning paths."""
import cv2 as cv
import numpy as np
import math
from enum import Enum

class Direction(Enum):
    FORWARD = 1
    REVERSE = -1

class CarParameters:
    def __init__(self):
        # Car dimensions in pixels (can be adjusted based on image scale)
        self.length = 100  # Length of the car
        self.width = 50   # Width of the car
        self.wheelbase = 60  # Distance between front and rear axles
        self.min_turn_radius = 120  # Minimum turning radius
        self.max_steering_angle = math.radians(35)  # Maximum steering angle in radians

class ParkingPath:
    def __init__(self, start_pos, start_angle, end_pos, end_angle):
        self.start_pos = start_pos
        self.start_angle = start_angle
        self.end_pos = end_pos
        self.end_angle = end_angle
        self.segments = []  # List of (points, direction) for each path segment

def draw_car(image, position, angle, car_params, color=(0, 0, 255)):
    """Draw car representation at given position and angle."""
    x, y = position
    length = car_params.length
    width = car_params.width
    
    # Calculate corner points
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    points = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2]
    ])
    
    # Rotate and translate points
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    points = np.dot(points, rotation_matrix.T)
    points = points + np.array([x, y])
    
    # Draw car body
    points = points.astype(np.int32)
    cv.polylines(image, [points], True, color, 2)
    
    # Draw direction indicator (front of car)
    front_center = np.array([x + (length/2)*cos_angle, y + (length/2)*sin_angle], dtype=np.int32)
    cv.circle(image, tuple(front_center), 5, color, -1)

def calculate_turning_path(start_pos, start_angle, end_pos, end_angle, car_params, direction=Direction.FORWARD):
    """Calculate path with turning radius constraints."""
    x1, y1 = start_pos
    x2, y2 = end_pos
    R = car_params.min_turn_radius
    
    # Calculate centers of possible turning circles
    left_center_start = (
        x1 + R * math.cos(start_angle + math.pi/2),
        y1 + R * math.sin(start_angle + math.pi/2)
    )
    right_center_start = (
        x1 + R * math.cos(start_angle - math.pi/2),
        y1 + R * math.sin(start_angle - math.pi/2)
    )
    
    # Generate arc points
    def generate_arc_points(center, start, angle_range, steps=20):
        points = []
        cx, cy = center
        for t in np.linspace(0, angle_range, steps):
            x = cx + R * math.cos(t)
            y = cy + R * math.sin(t)
            points.append((int(x), int(y)))
        return points

    # For now, simplified path: turn -> straight -> turn
    path_points = []
    
    # First turn
    turn_angle = math.pi/4 if direction == Direction.FORWARD else -math.pi/4
    arc_points = generate_arc_points(right_center_start, start_angle, turn_angle)
    path_points.extend(arc_points)
    
    # Straight segment
    straight_length = math.sqrt((x2-x1)**2 + (y2-y1)**2) * 0.5
    current_pos = path_points[-1]
    angle = start_angle + turn_angle
    path_points.append((
        int(current_pos[0] + straight_length * math.cos(angle)),
        int(current_pos[1] + straight_length * math.sin(angle))
    ))
    
    return path_points

def draw_parking_path(image, path_points, color=(255, 255, 0), thickness=2):
    """Draw the parking path with direction indicators."""
    if len(path_points) < 2:
        return
        
    # Draw path segments
    for i in range(len(path_points)-1):
        pt1 = path_points[i]
        pt2 = path_points[i+1]
        cv.line(image, pt1, pt2, color, thickness)
        
        # Draw direction arrow
        mid_point = ((pt1[0] + pt2[0])//2, (pt1[1] + pt2[1])//2)
        angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        arrow_length = 20
        arrow_point = (
            int(mid_point[0] + arrow_length * math.cos(angle)),
            int(mid_point[1] + arrow_length * math.sin(angle))
        )
        cv.arrowedLine(image, mid_point, arrow_point, color, thickness)

def process_parking(image_path, target_slot_center, target_angle):
    """Process parking assistance for a specific slot."""
    # Load image
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Initialize car parameters
    car_params = CarParameters()
    
    # Set initial car position (center of image bottom)
    height, width = image.shape[:2]
    start_pos = (width//2, height-100)
    start_angle = -math.pi/2  # Pointing upward
    
    # Calculate parking path
    path_points = calculate_turning_path(
        start_pos, start_angle,
        target_slot_center, target_angle,
        car_params
    )
    
    # Draw initial car position
    draw_car(image, start_pos, start_angle, car_params)
    
    # Draw target parking slot
    cv.circle(image, target_slot_center, 5, (0, 255, 0), -1)
    
    # Draw parking path
    draw_parking_path(image, path_points)
    
    # Display result
    cv.imshow('Parking Assistance', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    # Example usage
    image_path = 'parking_lot.jpg'
    target_slot_center = (300, 200)  # Example coordinates
    target_angle = math.pi/4  # Example angle
    
    process_parking(image_path, target_slot_center, target_angle)

if __name__ == '__main__':
    main() 