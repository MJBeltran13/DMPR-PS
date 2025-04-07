"""Smart Parking Assistance System with realistic car path planning."""
import cv2 as cv
import argparse
import numpy as np
import csv
from datetime import datetime
import math
import os
import json

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


def generate_path(green_box, red_box, image, control_point1=None, control_point2=None):
    """Generate a realistic car path from current position (green) to target position (red)."""
    # Get the centers and end coordinates of both boxes
    _, green_end = green_box.get_end_coordinates()
    _, red_end = red_box.get_end_coordinates()
    
    # Extract points as integers
    start_x, start_y = int(green_end[0]), int(green_end[1])
    target_x, target_y = int(red_end[0]), int(red_end[1])
    
    # Calculate distance between points
    distance = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
    
    # Generate intermediate points for a smooth path
    # Using Bezier curve for smoother path
    num_points = 50
    points = []
    
    # Create control points based on car orientation angles if not provided
    if control_point1 is None or control_point2 is None:
        green_angle_rad = math.radians(green_box.angle)
        red_angle_rad = math.radians(red_box.angle)
        
        # Calculate control points (this is simplified - can be made more sophisticated)
        ctrl_dist = distance * 0.5
        
        # Control point for start (based on green box angle)
        ctrl1_x = start_x + ctrl_dist * math.sin(green_angle_rad)
        ctrl1_y = start_y - ctrl_dist * math.cos(green_angle_rad)
        
        # Control point for end (based on red box angle)
        ctrl2_x = target_x + ctrl_dist * math.sin(red_angle_rad + math.pi)
        ctrl2_y = target_y - ctrl_dist * math.cos(red_angle_rad + math.pi)
    else:
        # Use provided control points
        ctrl1_x, ctrl1_y = control_point1.position
        ctrl2_x, ctrl2_y = control_point2.position
    
    # Generate Bezier curve points
    for i in range(num_points + 1):
        t = i / num_points
        # Cubic Bezier formula
        x = int((1-t)**3 * start_x + 3*(1-t)**2*t*ctrl1_x + 3*(1-t)*t**2*ctrl2_x + t**3*target_x)
        y = int((1-t)**3 * start_y + 3*(1-t)**2*t*ctrl1_y + 3*(1-t)*t**2*ctrl2_y + t**3*target_y)
        points.append((x, y))
    
    # Draw the path
    path_color = (255, 165, 0)  # Orange
    
    # Draw the main path
    for i in range(1, len(points)):
        cv.line(image, points[i-1], points[i], path_color, 2)
    
    # Draw control points and their connections
    cv.circle(image, (int(ctrl1_x), int(ctrl1_y)), 5, (255, 0, 255), -1)  # Magenta
    cv.circle(image, (int(ctrl2_x), int(ctrl2_y)), 5, (255, 0, 255), -1)
    
    # Draw lines connecting control points to endpoints
    cv.line(image, (start_x, start_y), (int(ctrl1_x), int(ctrl1_y)), (100, 100, 255), 1, cv.LINE_AA)
    cv.line(image, (target_x, target_y), (int(ctrl2_x), int(ctrl2_y)), (100, 100, 255), 1, cv.LINE_AA)
    
    # Draw path info
    cv.putText(image, f"Path Length: {distance:.1f} pixels", (10, image.shape[0] - 50), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw instruction text for feedback
    cv.putText(image, "Press 'a' to approve path, 'r' to reject, 'm' to modify control points", 
              (10, image.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Store and return control points for potential reuse
    if control_point1 is None:
        control_point1 = DraggableControlPoint((ctrl1_x, ctrl1_y))
    if control_point2 is None:
        control_point2 = DraggableControlPoint((ctrl2_x, ctrl2_y))
        
    return image, points, control_point1, control_point2


def save_path_data(green_box, red_box, path_points, is_correct, control_points=None, filename='path_data.json'):
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
        'path_points': path_points
    }
    
    # Add control points if available
    if control_points:
        data['control_points'] = [
            {'x': control_points[0].position[0], 'y': control_points[0].position[1]},
            {'x': control_points[1].position[0], 'y': control_points[1].position[1]}
        ]
    
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


def mouse_callback(event, x, y, flags, param):
    red_box, green_box, original_image, control_points, edit_control_points = param
    
    if event == cv.EVENT_LBUTTONDOWN:
        if edit_control_points and control_points and len(control_points) == 2:
            # Check if clicked on control points first when in edit mode
            if control_points[0].is_inside(x, y):
                control_points[0].start_drag(x, y)
            elif control_points[1].is_inside(x, y):
                control_points[1].start_drag(x, y)
            # Then check boxes
            elif red_box.is_inside(x, y):
                red_box.start_drag(x, y)
            elif green_box.is_inside(x, y):
                green_box.start_drag(x, y)
        else:
            # Normal mode, only drag boxes
            if red_box.is_inside(x, y):
                red_box.start_drag(x, y)
            elif green_box.is_inside(x, y):
                green_box.start_drag(x, y)
    
    elif event == cv.EVENT_MOUSEMOVE:
        update_needed = False
        
        # Handle control point dragging first
        if edit_control_points and control_points and len(control_points) == 2:
            for cp in control_points:
                if cp.dragging:
                    cp.update_position(x, y)
                    update_needed = True
        
        # Handle box dragging
        if red_box.dragging:
            red_box.update_position(x, y)
            update_needed = True
        elif green_box.dragging:
            green_box.update_position(x, y)
            update_needed = True

        # Redraw if needed
        if update_needed:
            # The main loop will handle redrawing
            pass
    
    elif event == cv.EVENT_LBUTTONUP:
        if edit_control_points and control_points and len(control_points) == 2:
            for cp in control_points:
                cp.stop_drag()
        
        red_box.stop_drag()
        green_box.stop_drag()


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
    red_box = DraggableBox((50, 50), red_color, box_width, box_height, "Target Position")
    green_box = DraggableBox((image.shape[1] - box_width - 50, 50), green_color, box_width, box_height, "Current Position")

    # Path visualization variables
    show_path = False
    path_points = []
    control_points = []
    edit_control_points = False
    
    # Set up window and mouse callback
    cv.namedWindow('Image Display')
    cv.setMouseCallback('Image Display', mouse_callback, 
                      param=(red_box, green_box, image.copy(), control_points, edit_control_points))

    # Display the image
    display_image = image.copy()
    red_box.draw(display_image)
    green_box.draw(display_image)
    cv.imshow('Image Display', display_image)
    
    # Show instructions
    print("\nControls:")
    print("- 'z'/'x': Rotate green box counter-clockwise/clockwise")
    print("- 'c'/'v': Rotate red box counter-clockwise/clockwise")
    print("- 'p': Generate parking path")
    print("- 'm': Toggle control point editing mode (after generating path)")
    print("- 'a': Approve path as correct")
    print("- 'r': Reject path as incorrect")
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
            show_path = False
        elif key == ord('x'):
            green_box.rotate(5)
            show_path = False

        # Save parameters to CSV when spacebar is pressed
        elif key == ord(' '):
            save_to_csv(red_box, green_box)
            print(f"Parameters saved to box_parameters.csv at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generate and show path when 'p' is pressed
        elif key == ord('p'):
            show_path = True
            # Reset control points when generating a new path
            if not edit_control_points:
                control_points = []
            print("Generating parking path...")
        
        # Toggle control point editing mode
        elif key == ord('m') and show_path:
            edit_control_points = not edit_control_points
            # Update mouse callback parameters
            cv.setMouseCallback('Image Display', mouse_callback, 
                              param=(red_box, green_box, image.copy(), control_points, edit_control_points))
            print(f"Control point editing mode: {'ON' if edit_control_points else 'OFF'}")
        
        # Approve path
        elif key == ord('a') and show_path:
            # Save as a correct path example
            save_path_data(green_box, red_box, path_points, True, control_points)
            print("Path approved and saved as correct example.")
        
        # Reject path
        elif key == ord('r') and show_path:
            # Save as an incorrect path example
            save_path_data(green_box, red_box, path_points, False, control_points)
            print("Path rejected and saved as incorrect example.")

        # Redraw the image with updated box positions and rotations
        display_image = image.copy()
        
        # If path should be shown, generate and draw it
        if show_path:
            display_image, path_points, control_point1, control_point2 = generate_path(
                green_box, red_box, display_image, 
                None if not control_points else control_points[0],
                None if not control_points else control_points[1]
            )
            
            # Update control points if needed
            if not control_points or len(control_points) != 2:
                control_points = [control_point1, control_point2]
        
        # Draw control points in edit mode
        if edit_control_points and control_points and show_path:
            cv.putText(display_image, "Control Point Edit Mode: ON", (10, 30), 
                     cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for cp in control_points:
                cp.draw(display_image)
        
        red_box.draw(display_image)
        green_box.draw(display_image)
        cv.imshow('Image Display', display_image)

        if key == ord('q'):
            break
    
    cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Image Display')
    parser.add_argument('--image', default='data/images/ps2.0/testing/outdoor-normal daylight/193.jpg',
                      help='Path to input image')
    
    args = parser.parse_args()
    
    # Process image
    print(f"Displaying image: {args.image}")
    process_image(args.image)

if __name__ == '__main__':
    main() 