import cv2
import numpy as np
import argparse
import time
import os
from datetime import datetime
import math

class SmartParkingAssistant:
    def __init__(self, image_path=None, parking_spots_file=None):
        """
        Initialize the Smart Parking Assistant
        
        Args:
            image_path: Path to the image file for testing
            parking_spots_file: Path to save/load parking spot coordinates
        """
        self.image_path = image_path
        self.parking_spots_file = parking_spots_file
        self.current_box = None
        self.target_box = None
        self.drawing = False
        self.dragging = False
        self.drag_box = None  # 'current' or 'target'
        self.drag_start = None
        self.roi_points = []
        self.box_type = None  # 'current' or 'target'
        self.path_points = []
        self.default_width = 116  # Default box width
        self.default_height = 208  # Default box height
        
        # Load existing boxes if file exists
        if parking_spots_file and os.path.exists(parking_spots_file):
            self.load_boxes()
    
    def load_boxes(self):
        """Load boxes from file"""
        try:
            with open(self.parking_spots_file, 'r') as f:
                data = f.readlines()
                if len(data) >= 2:
                    # Load current box
                    current_coords = list(map(int, data[0].strip().split(',')))
                    self.current_box = {
                        'coords': current_coords,
                        'center': [(current_coords[0] + current_coords[2]) // 2, 
                                  (current_coords[1] + current_coords[3]) // 2],
                        'angle': 0  # Default angle
                    }
                    
                    # Load target box
                    target_coords = list(map(int, data[1].strip().split(',')))
                    self.target_box = {
                        'coords': target_coords,
                        'center': [(target_coords[0] + target_coords[2]) // 2, 
                                  (target_coords[1] + target_coords[3]) // 2],
                        'angle': 0  # Default angle
                    }
                    
                    print("Loaded current and target boxes")
        except Exception as e:
            print(f"Error loading boxes: {e}")
    
    def save_boxes(self):
        """Save boxes to file"""
        if not self.parking_spots_file or not self.current_box or not self.target_box:
            return
        
        try:
            with open(self.parking_spots_file, 'w') as f:
                # Save current box
                current_coords_str = ','.join(map(str, self.current_box['coords']))
                f.write(f"{current_coords_str}\n")
                
                # Save target box
                target_coords_str = ','.join(map(str, self.target_box['coords']))
                f.write(f"{target_coords_str}\n")
                
            print("Saved current and target boxes")
        except Exception as e:
            print(f"Error saving boxes: {e}")
    
    def is_point_in_box(self, point, box):
        """Check if a point is inside a box"""
        if not box:
            return False
            
        x, y = point
        x1, y1, x2, y2 = box['coords']
        
        # Ensure coordinates are in the correct order
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for box drawing and dragging"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if we're clicking inside an existing box
            if self.current_box and self.is_point_in_box((x, y), self.current_box):
                self.dragging = True
                self.drag_box = 'current'
                self.drag_start = (x, y)
                self.roi_points = [self.current_box['coords'][:2]]  # Store original position
            elif self.target_box and self.is_point_in_box((x, y), self.target_box):
                self.dragging = True
                self.drag_box = 'target'
                self.drag_start = (x, y)
                self.roi_points = [self.target_box['coords'][:2]]  # Store original position
            else:
                # Start drawing a new box
                self.drawing = True
                self.roi_points = [(x, y)]
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Draw the rectangle as the user drags
                img_copy = self.current_frame.copy()
                
                # Calculate box dimensions based on default size
                x1, y1 = self.roi_points[0]
                x2, y2 = x, y
                
                # Ensure the box maintains the default aspect ratio
                width = abs(x2 - x1)
                height = int(width * (self.default_height / self.default_width))
                
                # Adjust the box to maintain the default size
                if x2 > x1:
                    x2 = x1 + self.default_width
                else:
                    x1 = x2 + self.default_width
                    x2 = self.roi_points[0][0]
                
                if y2 > y1:
                    y2 = y1 + self.default_height
                else:
                    y1 = y2 + self.default_height
                    y2 = self.roi_points[0][1]
                
                # Draw the rectangle
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw existing boxes
                self.draw_boxes(img_copy)
                
                cv2.imshow('Smart Parking Assistant', img_copy)
                
            elif self.dragging and self.drag_box:
                # Move the box
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                
                # Update the box position
                if self.drag_box == 'current':
                    x1, y1 = self.current_box['coords'][0] + dx, self.current_box['coords'][1] + dy
                    x2, y2 = self.current_box['coords'][2] + dx, self.current_box['coords'][3] + dy
                    self.current_box['coords'] = [x1, y1, x2, y2]
                    self.current_box['center'] = [(x1 + x2) // 2, (y1 + y2) // 2]
                elif self.drag_box == 'target':
                    x1, y1 = self.target_box['coords'][0] + dx, self.target_box['coords'][1] + dy
                    x2, y2 = self.target_box['coords'][2] + dx, self.target_box['coords'][3] + dy
                    self.target_box['coords'] = [x1, y1, x2, y2]
                    self.target_box['center'] = [(x1 + x2) // 2, (y1 + y2) // 2]
                
                # Update drag start position
                self.drag_start = (x, y)
                
                # Redraw the frame
                self.draw_boxes(self.current_frame)
                cv2.imshow('Smart Parking Assistant', self.current_frame)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                
                # Calculate box dimensions based on default size
                x1, y1 = self.roi_points[0]
                x2, y2 = x, y
                
                # Ensure the box maintains the default aspect ratio
                if x2 > x1:
                    x2 = x1 + self.default_width
                else:
                    x1 = x2 + self.default_width
                    x2 = self.roi_points[0][0]
                
                if y2 > y1:
                    y2 = y1 + self.default_height
                else:
                    y1 = y2 + self.default_height
                    y2 = self.roi_points[0][1]
                
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Add the new box
                if self.box_type == 'current':
                    self.current_box = {
                        'coords': [x1, y1, x2, y2],
                        'center': [center_x, center_y],
                        'angle': 0  # Default angle
                    }
                    print("Current box drawn")
                elif self.box_type == 'target':
                    self.target_box = {
                        'coords': [x1, y1, x2, y2],
                        'center': [center_x, center_y],
                        'angle': 0  # Default angle
                    }
                    print("Target box drawn")
                
                # Draw all boxes on the frame
                self.draw_boxes(self.current_frame)
                cv2.imshow('Smart Parking Assistant', self.current_frame)
                
            elif self.dragging:
                self.dragging = False
                self.drag_box = None
                self.drag_start = None
    
    def draw_boxes(self, frame):
        """Draw current and target boxes on the frame"""
        # Draw current box
        if self.current_box:
            x1, y1, x2, y2 = self.current_box['coords']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for current
            cv2.putText(frame, "Current", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw center point
            center_x, center_y = self.current_box['center']
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw direction indicator
            angle = self.current_box['angle']
            end_x = int(center_x + 30 * math.cos(math.radians(angle)))
            end_y = int(center_y + 30 * math.sin(math.radians(angle)))
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
            
            # Display box dimensions
            width = x2 - x1
            height = y2 - y1
            cv2.putText(frame, f"{width}x{height}", (x1, y2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add drag instruction
            cv2.putText(frame, "Drag to move", (x1, y2 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw target box
        if self.target_box:
            x1, y1, x2, y2 = self.target_box['coords']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for target
            cv2.putText(frame, "Target", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = self.target_box['center']
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # Draw direction indicator
            angle = self.target_box['angle']
            end_x = int(center_x + 30 * math.cos(math.radians(angle)))
            end_y = int(center_y + 30 * math.sin(math.radians(angle)))
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Display box dimensions
            width = x2 - x1
            height = y2 - y1
            cv2.putText(frame, f"{width}x{height}", (x1, y2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add drag instruction
            cv2.putText(frame, "Drag to move", (x1, y2 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def rotate_box(self, box, delta_angle):
        """Rotate a box by the given angle"""
        if not box:
            return
        
        # Update angle
        box['angle'] = (box['angle'] + delta_angle) % 360
        
        # Draw the updated frame
        self.draw_boxes(self.current_frame)
        cv2.imshow('Smart Parking Assistant', self.current_frame)
    
    def generate_parking_path(self):
        """Generate a path from current position to target parking spot"""
        if not self.current_box or not self.target_box:
            print("Both current and target boxes must be defined")
            return
        
        # Clear previous path
        self.path_points = []
        
        # Get centers and angles
        current_center = self.current_box['center']
        target_center = self.target_box['center']
        current_angle = self.current_box['angle']
        target_angle = self.target_box['angle']
        
        # Calculate distance and angle between centers
        dx = target_center[0] - current_center[0]
        dy = target_center[1] - current_center[1]
        distance = math.sqrt(dx*dx + dy*dy)
        angle_to_target = math.degrees(math.atan2(dy, dx))
        
        # Normalize angles to 0-360 range
        angle_to_target = (angle_to_target + 360) % 360
        current_angle = (current_angle + 360) % 360
        target_angle = (target_angle + 360) % 360
        
        # Calculate angle difference
        angle_diff = (target_angle - current_angle + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        
        # Generate path points
        # 1. Initial turn to face target
        turn_angle = angle_to_target - current_angle
        if turn_angle > 180:
            turn_angle -= 360
        elif turn_angle < -180:
            turn_angle += 360
            
        # Add turn points
        turn_steps = 10
        for i in range(turn_steps + 1):
            t = i / turn_steps
            angle = current_angle + t * turn_angle
            self.path_points.append({
                'x': current_center[0],
                'y': current_center[1],
                'angle': angle,
                'action': 'turn'
            })
        
        # 2. Move towards target
        move_steps = 20
        for i in range(1, move_steps + 1):
            t = i / move_steps
            x = current_center[0] + t * dx
            y = current_center[1] + t * dy
            self.path_points.append({
                'x': int(x),
                'y': int(y),
                'angle': angle_to_target,
                'action': 'move'
            })
        
        # 3. Final turn to align with target
        for i in range(1, turn_steps + 1):
            t = i / turn_steps
            angle = angle_to_target + t * (target_angle - angle_to_target)
            self.path_points.append({
                'x': target_center[0],
                'y': target_center[1],
                'angle': angle,
                'action': 'turn'
            })
        
        # Draw the path
        self.draw_path()
        
        # Generate parking instructions
        self.generate_parking_instructions()
    
    def draw_path(self):
        """Draw the generated path on the frame"""
        if not self.path_points:
            return
        
        # Create a copy of the current frame
        frame_with_path = self.current_frame.copy()
        
        # Draw path points and lines
        for i in range(len(self.path_points) - 1):
            pt1 = (self.path_points[i]['x'], self.path_points[i]['y'])
            pt2 = (self.path_points[i+1]['x'], self.path_points[i+1]['y'])
            
            # Draw line between points
            cv2.line(frame_with_path, pt1, pt2, (255, 0, 0), 2)
            
            # Draw point
            cv2.circle(frame_with_path, pt1, 3, (255, 0, 0), -1)
        
        # Draw the last point
        last_pt = (self.path_points[-1]['x'], self.path_points[-1]['y'])
        cv2.circle(frame_with_path, last_pt, 3, (255, 0, 0), -1)
        
        # Draw direction indicators at key points
        for i in range(0, len(self.path_points), 5):
            pt = self.path_points[i]
            angle = pt['angle']
            center = (pt['x'], pt['y'])
            
            # Draw direction arrow
            end_x = int(center[0] + 20 * math.cos(math.radians(angle)))
            end_y = int(center[1] + 20 * math.sin(math.radians(angle)))
            cv2.arrowedLine(frame_with_path, center, (end_x, end_y), (0, 255, 255), 2)
        
        # Show the frame with path
        cv2.imshow('Smart Parking Assistant', frame_with_path)
    
    def generate_parking_instructions(self):
        """Generate step-by-step parking instructions"""
        if not self.path_points:
            print("No path generated yet")
            return
        
        print("\n=== Parking Instructions ===")
        
        # Count turns and moves
        turns = sum(1 for pt in self.path_points if pt['action'] == 'turn')
        moves = sum(1 for pt in self.path_points if pt['action'] == 'move')
        
        # Calculate total angle change
        total_angle_change = 0
        for i in range(1, len(self.path_points)):
            angle_diff = self.path_points[i]['angle'] - self.path_points[i-1]['angle']
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
            total_angle_change += angle_diff
        
        # Generate instructions
        print(f"1. Start at the current position (red box)")
        print(f"2. Turn {'left' if total_angle_change < 0 else 'right'} approximately {abs(total_angle_change):.1f} degrees")
        print(f"3. Move forward approximately {moves * 5} units")
        print(f"4. Turn {'left' if self.target_box['angle'] - self.path_points[-2]['angle'] < 0 else 'right'} to align with the target parking spot")
        print(f"5. Park in the target spot (green box)")
        
        # Calculate parking difficulty
        difficulty = "Easy"
        if abs(total_angle_change) > 90 or moves > 15:
            difficulty = "Moderate"
        if abs(total_angle_change) > 150 or moves > 25:
            difficulty = "Difficult"
            
        print(f"\nParking Difficulty: {difficulty}")
    
    def run(self):
        """Run the smart parking assistant"""
        # Load the image
        if not self.image_path or not os.path.exists(self.image_path):
            print(f"Error: Image file not found: {self.image_path}")
            return
        
        frame = cv2.imread(self.image_path)
        if frame is None:
            print(f"Error: Could not read image: {self.image_path}")
            return
        
        self.current_frame = frame.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow('Smart Parking Assistant')
        cv2.setMouseCallback('Smart Parking Assistant', self.mouse_callback)
        
        # Instructions
        print("\n=== Smart Parking Assistant ===")
        print("Press 'c' to draw current position box")
        print("Press 't' to draw target parking spot box")
        print("Press 'r' to rotate current box clockwise")
        print("Press 'l' to rotate current box counterclockwise")
        print("Press 'p' to generate parking path")
        print("Press 's' to save boxes")
        print("Press 'q' to quit")
        print("Click and drag on boxes to move them")
        
        # Draw existing boxes
        self.draw_boxes(self.current_frame)
        
        # Show frame
        cv2.imshow('Smart Parking Assistant', self.current_frame)
        
        while True:
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.box_type = 'current'
                print("Draw current position box (red)")
            elif key == ord('t'):
                self.box_type = 'target'
                print("Draw target parking spot box (green)")
            elif key == ord('r'):
                self.rotate_box(self.current_box, 15)  # Rotate 15 degrees clockwise
            elif key == ord('l'):
                self.rotate_box(self.current_box, -15)  # Rotate 15 degrees counterclockwise
            elif key == ord('p'):
                self.generate_parking_path()
            elif key == ord('s'):
                self.save_boxes()
        
        # Clean up
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Smart Parking Assistant')
    parser.add_argument('--image', type=str, default='data/images/ps2.0/testing/outdoor-normal daylight/193.jpg', 
                        help='Path to the image file for testing')
    parser.add_argument('--spots', type=str, default='parking_spots.txt',
                        help='File to save/load box coordinates')
    args = parser.parse_args()
    
    # Create and run the parking assistant
    assistant = SmartParkingAssistant(args.image, args.spots)
    assistant.run()

if __name__ == "__main__":
    main()
