import cv2
import numpy as np
import argparse
import os

class DragDropBoxes:
    def __init__(self, image_path=None, boxes_file=None):
        """
        Initialize the Drag and Drop Boxes application
        
        Args:
            image_path: Path to the image file for testing
            boxes_file: Path to save/load box coordinates
        """
        self.image_path = image_path
        self.boxes_file = boxes_file
        self.boxes = {}  # Dictionary to store multiple boxes
        self.drawing = False
        self.dragging = False
        self.drag_box_id = None  # ID of the box being dragged
        self.drag_start = None
        self.roi_points = []
        self.current_frame = None
        self.default_width = 100  # Default box width
        self.default_height = 100  # Default box height
        
        # Load existing boxes if file exists
        if boxes_file and os.path.exists(boxes_file):
            self.load_boxes()
    
    def load_boxes(self):
        """Load boxes from file"""
        try:
            with open(self.boxes_file, 'r') as f:
                data = f.readlines()
                for i, line in enumerate(data):
                    coords = list(map(int, line.strip().split(',')))
                    box_id = f"box_{i}"
                    self.boxes[box_id] = {
                        'coords': coords,
                        'center': [(coords[0] + coords[2]) // 2, 
                                  (coords[1] + coords[3]) // 2],
                        'color': (0, 255, 0)  # Default color
                    }
                print(f"Loaded {len(self.boxes)} boxes")
        except Exception as e:
            print(f"Error loading boxes: {e}")
    
    def save_boxes(self):
        """Save boxes to file"""
        if not self.boxes_file or not self.boxes:
            return
        
        try:
            with open(self.boxes_file, 'w') as f:
                for box_id, box in self.boxes.items():
                    coords_str = ','.join(map(str, box['coords']))
                    f.write(f"{coords_str}\n")
                
            print(f"Saved {len(self.boxes)} boxes")
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
    
    def find_box_at_point(self, point):
        """Find the box that contains the given point"""
        for box_id, box in self.boxes.items():
            if self.is_point_in_box(point, box):
                return box_id
        return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for box drawing and dragging"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if we're clicking inside an existing box
            box_id = self.find_box_at_point((x, y))
            if box_id:
                self.dragging = True
                self.drag_box_id = box_id
                self.drag_start = (x, y)
                self.roi_points = [self.boxes[box_id]['coords'][:2]]  # Store original position
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
                
                cv2.imshow('Drag and Drop Boxes', img_copy)
                
            elif self.dragging and self.drag_box_id:
                # Move the box
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                
                # Update the box position
                box = self.boxes[self.drag_box_id]
                x1, y1 = box['coords'][0] + dx, box['coords'][1] + dy
                x2, y2 = box['coords'][2] + dx, box['coords'][3] + dy
                box['coords'] = [x1, y1, x2, y2]
                box['center'] = [(x1 + x2) // 2, (y1 + y2) // 2]
                
                # Update drag start position
                self.drag_start = (x, y)
                
                # Redraw the frame
                self.draw_boxes(self.current_frame)
                cv2.imshow('Drag and Drop Boxes', self.current_frame)
                
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
                
                # Generate a unique ID for the new box
                box_id = f"box_{len(self.boxes)}"
                
                # Add the new box
                self.boxes[box_id] = {
                    'coords': [x1, y1, x2, y2],
                    'center': [center_x, center_y],
                    'color': (0, 255, 0)  # Default color
                }
                print(f"Box {box_id} drawn")
                
                # Draw all boxes on the frame
                self.draw_boxes(self.current_frame)
                cv2.imshow('Drag and Drop Boxes', self.current_frame)
                
            elif self.dragging:
                self.dragging = False
                self.drag_box_id = None
                self.drag_start = None
    
    def draw_boxes(self, frame):
        """Draw all boxes on the frame"""
        for box_id, box in self.boxes.items():
            x1, y1, x2, y2 = box['coords']
            color = box['color']
            
            # Draw the rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw box ID
            cv2.putText(frame, box_id, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x, center_y = box['center']
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Display box dimensions
            width = x2 - x1
            height = y2 - y1
            cv2.putText(frame, f"{width}x{height}", (x1, y2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add drag instruction
            cv2.putText(frame, "Drag to move", (x1, y2 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def run(self):
        """Run the drag and drop boxes application"""
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
        cv2.namedWindow('Drag and Drop Boxes')
        cv2.setMouseCallback('Drag and Drop Boxes', self.mouse_callback)
        
        # Instructions
        print("\n=== Drag and Drop Boxes ===")
        print("Click and drag to draw a new box")
        print("Click and drag on existing boxes to move them")
        print("Press 's' to save boxes")
        print("Press 'q' to quit")
        
        # Draw existing boxes
        self.draw_boxes(self.current_frame)
        
        # Show frame
        cv2.imshow('Drag and Drop Boxes', self.current_frame)
        
        while True:
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_boxes()
        
        # Clean up
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Drag and Drop Boxes')
    parser.add_argument('--image', type=str, default='test_images/test_image.jpg', 
                        help='Path to the image file for testing')
    parser.add_argument('--boxes', type=str, default='boxes.txt',
                        help='File to save/load box coordinates')
    args = parser.parse_args()
    
    # Create and run the application
    app = DragDropBoxes(args.image, args.boxes)
    app.run()

if __name__ == "__main__":
    main() 