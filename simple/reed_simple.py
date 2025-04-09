import cv2 as cv
import numpy as np
import math

# Global configuration parameters
CONFIG = {
    'TURNING_RADIUS': 38.0,
    'DISTANCE_SCALING': 3.2,
    'MIN_TURNING_RADIUS': 35.0,
    'MAX_TURNING_RADIUS': 42.0,
    'PATH_POINTS': 120,
    'PATH_COLOR': (0, 255, 0),
    'REVERSE_COLOR': (0, 0, 255),
    'PATH_THICKNESS': 2,
    'SHOW_GRID': True,
    'GRID_SIZE': 50,
    'GRID_COLOR': (100, 100, 100),
}

class DraggableBox:
    def __init__(self, top_left, color, width, height, label):
        self.top_left = top_left
        self.color = color
        self.width = width
        self.height = height
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0
        self.angle = 0
        self.label = label

    def draw(self, image):
        center_x = self.top_left[0] + self.width // 2
        center_y = self.top_left[1] + self.height // 2
        rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), self.angle, 1.0)

        corners = np.array([
            [self.top_left[0], self.top_left[1]],
            [self.top_left[0] + self.width, self.top_left[1]],
            [self.top_left[0] + self.width, self.top_left[1] + self.height],
            [self.top_left[0], self.top_left[1] + self.height]
        ])

        rotated_corners = cv.transform(np.array([corners]), rotation_matrix)[0]
        cv.polylines(image, [np.int32(rotated_corners)], isClosed=True, color=self.color, thickness=2)

        # Calculate front and end coordinates
        front_coord = ((rotated_corners[0][0] + rotated_corners[1][0]) // 2, 
                      (rotated_corners[0][1] + rotated_corners[1][1]) // 2)
        end_coord = ((rotated_corners[2][0] + rotated_corners[3][0]) // 2, 
                    (rotated_corners[2][1] + rotated_corners[3][1]) // 2)

        # Draw a dot at the end coordinate
        cv.circle(image, (int(end_coord[0]), int(end_coord[1])), 5, self.color, -1)
    
    def get_end_coordinates(self):
        center_x = self.top_left[0] + self.width // 2
        center_y = self.top_left[1] + self.height // 2
        rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), self.angle, 1.0)
        
        corners = np.array([
            [self.top_left[0], self.top_left[1]],
            [self.top_left[0] + self.width, self.top_left[1]],
            [self.top_left[0] + self.width, self.top_left[1] + self.height],
            [self.top_left[0], self.top_left[1] + self.height]
        ])
        
        rotated_corners = cv.transform(np.array([corners]), rotation_matrix)[0]
        front_coord = ((rotated_corners[0][0] + rotated_corners[1][0]) // 2, 
                      (rotated_corners[0][1] + rotated_corners[1][1]) // 2)
        end_coord = ((rotated_corners[2][0] + rotated_corners[3][0]) // 2, 
                    (rotated_corners[2][1] + rotated_corners[3][1]) // 2)
        
        return front_coord, end_coord

    def is_inside(self, x, y):
        return (self.top_left[0] <= x <= self.top_left[0] + self.width and 
                self.top_left[1] <= y <= self.top_left[1] + self.height)

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

def generate_path(green_box, red_box, image, rotation_offset=0):
    _, green_end = green_box.get_end_coordinates()
    _, red_end = red_box.get_end_coordinates()
    
    start_x, start_y = int(green_end[0]), int(green_end[1])
    target_x, target_y = int(red_end[0]), int(red_end[1])
    
    # Calculate angles
    green_angle = math.radians(green_box.angle)
    red_angle = math.radians(red_box.angle)
    
    # Calculate distance
    distance = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
    
    # Calculate turning radius
    turning_radius = min(CONFIG['MAX_TURNING_RADIUS'], 
                        max(CONFIG['MIN_TURNING_RADIUS'], 
                            distance / CONFIG['DISTANCE_SCALING']))
    
    # Generate path points
    points = []
    num_points = CONFIG['PATH_POINTS']
    
    # First segment (forward)
    forward_points = int(num_points * 0.4)
    for i in range(forward_points):
        t = i / (forward_points - 1)
        # Create a curved path using quadratic Bezier curve
        control_x = start_x + distance * 0.5 * math.cos(green_angle)
        control_y = start_y + distance * 0.5 * math.sin(green_angle)
        x = (1-t)**2 * start_x + 2*(1-t)*t * control_x + t**2 * target_x
        y = (1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * target_y
        points.append((int(x), int(y)))
    
    # Second segment (reverse)
    reverse_points = num_points - forward_points
    for i in range(reverse_points):
        t = i / (reverse_points - 1)
        x = target_x + t * (points[-1][0] - target_x)
        y = target_y + t * (points[-1][1] - target_y)
        points.append((int(x), int(y)))
    
    # Draw the path
    # Forward path (blue)
    for i in range(1, forward_points):
        cv.line(image, points[i-1], points[i], (255, 165, 0), CONFIG['PATH_THICKNESS'])
    
    # Reverse path (red)
    for i in range(forward_points, len(points)-1):
        cv.line(image, points[i], points[i+1], CONFIG['REVERSE_COLOR'], CONFIG['PATH_THICKNESS'])
    
    return image, points

def draw_grid(image):
    if not CONFIG['SHOW_GRID']:
        return image
        
    grid_image = image.copy()
    h, w = grid_image.shape[:2]
    grid_size = CONFIG['GRID_SIZE']
    grid_color = CONFIG['GRID_COLOR']
    
    for x in range(0, w, grid_size):
        cv.line(grid_image, (x, 0), (x, h), grid_color, 1)
        if x % 100 == 0:
            cv.putText(grid_image, str(x), (x + 5, 15), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    for y in range(0, h, grid_size):
        cv.line(grid_image, (0, y), (w, y), grid_color, 1)
        if y % 100 == 0:
            cv.putText(grid_image, str(y), (5, y + 15), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return grid_image

def mouse_callback(event, x, y, flags, param):
    red_box, green_box, image = param
    
    if event == cv.EVENT_LBUTTONDOWN:
        if red_box.is_inside(x, y):
            red_box.start_drag(x, y)
        elif green_box.is_inside(x, y):
            green_box.start_drag(x, y)
    
    elif event == cv.EVENT_MOUSEMOVE:
        red_box.update_position(x, y)
        green_box.update_position(x, y)
    
    elif event == cv.EVENT_LBUTTONUP:
        red_box.stop_drag()
        green_box.stop_drag()
    
    display_image = image.copy()
    if CONFIG['SHOW_GRID']:
        display_image = draw_grid(display_image)
    display_image, _ = generate_path(green_box, red_box, display_image)
    red_box.draw(display_image)
    green_box.draw(display_image)
    cv.imshow('Path Planning', display_image)

def main():
    # Read image
    image = cv.imread('data/images/ps2.0/testing/outdoor-normal daylight/193.jpg')
    if image is None:
        print("Error: Could not read image")
        return
    
    # Initialize boxes
    box_width, box_height = 116, 208
    red_box = DraggableBox((503, 378), (0, 0, 255), box_width, box_height, "Target")
    red_box.angle = 90
    green_box = DraggableBox((242, 180), (0, 255, 0), box_width, box_height, "Current")
    
    # Setup display
    cv.namedWindow('Path Planning')
    cv.setMouseCallback('Path Planning', mouse_callback, (red_box, green_box, image.copy()))
    
    # Initial display
    display_image = image.copy()
    if CONFIG['SHOW_GRID']:
        display_image = draw_grid(display_image)
    display_image, _ = generate_path(green_box, red_box, display_image)
    red_box.draw(display_image)
    green_box.draw(display_image)
    cv.imshow('Path Planning', display_image)
    
    print("\nControls:")
    print("- Left click and drag boxes to move them")
    print("- 'z'/'x': Rotate green box counter-clockwise/clockwise")
    print("- 'c'/'v': Rotate red box counter-clockwise/clockwise")
    print("- 'g': Toggle grid")
    print("- 'q': Quit")
    
    while True:
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('z'):
            green_box.rotate(-5)
        elif key == ord('x'):
            green_box.rotate(5)
        elif key == ord('c'):
            red_box.rotate(-5)
        elif key == ord('v'):
            red_box.rotate(5)
        elif key == ord('g'):
            CONFIG['SHOW_GRID'] = not CONFIG['SHOW_GRID']
        elif key == ord('q'):
            break
        
        if key in [ord('z'), ord('x'), ord('c'), ord('v'), ord('g')]:
            display_image = image.copy()
            if CONFIG['SHOW_GRID']:
                display_image = draw_grid(display_image)
            display_image, _ = generate_path(green_box, red_box, display_image)
            red_box.draw(display_image)
            green_box.draw(display_image)
            cv.imshow('Path Planning', display_image)
    
    cv.destroyAllWindows()

if __name__ == '__main__':
    main() 