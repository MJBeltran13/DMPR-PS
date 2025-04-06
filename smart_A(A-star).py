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

def draw_car(image, position, angle, car_params):
    """Draw a single red box representing the car at the given position."""
    x, y = position  # Unpack only x and y
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

def normalize_angle(angle):
    """Normalize angle to range [-π, π]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def calculate_parking_path(start_pos, target_pos, car_params):
    """
    Generate a realistic path from the car's position to the target parking slot
    considering the car's dimensions, angle, and turning constraints.
    
    Parameters:
    - start_pos: Tuple (x, y) of starting position
    - target_pos: Tuple (x, y) of target position (parking slot center)
    - car_params: CarParameters object with vehicle specs
    
    Returns:
    - List of points representing the path
    """
    x1, y1 = start_pos
    x2, y2 = target_pos
    
    # Calculate the angle to the target position
    target_angle = math.atan2(y2 - y1, x2 - x1)
    
    # Normalize the target angle
    target_angle = normalize_angle(target_angle)
    
    # Current car angle
    car_angle = car_params.angle
    
    # Print parameters
    print(f"Starting Position: {start_pos}, Starting Angle: {math.degrees(car_angle)}°")
    print(f"Target Position: {target_pos}, Target Angle: {math.degrees(target_angle)}°")
    
    # Generate the curved path using the updated logic
    return generate_curved_parking_path(start_pos, target_pos, car_angle, target_angle, car_params)

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

        # Draw circles for starting and ending points
        cv.circle(image, (int(start_pt[0]), int(start_pt[1])), 5, (255, 0, 0), -1)  # Red circle for start
        cv.circle(image, (int(end_pt[0]), int(end_pt[1])), 5, (0, 255, 0), -1)  # Green circle for end

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for car dragging."""
    car, image, car_params, slot_centers, last_selected_slot = param
    
    if event == cv.EVENT_LBUTTONDOWN:
        # Check if click is near car center
        car_x, car_y = car.pos
        distance = math.sqrt((x - car_x)**2 + (y - car_y)**2)
        if distance < 30:  # Click within 30 pixels of car center
            car.dragging = True
            car.offset_x = car_x - x
            car.offset_y = car_y - y
    
    elif event == cv.EVENT_MOUSEMOVE:
        if car.dragging:
            # Update car position
            car.pos = (x + car.offset_x, y + car.offset_y)
            
            # Create a fresh copy of the image and redraw everything
            display_image = image.copy()
            
            # If a slot is selected, update the path
            if last_selected_slot[0] is not None and 0 <= last_selected_slot[0] < len(slot_centers):
                target_pos = slot_centers[last_selected_slot[0]]
                path_points = calculate_parking_path(car.pos, target_pos, car_params)
                draw_parking_path(display_image, path_points, car.angle)
                
                # Draw car positions along the path
                for i, point in enumerate(path_points[::5]):
                    if i < len(path_points[::5]) - 1:
                        next_point = path_points[::5][i + 1]
                        angle = math.atan2(next_point[1] - point[1],
                                         next_point[0] - point[0])
                        draw_car(display_image, point, angle, car_params)
            
            # Draw current car position
            draw_car(display_image, car.pos, 0, car_params)
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
    
    # Set initial car position (adjusted to be higher in the image)
    height, width = image.shape[:2]
    initial_pos = (width//2, height-50)
    initial_angle = -math.pi/2  # Pointing upward
    
    # Create draggable car instance
    car = DraggableCar(initial_pos, initial_angle)
    
    print(f"\nImage dimensions: {width}x{height}")
    print(f"Car starting position: {car.pos}")
    
    # Detect parking slots
    pred_points = detect_marking_points(model, image, config.CONFID_THRESH_FOR_POINT, device)
    slots = None
    slot_centers = []
    
    if pred_points:
        marking_points = list(list(zip(*pred_points))[1])
        slots = inference_slots(marking_points)
        
        print(f"\nDetected {len(slots)} parking slots")
        
        # Process each slot
        for idx, slot in enumerate(slots):
            info = get_slot_info(image, marking_points, slot)
            center_x = int(info['center'][0])
            center_y = int(info['center'][1])
            slot_centers.append((center_x, center_y))
            
            print(f"Slot #{idx + 1} center: ({center_x}, {center_y})")
            
            # Draw slot marker
            cv.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
            cv.putText(image, f"#{idx + 1}", 
                      (center_x + 10, center_y + 10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Plot basic slot visualization
    plot_points(image, pred_points)
    plot_slots(image, pred_points, slots)
    
    # Create window and set mouse callback
    cv.namedWindow('Smart Parking Assistance')
    last_selected_slot = [None]  # Use list to allow modification in callback
    cv.setMouseCallback('Smart Parking Assistance', 
                       mouse_callback, 
                       param=(car, image, car_params, slot_centers, last_selected_slot))
    
    # Add instructions to the image
    instructions = [
        "Controls:",
        "- Drag car with left mouse button",
        "- Press 1-9 to select parking slot",
        "- Press SPACE to generate new path",
        "- Press 'q' to quit"
    ]
    
    y_offset = 30
    for instruction in instructions:
        cv.putText(image, instruction, (10, y_offset),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    # Variables for path variation
    current_variation = 0.0
    
    # Initial car draw
    display_image = image.copy()
    draw_car(display_image, car.pos, 0, car_params)
    cv.imshow('Smart Parking Assistance', display_image)
    
    while True:
        # Create a fresh copy of the base image
        display_image = image.copy()
        
        # If a slot was selected, show the path
        if last_selected_slot[0] is not None and 0 <= last_selected_slot[0] < len(slot_centers):
            target_pos = slot_centers[last_selected_slot[0]]
            path_points = calculate_parking_path(car.pos, target_pos, car_params)
            draw_parking_path(display_image, path_points, car.angle)
            
            # Draw car positions along the path (improved spacing and angle calculation)
            # Use fewer ghost cars with better spacing
            num_ghost_cars = min(5, len(path_points) // 10)
            if num_ghost_cars > 0:
                indices = [i * len(path_points) // num_ghost_cars for i in range(num_ghost_cars)]
                
                for idx in indices:
                    if idx < len(path_points) - 1:
                        # Calculate a smoother angle by looking ahead multiple points
                        look_ahead = min(idx + 5, len(path_points) - 1)
                        angle = math.atan2(
                            path_points[look_ahead][1] - path_points[idx][1],
                            path_points[look_ahead][0] - path_points[idx][0]
                        )
                        # Draw ghost car with partially transparent color
                        draw_car(display_image, path_points[idx], angle, car_params)
        
        # Always draw the current car position on top
        draw_car(display_image, car.pos, 0, car_params)
        cv.imshow('Smart Parking Assistance', display_image)
        
        # Handle keyboard input
        key = cv.waitKey(1) & 0xFF
        
        # Check for number keys 1-9
        if ord('1') <= key <= ord('9'):
            slot_num = key - ord('1')  # Convert to 0-based index
            if slot_num < len(slot_centers):
                last_selected_slot[0] = slot_num
                current_variation = 0.0  # Reset variation for new slot
                print(f"Selected parking slot #{slot_num + 1}")
                
                # Save if requested
                if save_output:
                    output_path = 'parking_path.jpg'
                    cv.imwrite(output_path, display_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
                    print(f"\nResults saved to {output_path}")
            else:
                print(f"Invalid slot number. Please enter a number between 1 and {len(slot_centers)}")
        
        # Generate new path variation on spacebar
        elif key == ord(' '):
            if last_selected_slot[0] is not None:
                current_variation = np.random.uniform(-1.0, 1.0)
                print(f"Generating new path variation: {current_variation:.2f}")
        
        # Check for rotation keys
        elif key == ord('z'):  # Rotate left
            car.angle += math.radians(5)  # Rotate 5 degrees left
        elif key == ord('x'):  # Rotate right
            car.angle -= math.radians(5)  # Rotate 5 degrees right
        
        # Quit on 'q'
        elif key == ord('q'):
            break
    
    cv.destroyAllWindows()

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