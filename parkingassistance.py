"""Parking Assistance System using DMPR-PS."""
import cv2 as cv
import torch
import argparse
import math
import numpy as np
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

def get_cardinal_direction(angle):
    """Convert angle to nearest cardinal/intercardinal direction."""
    # Normalize angle to 0-360
    angle = angle % 360
    
    # Define cardinal and intercardinal angles
    directions = {
        0: ("N", 0),     # North
        45: ("NE", 45),  # Northeast
        90: ("E", 90),   # East
        135: ("SE", 135), # Southeast
        180: ("S", 180), # South
        225: ("SW", 225), # Southwest
        270: ("W", 270), # West
        315: ("NW", 315), # Northwest
        360: ("N", 0)    # North
    }
    
    # Find closest direction
    closest_angle = min(directions.keys(), key=lambda x: abs(x - angle))
    return directions[closest_angle]

def draw_cardinal_path(image, start_point, end_point, color=(255, 255, 0), thickness=2):
    """Draw a path using cardinal/intercardinal directions that exactly reaches the endpoint."""
    x1, y1 = start_point
    x2, y2 = end_point
    
    # Calculate direct distance
    direct_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Calculate angle
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if angle < 0:
        angle += 360
        
    # Get nearest cardinal direction
    direction_name, cardinal_angle = get_cardinal_direction(angle)
    
    # Calculate endpoint if we went only in cardinal direction
    rad_angle = math.radians(cardinal_angle)
    cardinal_x = x1 + direct_distance * math.cos(rad_angle)
    cardinal_y = y1 + direct_distance * math.sin(rad_angle)
    
    # If the cardinal direction doesn't reach exactly, use two segments
    if abs(cardinal_x - x2) > 1 or abs(cardinal_y - y2) > 1:
        # First try horizontal then vertical
        path1 = [(x1, y1), (x2, y1), (x2, y2)]
        # Then try vertical then horizontal
        path2 = [(x1, y1), (x1, y2), (x2, y2)]
        
        # Calculate total distances for each path
        dist1 = abs(x2 - x1) + abs(y2 - y1)
        dist2 = dist1  # Same total distance
        
        # Choose the path with shorter first segment
        if abs(x2 - x1) < abs(y2 - y1):
            path = path1
            directions = ["E" if x2 > x1 else "W", "S" if y2 > y1 else "N"]
        else:
            path = path2
            directions = ["S" if y2 > y1 else "N", "E" if x2 > x1 else "W"]
        
        # Draw the two segments
        cv.line(image, (int(path[0][0]), int(path[0][1])), 
                (int(path[1][0]), int(path[1][1])), color, thickness)
        cv.line(image, (int(path[1][0]), int(path[1][1])), 
                (int(path[2][0]), int(path[2][1])), color, thickness)
        
        # Calculate midpoints for labels
        mid1 = ((path[0][0] + path[1][0])/2, (path[0][1] + path[1][1])/2)
        mid2 = ((path[1][0] + path[2][0])/2, (path[1][1] + path[2][1])/2)
        
        # Draw direction labels
        cv.putText(image, f"{directions[0]}", 
                  (int(mid1[0]), int(mid1[1])), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv.putText(image, f"{directions[1]}", 
                  (int(mid2[0]), int(mid2[1])), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return path[1], direct_distance, f"{directions[0]}-{directions[1]}"
    else:
        # Draw single line if cardinal direction reaches exactly
        cv.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return (x2, y2), direct_distance, direction_name

def process_image(image_path, model, device, save_output=False):
    """Process a single image and detect parking slots."""
    # Read image
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Get image dimensions and car center position
    height, width = image.shape[:2]
    car_center_x = width // 2
    car_center_y = height // 2
    
    # Detect marking points
    pred_points = detect_marking_points(model, image, config.CONFID_THRESH_FOR_POINT, device)
    
    # Detect parking slots
    slots = None
    if pred_points:
        marking_points = list(list(zip(*pred_points))[1])
        slots = inference_slots(marking_points)
        
        # Print detailed information for each slot
        print("\nDetected Parking Slots Information:")
        print("-" * 50)
        slot_centers = []  # Store slot centers for later use
        
        for idx, slot in enumerate(slots):
            info = get_slot_info(image, marking_points, slot)
            print(f"\nParking Slot #{idx + 1}:")
            print(f"Center coordinates (x, y): ({info['center'][0]:.2f}, {info['center'][1]:.2f})")
            print(f"Angle: {info['angle']:.2f}°")
            print(f"Point 1 (x, y): ({info['point1'][0]}, {info['point1'][1]})")
            print(f"Point 2 (x, y): ({info['point2'][0]}, {info['point2'][1]})")
            
            # Draw slot number and dot on image
            center_x = int(info['center'][0])
            center_y = int(info['center'][1])
            slot_centers.append((center_x, center_y))
            
            # Draw dot
            cv.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)  # Green filled circle
            # Draw slot number next to the dot
            cv.putText(image, f"#{idx + 1}", 
                      (center_x + 10, center_y + 10),  # Offset text position
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Plot results
    plot_points(image, pred_points)
    plot_slots(image, pred_points, slots)
    
    # Draw car marker in the center of the image
    cv.circle(image, (car_center_x, car_center_y), 10, (0, 0, 255), -1)  # Filled red circle
    cv.line(image, (car_center_x - 15, car_center_y), (car_center_x + 15, car_center_y), (0, 0, 255), 2)  # Horizontal line
    cv.line(image, (car_center_x, car_center_y - 15), (car_center_x, car_center_y + 15), (0, 0, 255), 2)  # Vertical line
    cv.putText(image, "CAR", (car_center_x + 15, car_center_y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display initial image
    cv.imshow('Parking Assistance', image)
    cv.waitKey(1)  # Show image without waiting
    
    while True:
        print("\nEnter parking slot number to connect (1-{}) or 'q' to quit: ".format(len(slot_centers)))
        choice = input()
        
        if choice.lower() == 'q':
            break
        
        try:
            slot_num = int(choice)
            if 1 <= slot_num <= len(slot_centers):
                # Create a copy of the original image
                display_image = image.copy()
                
                # Draw the connecting path for the selected slot
                center_x, center_y = slot_centers[slot_num - 1]
                intermediate_point, distance, directions = draw_cardinal_path(
                    display_image, 
                    (car_center_x, car_center_y), 
                    (center_x, center_y)
                )
                
                # Display total distance
                cv.putText(display_image, f"Total: {distance:.1f}px", 
                          (car_center_x + 20, car_center_y - 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display updated image
                cv.imshow('Parking Assistance', display_image)
                cv.waitKey(1)
                
                # Print direction information
                print(f"Direction to parking slot #{slot_num}: {directions}")
                print(f"Total distance: {distance:.1f} pixels")
                
                # Save if requested
                if save_output:
                    output_path = 'detected_slots.jpg'
                    cv.imwrite(output_path, display_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
                    print(f"\nResults saved to {output_path}")
            else:
                print("Invalid slot number. Please enter a number between 1 and", len(slot_centers))
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
    
    cv.waitKey(0)
    return pred_points, slots

def main():
    parser = argparse.ArgumentParser(description='Parking Assistance System')
    parser.add_argument('--weights', default='dmpr_pretrained_weights.pth',
                      help='Path to model weights')
    parser.add_argument('--image', default='data/images/ps2.0/testing/indoor-parking lot/008.jpg',
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
    pred_points, slots = process_image(args.image, model, device, args.save)
    
    if pred_points:
        print(f"\nSummary: Found {len(pred_points)} marking points and {len(slots) if slots else 0} parking slots")
    else:
        print("No parking slots detected")
    
    cv.destroyAllWindows()

if __name__ == '__main__':
    main() 