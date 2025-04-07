"""Smart Parking Assistance System with realistic car path planning."""
import cv2 as cv
import argparse
import numpy as np
import csv
from datetime import datetime

class DraggableBox:
    def __init__(self, top_left, color, width, height):
        self.top_left = top_left
        self.color = color
        self.width = width
        self.height = height
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0
        self.angle = 0  # Initialize angle

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

        # Display coordinates and angle
        text = f"({center_x}, {center_y}), {self.angle}°"
        cv.putText(image, text, (self.top_left[0] + 5, self.top_left[1] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)

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


def mouse_callback(event, x, y, flags, param):
    red_box, green_box, original_image = param

    if event == cv.EVENT_LBUTTONDOWN:
        if red_box.is_inside(x, y):
            red_box.start_drag(x, y)
        elif green_box.is_inside(x, y):
            green_box.start_drag(x, y)

    elif event == cv.EVENT_MOUSEMOVE:
        if red_box.dragging:
            red_box.update_position(x, y)
        elif green_box.dragging:
            green_box.update_position(x, y)

        # Redraw the image with updated box positions
        display_image = original_image.copy()
        red_box.draw(display_image)
        green_box.draw(display_image)
        cv.imshow('Image Display', display_image)

    elif event == cv.EVENT_LBUTTONUP:
        red_box.stop_drag()
        green_box.stop_drag()


# Function to append box parameters to a CSV file with a timestamp
def save_to_csv(red_box, green_box, filename='box_parameters.csv'):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a header if the file is empty
        if file.tell() == 0:
            writer.writerow(['Timestamp', 'Box', 'Top Left X', 'Top Left Y', 'Width', 'Height', 'Angle'])
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Write parameters with timestamp
        writer.writerow([timestamp, 'Red', red_box.top_left[0], red_box.top_left[1], red_box.width, red_box.height, red_box.angle])
        writer.writerow([timestamp, 'Green', green_box.top_left[0], green_box.top_left[1], green_box.width, green_box.height, green_box.angle])


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

    # Initialize draggable boxes
    red_box = DraggableBox((50, 50), red_color, box_width, box_height)
    green_box = DraggableBox((image.shape[1] - box_width - 50, 50), green_color, box_width, box_height)

    # Set up window and mouse callback
    cv.namedWindow('Image Display')
    cv.setMouseCallback('Image Display', mouse_callback, param=(red_box, green_box, image.copy()))

    # Display the image
    display_image = image.copy()
    red_box.draw(display_image)
    green_box.draw(display_image)
    cv.imshow('Image Display', display_image)

    # Wait for 'q' to quit
    while True:
        key = cv.waitKey(1) & 0xFF

        # Rotate red box with 'c' and 'v' keys
        if key == ord('c'):
            red_box.rotate(-5)
        elif key == ord('v'):
            red_box.rotate(5)

        # Rotate green box with 'z' and 'x' keys
        elif key == ord('z'):
            green_box.rotate(-5)
        elif key == ord('x'):
            green_box.rotate(5)

        # Save parameters to CSV when spacebar is pressed
        elif key == ord(' '):
            save_to_csv(red_box, green_box)
            print(f"Parameters saved to box_parameters.csv at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Redraw the image with updated box positions and rotations
        display_image = image.copy()
        red_box.draw(display_image)
        green_box.draw(display_image)
        cv.imshow('Image Display', display_image)

        if key == ord('q'):
            break

    cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Image Display')
    parser.add_argument('--image', default='data/images/ps2.0/testing/outdoor-shadow/0035.jpg',
                      help='Path to input image')
    
    args = parser.parse_args()
    
    # Process image
    print(f"Displaying image: {args.image}")
    process_image(args.image)

if __name__ == '__main__':
    main() 