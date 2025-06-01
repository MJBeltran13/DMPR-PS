# Smart Parking Assistant

A computer vision-based system that helps generate parking paths and instructions between a current position and a target parking spot.

## Features

- Interactive box drawing for current position and target parking spot
- Direction control with rotation capabilities
- Path generation between current position and target
- Step-by-step parking instructions
- Visual path visualization with direction indicators
- Parking difficulty assessment

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install opencv-python numpy
```

## Usage

Run the smart parking assistant with:

```bash
python smart_path_assist.py
```

### Command Line Arguments

- `--image`: Specify the image file path (default: 'data/images/ps2.0/testing/outdoor-normal daylight/193.jpg')
- `--spots`: Specify the file to save/load box coordinates (default: 'parking_spots.txt')

Example:
```bash
python smart_path_assist.py --image my_parking_image.jpg --spots my_boxes.txt
```

### Controls

- Press `c` to draw current position box (red)
- Press `t` to draw target parking spot box (green)
- Press `r` to rotate current box clockwise
- Press `l` to rotate current box counterclockwise
- Press `p` to generate parking path and instructions
- Press `s` to save boxes
- Press `q` to quit the application

## How It Works

1. **Box Definition**: First, define your current position (red box) and target parking spot (green box) by drawing rectangles.
2. **Direction Setting**: Adjust the direction of your vehicle by rotating the current box.
3. **Path Generation**: Generate a path from your current position to the target parking spot.
4. **Instructions**: The system provides step-by-step instructions for parking, including turning angles and distances.
5. **Visualization**: The path is visualized on the image with direction indicators at key points.

## Path Generation Algorithm

The path generation algorithm works in three phases:

1. **Initial Turn**: Turn the vehicle to face the target parking spot
2. **Forward Movement**: Move the vehicle towards the target
3. **Final Alignment**: Turn the vehicle to align with the target parking spot

The system calculates the optimal path based on the positions and orientations of both boxes.

## Parking Difficulty

The system assesses the parking difficulty based on:
- Total angle change required
- Distance to travel
- Complexity of maneuvers

Difficulty levels:
- **Easy**: Simple maneuvers with minimal turning
- **Moderate**: More complex maneuvers with significant turning
- **Difficult**: Complex maneuvers requiring precise control

## License

This project is open source and available under the MIT License. 