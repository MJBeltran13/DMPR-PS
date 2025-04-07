import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.special import comb

class CarParameters:
    def __init__(self, length=208, width=116, wheelbase=150, min_turn_radius=200, max_steering_angle=35):
        self.length = length
        self.width = width
        self.wheelbase = wheelbase
        self.min_turn_radius = min_turn_radius
        self.max_steering_angle = np.radians(max_steering_angle)

# Function to read box parameters from a CSV file
def read_csv(filename='box_parameters.csv'):
    box_data = []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            box_data.append(row)
    return box_data

# Function to generate a simple straight path
def generate_straight_path(start_pos, target_pos, num_points=20):
    """Generate a simple straight line path between two points."""
    x1, y1 = start_pos
    x2, y2 = target_pos
    path_points = []

    for i in range(num_points + 1):
        t = i / num_points
        point_x = int(x1 + t * (x2 - x1))
        point_y = int(y1 + t * (y2 - y1))
        path_points.append((point_x, point_y))

    return path_points

# Function to generate a Bezier curve path
def generate_bezier_path(start_pos, target_pos, control_points=None, num_points=20):
    """Generate a smooth Bezier curve path."""
    
    # If control points are not provided, create default ones
    if control_points is None:
        # Calculate middle point
        mid_x = (start_pos[0] + target_pos[0]) / 2
        mid_y = (start_pos[1] + target_pos[1]) / 2
        
        # Create two control points to form an S-curve
        control_point1 = (start_pos[0], mid_y)
        control_point2 = (target_pos[0], mid_y)
        
        control_points = [control_point1, control_point2]
    
    # All points including start, control points, and target
    points = [start_pos] + control_points + [target_pos]
    n = len(points) - 1  # Degree of the Bezier curve
    
    path_points = []
    for i in range(num_points + 1):
        t = i / num_points
        point = np.zeros(2)
        
        # Calculate the Bezier point using the Bernstein polynomial form
        for j in range(n + 1):
            coeff = comb(n, j) * (t ** j) * ((1 - t) ** (n - j))
            point[0] += coeff * points[j][0]
            point[1] += coeff * points[j][1]
            
        path_points.append((int(point[0]), int(point[1])))
    
    return path_points

# Function for RRT (Rapidly-exploring Random Tree) path planning
def generate_rrt_path(start_pos, target_pos, obstacles=None, max_iterations=1000, step_size=20):
    """Generate a path using RRT algorithm."""
    if obstacles is None:
        obstacles = []  # List of obstacles, each as (x, y, radius)
    
    # Initialize the tree with the start position
    tree = [start_pos]
    parent = {0: None}  # Dictionary to keep track of parent nodes
    
    for i in range(max_iterations):
        # With some probability, sample the target position
        if random.random() < 0.1:
            random_point = target_pos
        else:
            # Generate a random point in the workspace
            random_point = (
                random.randint(min(start_pos[0], target_pos[0]) - 100, max(start_pos[0], target_pos[0]) + 100),
                random.randint(min(start_pos[1], target_pos[1]) - 100, max(start_pos[1], target_pos[1]) + 100)
            )
        
        # Find the nearest node in the tree
        nearest_idx = find_nearest_node(tree, random_point)
        nearest_node = tree[nearest_idx]
        
        # Steer towards the random point with a maximum step size
        new_node = steer(nearest_node, random_point, step_size)
        
        # Check if the new node is collision-free
        if is_collision_free(nearest_node, new_node, obstacles):
            # Add the new node to the tree
            tree.append(new_node)
            parent[len(tree) - 1] = nearest_idx
            
            # Check if we've reached the target
            if distance(new_node, target_pos) < step_size:
                # Construct the path by backtracking from the target
                path = [target_pos, new_node]
                current_idx = len(tree) - 1
                
                while parent[current_idx] is not None:
                    current_idx = parent[current_idx]
                    path.append(tree[current_idx])
                
                # Reverse the path (from start to target)
                path.reverse()
                return path
    
    # If we couldn't find a path, return a straight line path
    print("RRT could not find a path, returning straight path")
    return generate_straight_path(start_pos, target_pos)

# Helper function to find the nearest node in the tree
def find_nearest_node(tree, point):
    """Find the nearest node in the tree to the given point."""
    distances = [distance(node, point) for node in tree]
    return distances.index(min(distances))

# Helper function to calculate the distance between two points
def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Helper function to steer towards a point
def steer(from_point, to_point, step_size):
    """Steer from one point towards another with a maximum step size."""
    dist = distance(from_point, to_point)
    
    if dist <= step_size:
        return to_point
    
    # Calculate the direction vector
    direction = (
        (to_point[0] - from_point[0]) / dist,
        (to_point[1] - from_point[1]) / dist
    )
    
    # New point at step_size distance from from_point in the direction of to_point
    new_point = (
        int(from_point[0] + direction[0] * step_size),
        int(from_point[1] + direction[1] * step_size)
    )
    
    return new_point

# Helper function to check for collisions
def is_collision_free(point1, point2, obstacles):
    """Check if the path between two points is collision-free."""
    for obstacle in obstacles:
        obs_x, obs_y, obs_radius = obstacle
        
        # Check if either endpoint is inside the obstacle
        if distance((obs_x, obs_y), point1) <= obs_radius or distance((obs_x, obs_y), point2) <= obs_radius:
            return False
        
        # Check if the line segment intersects the obstacle
        # (Simplified check for demonstration)
        line_length = distance(point1, point2)
        
        if line_length == 0:
            continue
        
        # Calculate the closest point on the line segment to the obstacle center
        t = max(0, min(1, ((obs_x - point1[0]) * (point2[0] - point1[0]) + 
                           (obs_y - point1[1]) * (point2[1] - point1[1])) / (line_length ** 2)))
        
        closest_point = (
            point1[0] + t * (point2[0] - point1[0]),
            point1[1] + t * (point2[1] - point1[1])
        )
        
        # Check if the closest point is within the obstacle
        if distance((obs_x, obs_y), closest_point) <= obs_radius:
            return False
    
    return True

# Function to generate paths between all box positions
def generate_path_between_boxes(box_data, car_params, path_type='bezier'):
    """Generate paths between boxes based on the specified path type."""
    
    # Find which box is the target position and which is the current position
    target_box = None
    current_box = None
    
    for box in box_data:
        if 'Label' in box:
            if 'Target' in box['Label']:
                target_box = box
            elif 'Current' in box['Label']:
                current_box = box
        elif box['Box'] == 'Red':  # Fallback if no labels
            target_box = box
        elif box['Box'] == 'Green':
            current_box = box
    
    if not target_box or not current_box:
        print("Could not identify target and current positions")
        return {}
    
    # Extract parameters
    target_top_left_x = int(target_box['Top Left X'])
    target_top_left_y = int(target_box['Top Left Y'])
    target_width = int(target_box['Width'])
    target_height = int(target_box['Height'])
    target_angle = float(target_box['Angle'])
    
    current_top_left_x = int(current_box['Top Left X'])
    current_top_left_y = int(current_box['Top Left Y'])
    current_width = int(current_box['Width'])
    current_height = int(current_box['Height'])
    current_angle = float(current_box['Angle'])
    
    # Calculate the centers of the boxes
    target_center = (target_top_left_x + target_width // 2, target_top_left_y + target_height // 2)
    current_center = (current_top_left_x + current_width // 2, current_top_left_y + current_height // 2)
    
    # Generate path based on the specified type
    path_points = []
    if path_type == 'straight':
        path_points = generate_straight_path(current_center, target_center)
    elif path_type == 'bezier':
        path_points = generate_bezier_path(current_center, target_center)
    elif path_type == 'rrt':
        path_points = generate_rrt_path(current_center, target_center)
    else:
        print(f"Unknown path type: {path_type}, using Bezier path")
        path_points = generate_bezier_path(current_center, target_center)
    
    # Convert path to x and y coordinates
    path_x = [point[0] for point in path_points]
    path_y = [point[1] for point in path_points]
    
    return {'path': (path_x, path_y), 'current': current_center, 'target': target_center}

# Function to plot the paths with box positions
def plot_paths_with_boxes(paths, box_data):
    plt.figure(figsize=(10, 8))
    
    # Plot the path
    path_x, path_y = paths['path']
    plt.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
    
    # Mark start and end points
    plt.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
    plt.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='End')
    
    # Plot current and target positions as rectangles
    for box in box_data:
        # Extract parameters
        top_left_x = int(box['Top Left X'])
        top_left_y = int(box['Top Left Y'])
        width = int(box['Width'])
        height = int(box['Height'])
        angle = float(box['Angle'])
        
        # Calculate the center of the box
        center_x = top_left_x + width // 2
        center_y = top_left_y + height // 2
        
        # Create a rectangle patch
        rect = plt.Rectangle((top_left_x, top_left_y), width, height, 
                            angle=angle, 
                            edgecolor='r' if 'Red' in box['Box'] else 'g',
                            facecolor='none',
                            linewidth=2,
                            label=f"{box['Box']} Box")
        plt.gca().add_patch(rect)
        
        # Plot the center
        plt.plot(center_x, center_y, 'o', 
                color='r' if 'Red' in box['Box'] else 'g', 
                markersize=5)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Generated Path between Current and Target Positions')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # Equal aspect ratio
    plt.show()

if __name__ == '__main__':
    # Read box parameters from CSV
    box_data = read_csv()
    
    if not box_data:
        print("No data found in box_parameters.csv")
        exit(1)
    
    # Initialize car parameters
    car_params = CarParameters()
    
    # Generate path between the boxes
    print("Generating path between boxes...")
    paths = generate_path_between_boxes(box_data, car_params, path_type='bezier')
    
    if not paths:
        print("Failed to generate path")
        exit(1)
    
    # Plot the paths with box positions
    plot_paths_with_boxes(paths, box_data)
    
    print("Path generation complete. The plot shows the path from the current position to the target position.") 