import csv
import numpy as np
import matplotlib.pyplot as plt

# Function to read box parameters from a CSV file
def read_csv(filename='box_parameters.csv'):
    box_data = []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            box_data.append(row)
    return box_data

# Function to generate a path based on box parameters
def generate_path(box_data):
    paths = {}
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

        # Generate a simple straight path from the center
        path_x = [center_x, center_x + 100 * np.cos(np.radians(angle))]
        path_y = [center_y, center_y + 100 * np.sin(np.radians(angle))]

        paths[box['Box']] = (path_x, path_y)
    return paths

# Function to plot the paths
def plot_paths(paths):
    plt.figure()
    for box, (path_x, path_y) in paths.items():
        plt.plot(path_x, path_y, marker='o', label=f'{box} Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Generated Paths from Box Parameters')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Read box parameters from CSV
    box_data = read_csv()

    # Generate paths based on the box parameters
    paths = generate_path(box_data)

    # Plot the generated paths
    plot_paths(paths) 