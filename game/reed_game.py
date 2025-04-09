import tkinter as tk
import math
import reeds_shepp
import time
import random

CAR_WIDTH = 40
CAR_LENGTH = 70
TURN_RADIUS = 50
STEP_SIZE = 2.0
COLLISION_DISTANCE = 45  # minimal clearance
ANIMATION_SPEED = 30    # milliseconds between frames
ROTATION_SPEED = 0.05   # radians per frame
MAX_STEERING_ANGLE = math.radians(35)  # maximum steering angle in radians
ACCELERATION = 0.5      # acceleration rate
MAX_SPEED = 5.0         # maximum speed
MIN_SPEED = 0.5         # minimum speed when moving

start = (600, 400, math.radians(180))
goal = (150, 150, math.radians(180))

def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

class CarSimulator:
    def __init__(self, master):
        self.master = master
        self.master.title("Smart Reverse Parking")
        self.canvas = tk.Canvas(master, width=800, height=600, bg='gray20')
        self.canvas.pack()

        self.label = tk.Label(master, text="", font=("Arial", 16), fg="white", bg="gray20")
        self.label.place(x=10, y=10)
        
        self.steering_angle_label = tk.Label(master, text="", font=("Arial", 12), fg="white", bg="gray20")
        self.steering_angle_label.place(x=10, y=50)
        
        # Direction indicator
        self.direction_label = tk.Label(master, text="", font=("Arial", 12), fg="yellow", bg="gray20")
        self.direction_label.place(x=10, y=80)
        
        # Speed indicator
        self.speed_label = tk.Label(master, text="", font=("Arial", 12), fg="cyan", bg="gray20")
        self.speed_label.place(x=10, y=110)
        
        # Distance indicator
        self.distance_label = tk.Label(master, text="", font=("Arial", 12), fg="orange", bg="gray20")
        self.distance_label.place(x=10, y=140)

        self.path = reeds_shepp.path_sample(start, goal, TURN_RADIUS, STEP_SIZE)
        self.index = 1
        self.car_id = None
        self.side_car_positions = self.get_side_car_positions(goal)
        
        # Current state
        self.current_pos = list(start)  # [x, y, theta]
        self.target_pos = None
        self.in_transition = False
        self.direction = "Reversing"  # or "Forward"
        self.speed = 0.0  # current speed
        self.steering_angle = 0.0  # current steering angle
        self.steering_target = 0.0  # target steering angle
        self.steering_rate = 0.1  # rate at which steering changes
        self.last_update_time = time.time()
        self.obstacle_detected = False
        self.obstacle_distance = 0
        self.obstacle_direction = 0
        self.manual_control = False  # Initialize manual_control flag

        self.forward_added = False  # only allow one forward adjustment
        self.show_path = False
        self.path_toggle_button = tk.Button(master, text="Show Path", 
                                           command=self.toggle_path_visibility,
                                           bg="gray30", fg="white")
        self.path_toggle_button.place(x=10, y=200)

        self.animate()
        
        # Draw parking slot
        self.draw_parking_slot()
        
        # Add key bindings for manual control
        self.master.bind('<Left>', self.steer_left)
        self.master.bind('<Right>', self.steer_right)
        self.master.bind('<Up>', self.accelerate)
        self.master.bind('<Down>', self.brake)
        self.master.bind('<space>', self.toggle_direction)
        
        # Add manual control indicator
        self.manual_control_label = tk.Label(master, text="Manual Control: Arrow keys + Space", 
                                            font=("Arial", 10), fg="lightgreen", bg="gray20")
        self.manual_control_label.place(x=10, y=170)
        
        # Add key binding for toggling path visibility
        self.master.bind('<p>', lambda e: self.toggle_path_visibility())
        
        # Add key binding for toggling manual control
        self.master.bind('<m>', lambda e: self.toggle_manual_control())

    def steer_left(self, event=None):
        if self.manual_control:
            self.steering_target = min(self.steering_target + 0.1, MAX_STEERING_ANGLE)
    
    def steer_right(self, event=None):
        if self.manual_control:
            self.steering_target = max(self.steering_target - 0.1, -MAX_STEERING_ANGLE)
    
    def accelerate(self, event=None):
        if self.manual_control:
            self.speed = min(self.speed + ACCELERATION, MAX_SPEED)
    
    def brake(self, event=None):
        if self.manual_control:
            self.speed = max(self.speed - ACCELERATION, 0)
    
    def toggle_direction(self, event=None):
        if self.manual_control:
            if self.direction == "Forward":
                self.direction = "Reversing"
            else:
                self.direction = "Forward"
            self.speed = 0  # Reset speed when changing direction

    def draw_parking_slot(self):
        # Draw parking lines around the goal position
        x, y, theta = goal
        length = CAR_LENGTH + 20
        width = CAR_WIDTH + 20
        
        # Draw white dashed lines for parking slot
        corners = self.get_rotated_rect(x, y, width, length, theta)
        for i in range(4):
            j = (i + 1) % 4
            self.canvas.create_line(corners[i][0], corners[i][1], 
                                   corners[j][0], corners[j][1], 
                                   fill="white", dash=(5, 5))
        
        # Draw parking spot number
        self.canvas.create_text(x, y, text="P", fill="white", font=("Arial", 20, "bold"))

    def get_side_car_positions(self, center_pose):
        x, y, theta = center_pose
        space = CAR_WIDTH + 5
        dx = space * math.cos(theta)
        dy = space * math.sin(theta)

        left = (x - dx, y - dy, theta)
        right = (x + dx, y + dy, theta)
        return [left, right]

    def draw_car(self, x, y, theta, color="green"):
        # Draw car body
        corners = self.get_rotated_rect(x, y, CAR_WIDTH, CAR_LENGTH, theta)
        car_id = self.canvas.create_polygon(corners, fill=color, outline="black")
        
        # Draw front windshield to indicate front of car
        front_x = x + (CAR_LENGTH/3) * math.cos(theta)
        front_y = y + (CAR_LENGTH/3) * math.sin(theta)
        windshield = self.get_rotated_rect(front_x, front_y, CAR_WIDTH-10, 5, theta)
        self.canvas.create_polygon(windshield, fill="lightblue")
        
        # Draw wheels
        self.draw_wheels(x, y, theta)
        
        return car_id

    def draw_wheels(self, x, y, theta):
        wheel_width = 8
        wheel_length = 15
        wheel_offset = CAR_WIDTH/2 + 2
        
        # Calculate wheel positions
        wheel_offsets = [
            # Front right
            (CAR_LENGTH/2.5, wheel_offset),
            # Front left
            (CAR_LENGTH/2.5, -wheel_offset),
            # Rear right
            (-CAR_LENGTH/2.5, wheel_offset),
            # Rear left
            (-CAR_LENGTH/2.5, -wheel_offset)
        ]
        
        # For front wheels, we calculate steering angle based on path
        if not self.manual_control and self.index > 1 and self.index < len(self.path):
            # Calculate steering based on three points
            prev = self.path[self.index-1]
            curr = self.path[self.index]
            
            # Get change in heading
            delta_theta = curr[2] - prev[2]
            # Normalize to small angle
            delta_theta = (delta_theta + math.pi) % (2 * math.pi) - math.pi
            
            # Convert to steering angle (exaggerate for visibility)
            self.steering_target = delta_theta * 2  # Amplify for visual effect
            # Limit to realistic values
            self.steering_target = max(min(self.steering_target, MAX_STEERING_ANGLE), -MAX_STEERING_ANGLE)
        
        # Smoothly adjust steering angle
        if self.steering_angle < self.steering_target:
            self.steering_angle = min(self.steering_angle + self.steering_rate, self.steering_target)
        elif self.steering_angle > self.steering_target:
            self.steering_angle = max(self.steering_angle - self.steering_rate, self.steering_target)
        
        # Draw each wheel
        for i, (wheel_x_offset, wheel_y_offset) in enumerate(wheel_offsets):
            # Calculate wheel center in car coordinates
            wheel_x = x + wheel_x_offset * math.cos(theta) - wheel_y_offset * math.sin(theta)
            wheel_y = y + wheel_x_offset * math.sin(theta) + wheel_y_offset * math.cos(theta)
            
            # For front wheels, apply steering angle
            wheel_angle = theta
            if i < 2:  # Front wheels
                wheel_angle += self.steering_angle
            
            # Draw the wheel
            wheel_corners = self.get_rotated_rect(wheel_x, wheel_y, wheel_width, wheel_length, wheel_angle)
            self.canvas.create_polygon(wheel_corners, fill="black")

    def draw_slot_with_cars(self, x, y, theta):
        self.draw_car(x, y, theta, "red")

        for car in self.side_car_positions:
            self.draw_car(car[0], car[1], car[2], "red")

    def get_rotated_rect(self, cx, cy, w, h, angle_rad):
        corners = []
        for dx, dy in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]:
            x = cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            y = cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            corners.append((x, y))
        return corners

    def get_steering_instruction(self, prev_theta, curr_theta):
        delta = (math.degrees(curr_theta - prev_theta) + 360) % 360
        if delta > 180:
            delta -= 360
        
        # Calculate steering angle in degrees
        steering_angle = delta
        
        if delta > 2:
            return f"Turn Steering Right ({steering_angle:.1f}°)"
        elif delta < -2:
            return f"Turn Steering Left ({abs(steering_angle):.1f}°)"
        else:
            return "Straighten Steering"

    def check_collision(self, x, y):
        for sx, sy, _ in self.side_car_positions:
            distance = math.hypot(x - sx, y - sy)
            if distance < COLLISION_DISTANCE:
                self.obstacle_detected = True
                self.obstacle_distance = distance
                self.obstacle_direction = math.atan2(sy - y, sx - x)
                return True
        self.obstacle_detected = False
        return False

    def insert_forward_move(self):
        # Step back 5 points and go forward a little
        current = self.path[self.index]
        heading = current[2]
        new_x = current[0] + 40 * math.cos(heading)
        new_y = current[1] + 40 * math.sin(heading)

        # Insert forward movement into path
        forward_path = reeds_shepp.path_sample(current, (new_x, new_y, heading), TURN_RADIUS, STEP_SIZE)
        resume_path = reeds_shepp.path_sample((new_x, new_y, heading), goal, TURN_RADIUS, STEP_SIZE)

        self.path = self.path[:self.index] + forward_path + resume_path
        self.forward_added = True
        self.direction = "Forward"

    def smooth_rotation(self, current, target):
        """Smoothly rotate from current angle to target angle"""
        current_theta = current[2]
        target_theta = target[2]
        
        # Calculate angle difference
        delta = target_theta - current_theta
        # Normalize to [-pi, pi]
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        
        # If the difference is small enough, just set to target
        if abs(delta) < ROTATION_SPEED:
            return target
        
        # Otherwise, rotate by ROTATION_SPEED in the right direction
        if delta > 0:
            new_theta = current_theta + ROTATION_SPEED
        else:
            new_theta = current_theta - ROTATION_SPEED
            
        # Create new position
        return [current[0], current[1], new_theta]

    def draw_obstacle_warning(self):
        if self.obstacle_detected:
            x, y, _ = self.current_pos
            # Draw warning circle
            self.canvas.create_oval(x-30, y-30, x+30, y+30, outline="red", width=2)
            
            # Draw warning text
            self.canvas.create_text(x, y-40, text="OBSTACLE!", fill="red", font=("Arial", 12, "bold"))
            
            # Draw distance indicator
            distance_text = f"Distance: {self.obstacle_distance:.1f}"
            self.canvas.create_text(x, y-20, text=distance_text, fill="red", font=("Arial", 10))

    def draw_parking_guidelines(self):
        # Draw guidelines to help with parking
        x, y, theta = goal
        
        # Draw center line
        center_line_length = 100
        center_x1 = x - center_line_length * math.cos(theta)
        center_y1 = y - center_line_length * math.sin(theta)
        center_x2 = x + center_line_length * math.cos(theta)
        center_y2 = y + center_line_length * math.sin(theta)
        self.canvas.create_line(center_x1, center_y1, center_x2, center_y2, 
                               fill="yellow", dash=(5, 5), width=2)
        
        # Draw distance markers
        for i in range(1, 4):
            marker_distance = i * 20
            marker_x = x - marker_distance * math.cos(theta)
            marker_y = y - marker_distance * math.sin(theta)
            marker_width = 10
            marker_height = 5
            marker_corners = self.get_rotated_rect(marker_x, marker_y, marker_width, marker_height, theta)
            self.canvas.create_polygon(marker_corners, fill="yellow")

    def draw_path(self):
        """Draw the path that the car will follow"""
        if not self.show_path:
            return
            
        # Draw the path as a series of connected lines
        for i in range(1, len(self.path)):
            x1, y1, _ = self.path[i-1]
            x2, y2, _ = self.path[i]
            
            # Draw the path line
            self.canvas.create_line(x1, y1, x2, y2, fill="yellow", width=2)
            
            # Draw direction indicators at regular intervals
            if i % 5 == 0:  # Draw an arrow every 5 points
                # Calculate the direction
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    # Normalize
                    dx /= length
                    dy /= length
                    
                    # Draw a small arrow
                    arrow_size = 5
                    arrow_x = x1 + dx * length * 0.5
                    arrow_y = y1 + dy * length * 0.5
                    
                    # Calculate perpendicular direction for arrow
                    perp_x = -dy
                    perp_y = dx
                    
                    # Draw arrow head
                    self.canvas.create_line(
                        arrow_x, arrow_y,
                        arrow_x - arrow_size * dx + arrow_size * perp_x,
                        arrow_y - arrow_size * dy + arrow_size * perp_y,
                        fill="yellow", width=2
                    )
                    self.canvas.create_line(
                        arrow_x, arrow_y,
                        arrow_x - arrow_size * dx - arrow_size * perp_x,
                        arrow_y - arrow_size * dy - arrow_size * perp_y,
                        fill="yellow", width=2
                    )
        
        # Draw start and end points
        if len(self.path) > 0:
            start_x, start_y, _ = self.path[0]
            self.canvas.create_oval(start_x-5, start_y-5, start_x+5, start_y+5, fill="green")
            self.canvas.create_text(start_x, start_y-15, text="Start", fill="green", font=("Arial", 10, "bold"))
            
            end_x, end_y, _ = self.path[-1]
            self.canvas.create_oval(end_x-5, end_y-5, end_x+5, end_y+5, fill="red")
            self.canvas.create_text(end_x, end_y-15, text="Goal", fill="red", font=("Arial", 10, "bold"))

    def animate(self):
        self.canvas.delete("all")
        
        # Draw the path if enabled
        self.draw_path()
        
        # Draw the parking slot and cars
        self.draw_parking_slot()
        self.draw_slot_with_cars(goal[0], goal[1], goal[2])
        self.draw_parking_guidelines()

        if self.manual_control:
            # Manual control mode
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
            
            # Update position based on speed and direction
            if self.speed > 0:
                # Calculate movement direction
                direction_factor = 1 if self.direction == "Forward" else -1
                
                # Update position
                x, y, theta = self.current_pos
                new_x = x + direction_factor * self.speed * math.cos(theta) * dt
                new_y = y + direction_factor * self.speed * math.sin(theta) * dt
                
                # Update heading based on steering angle
                new_theta = theta + direction_factor * self.speed * math.tan(self.steering_angle) * dt / CAR_LENGTH
                
                # Check for collisions
                if not self.check_collision(new_x, new_y):
                    self.current_pos = [new_x, new_y, new_theta]
                else:
                    # Slow down when collision detected
                    self.speed = max(0, self.speed - ACCELERATION * 2)
            
            # Gradually reduce speed when not accelerating
            if self.speed > 0:
                self.speed = max(0, self.speed - ACCELERATION * 0.1)
            
            # Draw car
            this_car = self.draw_car(self.current_pos[0], self.current_pos[1], self.current_pos[2], "green")
            
            # Update labels
            self.label.config(text="Manual Control Mode")
            self.steering_angle_label.config(text=f"Steering: {math.degrees(self.steering_angle):.1f}°")
            self.direction_label.config(text=f"Direction: {self.direction}")
            self.speed_label.config(text=f"Speed: {self.speed:.1f}")
            
            # Calculate distance to goal
            x, y, _ = self.current_pos
            goal_x, goal_y, _ = goal
            distance = math.hypot(goal_x - x, goal_y - y)
            self.distance_label.config(text=f"Distance to goal: {distance:.1f}")
            
            # Draw obstacle warning if needed
            self.draw_obstacle_warning()
            
            self.master.after(ANIMATION_SPEED, self.animate)
            return

        if self.index < len(self.path):
            # Get the current target position
            target_x, target_y, target_theta = self.path[self.index]
            
            # Get the current position
            x, y, theta = self.current_pos
            
            # Calculate distance to target
            distance = math.hypot(target_x - x, target_y - y)
            
            # Check collision
            if self.check_collision(x, y) and not self.forward_added:
                self.label.config(text="Obstacle detected! Driving forward to adjust...")
                self.insert_forward_move()
                self.direction = "Forward"
                self.master.after(800, self.animate)
                return
            
            # Determine if we're moving forward or backward
            if self.index > 1:
                prev_x, prev_y, _ = self.path[self.index-1]
                curr_x, curr_y, _ = self.path[self.index]
                
                # Calculate dot product to determine forward/reverse
                prev_vec = (curr_x - prev_x, curr_y - prev_y)
                next_vec = (target_x - curr_x, target_y - curr_y)
                dot_product = prev_vec[0]*next_vec[0] + prev_vec[1]*next_vec[1]
                
                # If dot product is negative, we changed direction
                if dot_product < 0:
                    if self.direction == "Forward":
                        self.direction = "Reversing"
                    else:
                        self.direction = "Forward"
            
            # Simple approach: directly set position and orientation to match path
            # This ensures the car follows the path exactly
            if distance < STEP_SIZE:
                # We're close enough to the target, move to next point
                self.index += 1
            else:
                # Move towards target
                dx = target_x - x
                dy = target_y - y
                length = math.sqrt(dx*dx + dy*dy)
                
                # Normalize
                dx /= length
                dy /= length
                
                # Move by step size
                new_x = x + dx * STEP_SIZE
                new_y = y + dy * STEP_SIZE
                
                # Calculate target orientation
                target_heading = math.atan2(dy, dx)
                
                # If moving backward, flip the target heading
                if self.direction == "Reversing":
                    target_heading = normalize_angle(target_heading + math.pi)
                
                # Smoothly rotate towards target heading
                angle_diff = normalize_angle(target_heading - theta)
                if abs(angle_diff) > ROTATION_SPEED:
                    if angle_diff > 0:
                        new_theta = theta + ROTATION_SPEED
                    else:
                        new_theta = theta - ROTATION_SPEED
                else:
                    new_theta = target_heading
                
                # Update position
                self.current_pos = [new_x, new_y, new_theta]
            
            # Get steering instruction
            instruction = ""
            if self.index > 1 and self.index < len(self.path):
                prev_theta = self.path[self.index-1][2]
                curr_theta = self.path[self.index][2]
                instruction = self.get_steering_instruction(prev_theta, curr_theta)
            
            # Draw car
            this_car = self.draw_car(self.current_pos[0], self.current_pos[1], self.current_pos[2], "green")
            
            # Update labels
            self.label.config(text=f"Step {self.index}: {instruction}")
            self.steering_angle_label.config(text=f"Heading: {math.degrees(self.current_pos[2]):.1f}°")
            self.direction_label.config(text=f"Direction: {self.direction}")
            
            # Calculate speed based on path curvature
            if self.index > 1 and self.index < len(self.path):
                prev_x, prev_y, _ = self.path[self.index-1]
                curr_x, curr_y, _ = self.path[self.index]
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                self.speed = math.hypot(dx, dy) / (ANIMATION_SPEED / 1000.0)
            else:
                self.speed = MIN_SPEED
                
            self.speed_label.config(text=f"Speed: {self.speed:.1f}")
            
            # Calculate distance to goal
            x, y, _ = self.current_pos
            goal_x, goal_y, _ = goal
            distance = math.hypot(goal_x - x, goal_y - y)
            self.distance_label.config(text=f"Distance to goal: {distance:.1f}")
            
            # Draw obstacle warning if needed
            self.draw_obstacle_warning()
            
            self.master.after(ANIMATION_SPEED, self.animate)
        else:
            self.label.config(text="Arrived! Straighten Wheel & Stop.")
            self.steering_angle_label.config(text=f"Final heading: {math.degrees(self.current_pos[2]):.1f}°")
            self.direction_label.config(text="Parking Complete!")
            self.speed_label.config(text="Speed: 0.0")
            
            # Calculate final distance
            x, y, _ = self.current_pos
            goal_x, goal_y, _ = goal
            distance = math.hypot(goal_x - x, goal_y - y)
            self.distance_label.config(text=f"Final distance: {distance:.1f}")
            
            # Draw success message
            self.canvas.create_text(400, 300, text="PARKING SUCCESSFUL!", 
                                   fill="green", font=("Arial", 24, "bold"))

    def toggle_manual_control(self):
        self.manual_control = not self.manual_control
        if self.manual_control:
            self.manual_control_label.config(text="Manual Control: ON", fg="green")
        else:
            self.manual_control_label.config(text="Manual Control: OFF", fg="lightgreen")

    def toggle_path_visibility(self):
        """Toggle the visibility of the path visualization"""
        self.show_path = not self.show_path
        if self.show_path:
            self.path_toggle_button.config(text="Hide Path", bg="green")
        else:
            self.path_toggle_button.config(text="Show Path", bg="gray30")

def setup_manual_control_toggle(app):
    app.master.bind('<m>', lambda e: app.toggle_manual_control())

root = tk.Tk()
app = CarSimulator(root)
setup_manual_control_toggle(app)
root.mainloop()
