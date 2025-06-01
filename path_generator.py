import math
import cv2
import numpy as np

class PathGenerator:
    def __init__(self):
        """Initialize the Path Generator"""
        self.path_points = []
    
    def generate_parking_path(self, current_box, target_box):
        """
        Generate a path from current position to target parking spot
        
        Args:
            current_box: Dictionary with 'center' and 'angle' keys
            target_box: Dictionary with 'center' and 'angle' keys
            
        Returns:
            List of path points with x, y, angle, and action
        """
        if not current_box or not target_box:
            print("Both current and target boxes must be defined")
            return []
        
        # Clear previous path
        self.path_points = []
        
        # Get centers and angles
        current_center = current_box['center']
        target_center = target_box['center']
        current_angle = current_box['angle']
        target_angle = target_box['angle']
        
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
        
        return self.path_points
    
    def draw_path(self, frame, path_points):
        """
        Draw the generated path on the frame
        
        Args:
            frame: Image to draw on
            path_points: List of path points with x, y, angle, and action
            
        Returns:
            Frame with path drawn on it
        """
        if not path_points:
            return frame
        
        # Create a copy of the frame
        frame_with_path = frame.copy()
        
        # Draw path points and lines
        for i in range(len(path_points) - 1):
            pt1 = (path_points[i]['x'], path_points[i]['y'])
            pt2 = (path_points[i+1]['x'], path_points[i+1]['y'])
            
            # Draw line between points
            cv2.line(frame_with_path, pt1, pt2, (255, 0, 0), 2)
            
            # Draw point
            cv2.circle(frame_with_path, pt1, 3, (255, 0, 0), -1)
        
        # Draw the last point
        last_pt = (path_points[-1]['x'], path_points[-1]['y'])
        cv2.circle(frame_with_path, last_pt, 3, (255, 0, 0), -1)
        
        # Draw direction indicators at key points
        for i in range(0, len(path_points), 5):
            pt = path_points[i]
            angle = pt['angle']
            center = (pt['x'], pt['y'])
            
            # Draw direction arrow
            end_x = int(center[0] + 20 * math.cos(math.radians(angle)))
            end_y = int(center[1] + 20 * math.sin(math.radians(angle)))
            cv2.arrowedLine(frame_with_path, center, (end_x, end_y), (0, 255, 255), 2)
        
        return frame_with_path
    
    def generate_parking_instructions(self, path_points, target_box):
        """
        Generate step-by-step parking instructions
        
        Args:
            path_points: List of path points with x, y, angle, and action
            target_box: Dictionary with 'angle' key
            
        Returns:
            Dictionary with instructions and difficulty
        """
        if not path_points:
            return {"error": "No path generated yet"}
        
        # Count turns and moves
        turns = sum(1 for pt in path_points if pt['action'] == 'turn')
        moves = sum(1 for pt in path_points if pt['action'] == 'move')
        
        # Calculate total angle change
        total_angle_change = 0
        for i in range(1, len(path_points)):
            angle_diff = path_points[i]['angle'] - path_points[i-1]['angle']
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
            total_angle_change += angle_diff
        
        # Generate instructions
        instructions = [
            "1. Start at the current position (red box)",
            f"2. Turn {'left' if total_angle_change < 0 else 'right'} approximately {abs(total_angle_change):.1f} degrees",
            f"3. Move forward approximately {moves * 5} units",
            f"4. Turn {'left' if target_box['angle'] - path_points[-2]['angle'] < 0 else 'right'} to align with the target parking spot",
            "5. Park in the target spot (green box)"
        ]
        
        # Calculate parking difficulty
        difficulty = "Easy"
        if abs(total_angle_change) > 90 or moves > 15:
            difficulty = "Moderate"
        if abs(total_angle_change) > 150 or moves > 25:
            difficulty = "Difficult"
        
        return {
            "instructions": instructions,
            "difficulty": difficulty,
            "total_angle_change": total_angle_change,
            "moves": moves
        }

# Example usage
if __name__ == "__main__":
    # Create a path generator
    path_gen = PathGenerator()
    
    # Example boxes
    current_box = {
        'center': [100, 100],
        'angle': 0
    }
    
    target_box = {
        'center': [300, 300],
        'angle': 90
    }
    
    # Generate path
    path_points = path_gen.generate_parking_path(current_box, target_box)
    
    # Generate instructions
    result = path_gen.generate_parking_instructions(path_points, target_box)
    
    # Print instructions
    print("\n=== Parking Instructions ===")
    for instruction in result["instructions"]:
        print(instruction)
    print(f"\nParking Difficulty: {result['difficulty']}") 