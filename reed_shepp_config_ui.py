import os
import json
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox

class ReedSheppConfigUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Reed-Shepp Path Configuration")
        self.root.geometry("800x600")
        
        # Default config matches the original CONFIG dictionary
        self.config = {
            'TURNING_RADIUS': 30.0,
            'DISTANCE_SCALING': 4.0,
            'MIN_TURNING_RADIUS': 25.0,
            'MAX_TURNING_RADIUS': 50.0,
            'PATH_POINTS': 80,
            'PATH_COLOR': (0, 255, 0),
            'REVERSE_COLOR': (0, 0, 255),
            'PATH_THICKNESS': 2,
            'INDICATOR_COLOR': (50, 200, 50),
            'MAX_INDICATORS': 5,
            'INDICATOR_LENGTH': 10,
            'DEBUG_VISUALIZATION': False,
            'PATH_SMOOTHING': 0.1,
            'FAVOR_REVERSING': True,
            'SHOW_GRID': False,
            'GRID_SIZE': 50,
            'GRID_COLOR': (100, 100, 100),
        }
        
        # Additional parameters from the compute_reed_shepp_path function
        self.additional_params = {
            'FIXED_START_DISTANCE': 1.0,
            'SHORT_PATH_THRESHOLD': 100,
            'MEDIUM_PATH_THRESHOLD': 200,
            'LONG_PATH_THRESHOLD': 300,
            'SHORT_PATH_FACTOR': 0.15,
            'MEDIUM_PATH_FACTOR': 0.2,
            'LONG_PATH_FACTOR': 0.25,
            'SHORT_PATH_REDUCTION': 6,
            'MEDIUM_PATH_REDUCTION': 4,
            'LONG_PATH_REDUCTION': 2,
            'APPROACH_PATH_THRESHOLD': 180,
            'APPROACH_PATH_FACTOR': 0.4,
            'MAX_APPROACH_DISTANCE': 180,
            'NUM_REVERSE_SEGMENTS': 10,
            'MIN_PERP_OFFSET': 20,
            'MAX_PERP_OFFSET': 40,
            'PERP_OFFSET_FACTOR': 0.15,
            'MIN_MID_FACTOR': 0.4,
            'MAX_MID_FACTOR': 0.7,
            'STRAIGHT_CURVE_THRESHOLD': 0.9,
            'DIRECT_CURVE_THRESHOLD': 0.7,
            'SMALL_ANGLE_THRESHOLD': 30,
            'MODERATE_ANGLE_THRESHOLD': 60,
            'VERY_SHORT_CURVE_THRESHOLD': 50,
            'SHORT_CURVE_THRESHOLD': 100,
            'FIXED_SEGMENT_POINTS': 10,
            'MAX_REDUCTION_PERCENT': 0.8,
        }
        
        # Combine all parameters
        self.all_params = {**self.config, **self.additional_params}
        
        # Create the main notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_main_tab()
        self.create_path_tab()
        self.create_visual_tab()
        self.create_advanced_tab()
        
        # Create footer with action buttons
        self.create_footer()
        
        # Load config if available
        self.load_config()
    
    def create_main_tab(self):
        main_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_frame, text="Main Settings")
        
        # Create scrollable frame
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Turning radius parameters
        self.create_section_label(scrollable_frame, "Turning Radius Parameters")
        
        self.create_slider(scrollable_frame, "Turning Radius", "TURNING_RADIUS", 5, 100, 0.1)
        self.create_slider(scrollable_frame, "Distance Scaling", "DISTANCE_SCALING", 1, 10, 0.1)
        self.create_slider(scrollable_frame, "Min Turning Radius", "MIN_TURNING_RADIUS", 5, 50, 0.1)
        self.create_slider(scrollable_frame, "Max Turning Radius", "MAX_TURNING_RADIUS", 10, 100, 0.1)
        
        # Fixed start distance
        self.create_slider(scrollable_frame, "Fixed Start Distance", "FIXED_START_DISTANCE", 0, 50, 0.1)
    
    def create_path_tab(self):
        path_frame = ttk.Frame(self.notebook)
        self.notebook.add(path_frame, text="Path Generation")
        
        # Create scrollable frame
        canvas = tk.Canvas(path_frame)
        scrollbar = ttk.Scrollbar(path_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Path generation parameters
        self.create_section_label(scrollable_frame, "Path Generation Parameters")
        
        self.create_slider(scrollable_frame, "Total Path Points", "PATH_POINTS", 20, 200, 1)
        self.create_slider(scrollable_frame, "Fixed Segment Points", "FIXED_SEGMENT_POINTS", 3, 30, 1)
        self.create_slider(scrollable_frame, "Path Smoothing Factor", "PATH_SMOOTHING", 0, 1, 0.01)
        self.create_checkbox(scrollable_frame, "Favor Reversing", "FAVOR_REVERSING")
        
        # Path threshold settings
        self.create_section_label(scrollable_frame, "Path Distance Thresholds")
        
        self.create_slider(scrollable_frame, "Short Path Threshold", "SHORT_PATH_THRESHOLD", 50, 200, 1)
        self.create_slider(scrollable_frame, "Medium Path Threshold", "MEDIUM_PATH_THRESHOLD", 100, 300, 1)
        self.create_slider(scrollable_frame, "Long Path Threshold", "LONG_PATH_THRESHOLD", 200, 500, 1)
        
        # Path ratio factors
        self.create_section_label(scrollable_frame, "Path Ratio Factors")
        
        self.create_slider(scrollable_frame, "Short Path Factor", "SHORT_PATH_FACTOR", 0.05, 0.5, 0.01)
        self.create_slider(scrollable_frame, "Medium Path Factor", "MEDIUM_PATH_FACTOR", 0.05, 0.5, 0.01)
        self.create_slider(scrollable_frame, "Long Path Factor", "LONG_PATH_FACTOR", 0.05, 0.5, 0.01)
        
        # Point reduction parameters
        self.create_section_label(scrollable_frame, "Point Reduction Parameters")
        
        self.create_slider(scrollable_frame, "Short Path Reduction", "SHORT_PATH_REDUCTION", 1, 10, 1)
        self.create_slider(scrollable_frame, "Medium Path Reduction", "MEDIUM_PATH_REDUCTION", 1, 10, 1)
        self.create_slider(scrollable_frame, "Long Path Reduction", "LONG_PATH_REDUCTION", 0, 5, 1)
        self.create_slider(scrollable_frame, "Max Reduction Percent", "MAX_REDUCTION_PERCENT", 0.1, 0.95, 0.01)
    
    def create_visual_tab(self):
        visual_frame = ttk.Frame(self.notebook)
        self.notebook.add(visual_frame, text="Visual Settings")
        
        # Create scrollable frame
        canvas = tk.Canvas(visual_frame)
        scrollbar = ttk.Scrollbar(visual_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Visual parameters
        self.create_section_label(scrollable_frame, "Path Visualization")
        
        self.create_color_picker(scrollable_frame, "Path Color (Forward)", "PATH_COLOR")
        self.create_color_picker(scrollable_frame, "Path Color (Reverse)", "REVERSE_COLOR")
        self.create_slider(scrollable_frame, "Path Thickness", "PATH_THICKNESS", 1, 10, 1)
        
        # Indicator settings
        self.create_section_label(scrollable_frame, "Steering Indicators")
        
        self.create_color_picker(scrollable_frame, "Indicator Color", "INDICATOR_COLOR")
        self.create_slider(scrollable_frame, "Max Indicators", "MAX_INDICATORS", 0, 20, 1)
        self.create_slider(scrollable_frame, "Indicator Length", "INDICATOR_LENGTH", 5, 30, 1)
        
        # Grid settings
        self.create_section_label(scrollable_frame, "Grid Display")
        
        self.create_checkbox(scrollable_frame, "Show Grid", "SHOW_GRID")
        self.create_slider(scrollable_frame, "Grid Size", "GRID_SIZE", 10, 200, 10)
        self.create_color_picker(scrollable_frame, "Grid Color", "GRID_COLOR")
        
        # Debug
        self.create_section_label(scrollable_frame, "Debug Options")
        
        self.create_checkbox(scrollable_frame, "Debug Visualization", "DEBUG_VISUALIZATION")
    
    def create_advanced_tab(self):
        advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(advanced_frame, text="Advanced Settings")
        
        # Create scrollable frame
        canvas = tk.Canvas(advanced_frame)
        scrollbar = ttk.Scrollbar(advanced_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Approach parameters
        self.create_section_label(scrollable_frame, "Approach Parameters")
        
        self.create_slider(scrollable_frame, "Approach Path Threshold", "APPROACH_PATH_THRESHOLD", 50, 300, 1)
        self.create_slider(scrollable_frame, "Approach Path Factor", "APPROACH_PATH_FACTOR", 0.1, 0.9, 0.01)
        self.create_slider(scrollable_frame, "Max Approach Distance", "MAX_APPROACH_DISTANCE", 50, 300, 1)
        self.create_slider(scrollable_frame, "Number of Reverse Segments", "NUM_REVERSE_SEGMENTS", 3, 20, 1)
        
        # Perpendicular offset parameters
        self.create_section_label(scrollable_frame, "Perpendicular Offset Parameters")
        
        self.create_slider(scrollable_frame, "Min Perpendicular Offset", "MIN_PERP_OFFSET", 5, 50, 1)
        self.create_slider(scrollable_frame, "Max Perpendicular Offset", "MAX_PERP_OFFSET", 10, 100, 1)
        self.create_slider(scrollable_frame, "Perpendicular Offset Factor", "PERP_OFFSET_FACTOR", 0.05, 0.5, 0.01)
        
        # Midpoint parameters
        self.create_section_label(scrollable_frame, "Midpoint Parameters")
        
        self.create_slider(scrollable_frame, "Min Mid Factor", "MIN_MID_FACTOR", 0.1, 0.6, 0.01)
        self.create_slider(scrollable_frame, "Max Mid Factor", "MAX_MID_FACTOR", 0.5, 0.9, 0.01)
        
        # Curve analysis thresholds
        self.create_section_label(scrollable_frame, "Curve Analysis Thresholds")
        
        self.create_slider(scrollable_frame, "Straight Curve Threshold", "STRAIGHT_CURVE_THRESHOLD", 0.7, 0.99, 0.01)
        self.create_slider(scrollable_frame, "Direct Curve Threshold", "DIRECT_CURVE_THRESHOLD", 0.5, 0.9, 0.01)
        self.create_slider(scrollable_frame, "Small Angle Threshold", "SMALL_ANGLE_THRESHOLD", 10, 60, 1)
        self.create_slider(scrollable_frame, "Moderate Angle Threshold", "MODERATE_ANGLE_THRESHOLD", 30, 90, 1)
        self.create_slider(scrollable_frame, "Very Short Curve Threshold", "VERY_SHORT_CURVE_THRESHOLD", 10, 100, 1)
        self.create_slider(scrollable_frame, "Short Curve Threshold", "SHORT_CURVE_THRESHOLD", 50, 200, 1)
    
    def create_footer(self):
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(footer_frame, text="Save Config", command=self.save_config).pack(side='right', padx=5)
        ttk.Button(footer_frame, text="Reset to Defaults", command=self.reset_to_defaults).pack(side='right', padx=5)
        ttk.Button(footer_frame, text="Apply", command=self.apply_config).pack(side='right', padx=5)
    
    def create_section_label(self, parent, text):
        """Create a section header label"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(frame, text=text, font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=5)
    
    def create_slider(self, parent, label, param_key, min_val, max_val, step):
        """Create a labeled slider with entry box for direct value input"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=20, pady=5)
        
        ttk.Label(frame, text=label, width=25).pack(side='left')
        
        slider_var = tk.DoubleVar(value=self.all_params[param_key])
        
        entry = ttk.Entry(frame, width=10, textvariable=slider_var)
        entry.pack(side='right', padx=5)
        
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=slider_var, 
                          orient='horizontal', length=250)
        slider.pack(side='right', fill='x', expand=True, padx=5)
        
        # Update function for synchronizing slider and entry
        def update_value(*args):
            try:
                # Round to nearest step if needed
                value = float(slider_var.get())
                if step < 1:
                    decimal_places = len(str(step).split('.')[-1])
                    value = round(value / step) * step
                    value = round(value, decimal_places)
                else:
                    value = round(value / step) * step
                
                slider_var.set(value)
                self.all_params[param_key] = value
            except ValueError:
                pass
        
        slider_var.trace("w", update_value)
        
        return slider_var
    
    def create_checkbox(self, parent, label, param_key):
        """Create a labeled checkbox"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=20, pady=5)
        
        check_var = tk.BooleanVar(value=self.all_params[param_key])
        
        cb = ttk.Checkbutton(frame, text=label, variable=check_var)
        cb.pack(side='left')
        
        def update_value(*args):
            self.all_params[param_key] = check_var.get()
        
        check_var.trace("w", update_value)
        
        return check_var
    
    def create_color_picker(self, parent, label, param_key):
        """Create a color picker button"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=20, pady=5)
        
        ttk.Label(frame, text=label, width=25).pack(side='left')
        
        # Convert BGR to hex
        bgr_color = self.all_params[param_key]
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_color)
        
        # Color display rectangle
        color_rect = tk.Canvas(frame, width=30, height=20, bg=hex_color, highlightthickness=1)
        color_rect.pack(side='right', padx=5)
        
        def choose_color():
            # Show color picker
            color_tuple = colorchooser.askcolor(hex_color, title=f"Choose {label}")
            
            if color_tuple[0]:  # If user didn't cancel
                rgb_color = tuple(map(int, color_tuple[0]))
                bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                self.all_params[param_key] = bgr_color
                color_rect.config(bg=color_tuple[1])
        
        ttk.Button(frame, text="Choose Color", command=choose_color).pack(side='right', padx=5)
    
    def reset_to_defaults(self):
        """Reset all parameters to their default values"""
        # Default config
        default_config = {
            'TURNING_RADIUS': 30.0,
            'DISTANCE_SCALING': 4.0,
            'MIN_TURNING_RADIUS': 25.0,
            'MAX_TURNING_RADIUS': 50.0,
            'PATH_POINTS': 80,
            'PATH_COLOR': (0, 255, 0),
            'REVERSE_COLOR': (0, 0, 255),
            'PATH_THICKNESS': 2,
            'INDICATOR_COLOR': (50, 200, 50),
            'MAX_INDICATORS': 5,
            'INDICATOR_LENGTH': 10,
            'DEBUG_VISUALIZATION': False,
            'PATH_SMOOTHING': 0.1,
            'FAVOR_REVERSING': True,
            'SHOW_GRID': False,
            'GRID_SIZE': 50,
            'GRID_COLOR': (100, 100, 100),
        }
        
        # Default additional parameters
        default_additional = {
            'FIXED_START_DISTANCE': 1.0,
            'SHORT_PATH_THRESHOLD': 100,
            'MEDIUM_PATH_THRESHOLD': 200,
            'LONG_PATH_THRESHOLD': 300,
            'SHORT_PATH_FACTOR': 0.15,
            'MEDIUM_PATH_FACTOR': 0.2,
            'LONG_PATH_FACTOR': 0.25,
            'SHORT_PATH_REDUCTION': 6,
            'MEDIUM_PATH_REDUCTION': 4,
            'LONG_PATH_REDUCTION': 2,
            'APPROACH_PATH_THRESHOLD': 180,
            'APPROACH_PATH_FACTOR': 0.4,
            'MAX_APPROACH_DISTANCE': 180,
            'NUM_REVERSE_SEGMENTS': 10,
            'MIN_PERP_OFFSET': 20,
            'MAX_PERP_OFFSET': 40,
            'PERP_OFFSET_FACTOR': 0.15,
            'MIN_MID_FACTOR': 0.4,
            'MAX_MID_FACTOR': 0.7,
            'STRAIGHT_CURVE_THRESHOLD': 0.9,
            'DIRECT_CURVE_THRESHOLD': 0.7,
            'SMALL_ANGLE_THRESHOLD': 30,
            'MODERATE_ANGLE_THRESHOLD': 60,
            'VERY_SHORT_CURVE_THRESHOLD': 50,
            'SHORT_CURVE_THRESHOLD': 100,
            'FIXED_SEGMENT_POINTS': 10,
            'MAX_REDUCTION_PERCENT': 0.8,
        }
        
        # Update the parameters
        self.config = default_config.copy()
        self.additional_params = default_additional.copy()
        self.all_params = {**self.config, **self.additional_params}
        
        # Destroy and recreate the UI
        self.notebook.destroy()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Recreate tabs
        self.create_main_tab()
        self.create_path_tab()
        self.create_visual_tab()
        self.create_advanced_tab()
        
        messagebox.showinfo("Reset", "All parameters have been reset to defaults.")
    
    def save_config(self):
        """Save the current configuration to a JSON file"""
        # Split parameters back
        self.config = {k: self.all_params[k] for k in self.config.keys()}
        self.additional_params = {k: self.all_params[k] for k in self.additional_params.keys()}
        
        # Create config dict with both sets of parameters
        save_config = {
            'CONFIG': self.config,
            'ADDITIONAL_PARAMS': self.additional_params
        }
        
        try:
            # Make sure the config directory exists
            os.makedirs('config', exist_ok=True)
            
            # Save to JSON
            with open('config/reed_shepp_config.json', 'w') as f:
                json.dump(save_config, f, indent=4)
            
            messagebox.showinfo("Success", "Configuration saved to config/reed_shepp_config.json")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def load_config(self):
        """Load configuration from JSON file if available"""
        try:
            if os.path.exists('config/reed_shepp_config.json'):
                with open('config/reed_shepp_config.json', 'r') as f:
                    loaded_config = json.load(f)
                
                # Update config if keys exist
                if 'CONFIG' in loaded_config:
                    self.config.update(loaded_config['CONFIG'])
                
                if 'ADDITIONAL_PARAMS' in loaded_config:
                    self.additional_params.update(loaded_config['ADDITIONAL_PARAMS'])
                
                # Update combined dictionary
                self.all_params = {**self.config, **self.additional_params}
                
                # Recreate UI to reflect loaded values
                self.notebook.destroy()
                self.notebook = ttk.Notebook(self.root)
                self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
                
                # Recreate tabs
                self.create_main_tab()
                self.create_path_tab()
                self.create_visual_tab()
                self.create_advanced_tab()
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def apply_config(self):
        """Apply the current configuration to the Reed-Shepp path generation"""
        # Split parameters back
        self.config = {k: self.all_params[k] for k in self.config.keys()}
        
        # Generate Python code to update CONFIG variable
        code = "# Update CONFIG in reed_shepp_path_assist.py\n\n"
        code += "# Add this to the beginning of the file after importing CONFIG\n\n"
        code += "# Load custom configuration\n"
        code += "try:\n"
        code += "    import json\n"
        code += "    with open('config/reed_shepp_config.json', 'r') as f:\n"
        code += "        loaded_config = json.load(f)\n"
        code += "        \n"
        code += "    # Update CONFIG with loaded values\n"
        code += "    if 'CONFIG' in loaded_config:\n"
        code += "        CONFIG.update(loaded_config['CONFIG'])\n"
        code += "        print(\"Custom configuration loaded successfully.\")\n"
        code += "except Exception as e:\n"
        code += "    print(f\"Error loading custom configuration: {e}\")\n"
        
        # Display the code to add
        code_window = tk.Toplevel(self.root)
        code_window.title("Integration Code")
        code_window.geometry("700x500")
        
        # Add text widget with code
        text_frame = ttk.Frame(code_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap='word')
        text_widget.pack(fill='both', expand=True)
        
        # Add text to widget
        text_widget.insert('1.0', code)
        
        # Add button to copy code
        def copy_to_clipboard():
            self.root.clipboard_clear()
            self.root.clipboard_append(code)
            messagebox.showinfo("Copied", "Code copied to clipboard")
        
        ttk.Button(code_window, text="Copy Code", command=copy_to_clipboard).pack(pady=10)
        
        # Also save config
        self.save_config()

if __name__ == "__main__":
    root = tk.Tk()
    app = ReedSheppConfigUI(root)
    root.mainloop()
