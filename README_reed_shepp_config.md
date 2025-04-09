# Reed-Shepp Path Configuration Tool

This set of tools allows you to customize all parameters used in the Reed-Shepp path generation for parking path assistance.

## Files Overview

- `reed_shepp_config_ui.py`: Tkinter-based UI for customizing and saving path generation parameters
- `reed_shepp_config_integration.py`: Helper script to integrate saved configurations with the main program
- `reed_shepp_path_assist.py`: Main path generation program that can be configured

## Getting Started

### 1. Configure Parameters

Run the configuration UI to customize all path generation parameters:

```
python reed_shepp_config_ui.py
```

The UI is organized into tabs:
- **Main Settings**: Core turning radius parameters
- **Path Generation**: Parameters that control path points and distribution 
- **Visual Settings**: Colors, line thickness, grid display options
- **Advanced Settings**: Fine-tuning parameters for the path generation algorithm

After adjusting parameters, click "Save Config" to store them for later use.

### 2. Integration with Path Assistant

#### Option 1: Use the Integration Script

Run the integration script to load your configuration into the Reed-Shepp path assistant:

```
python reed_shepp_config_integration.py
```

Then run the path assistant as usual:

```
python reed_shepp_path_assist.py
```

#### Option 2: Manual Integration

Alternatively, you can manually integrate the configuration by adding these lines to `reed_shepp_path_assist.py` after the CONFIG declaration:

```python
# Load custom configuration
try:
    import json
    with open('config/reed_shepp_config.json', 'r') as f:
        loaded_config = json.load(f)
        
    # Update CONFIG with loaded values
    if 'CONFIG' in loaded_config:
        CONFIG.update(loaded_config['CONFIG'])
        print("Custom configuration loaded successfully.")
except Exception as e:
    print(f"Error loading custom configuration: {e}")
```

## Key Parameters

### Turning Radius Parameters
- `TURNING_RADIUS`: Default turning radius value
- `DISTANCE_SCALING`: How turning radius relates to distance between points
- `MIN_TURNING_RADIUS`: Minimum allowed turning radius for vehicle constraints
- `MAX_TURNING_RADIUS`: Maximum allowed turning radius

### Path Generation
- `PATH_POINTS`: Total number of points generated along the path
- `PATH_SMOOTHING`: Smoothing factor for the path (0-1)
- Various thresholds for short, medium, and long paths
- Point reduction parameters for different path characteristics

### Visual Settings
- Path colors for forward and reverse motion
- Line thickness and steering indicator settings
- Grid display options

### Advanced Parameters
- Approach distance and factors
- Perpendicular offset calculations 
- Curve analysis thresholds
- Midpoint calculation factors

## Customization Tips

1. **For Smoother Paths**: Increase `PATH_POINTS` and decrease `PATH_SMOOTHING`
2. **For Tighter Turns**: Decrease `MIN_TURNING_RADIUS` and `TURNING_RADIUS`
3. **For Better Reversed Parking**: Increase `FAVOR_REVERSING` and adjust `APPROACH_PATH_FACTOR`
4. **For Clearer Visualization**: Adjust path colors and thickness, and consider enabling the grid

## Applying Multiple Configurations

You can save multiple configuration files by renaming them in the `config` directory. To apply a specific configuration, 
update the file path in the integration code to point to your desired configuration file. 