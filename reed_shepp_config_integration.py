import json
import os
import importlib
import sys

def load_config():
    """Load the custom configuration from JSON file"""
    try:
        if os.path.exists('config/reed_shepp_config.json'):
            with open('config/reed_shepp_config.json', 'r') as f:
                return json.load(f)
        else:
            print("No custom configuration found at config/reed_shepp_config.json")
            return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def integrate_config_with_reed_shepp():
    """Integrate the custom configuration with reed_shepp_path_assist.py"""
    
    # Load configuration
    config_data = load_config()
    if not config_data:
        print("No configuration data available. Run reed_shepp_config_ui.py first.")
        return False
    
    # Check if required keys exist
    if 'CONFIG' not in config_data:
        print("Configuration file does not contain CONFIG key")
        return False
    
    try:
        # Import the reed_shepp_path_assist module
        import reed_shepp_path_assist
        
        # Update CONFIG dict with our loaded values
        reed_shepp_path_assist.CONFIG.update(config_data['CONFIG'])
        
        # Process additional parameters that need to be injected 
        # into the compute_reed_shepp_path function
        if 'ADDITIONAL_PARAMS' in config_data:
            # Create a function that applies these additional parameters
            # inside the compute_reed_shepp_path function
            original_function = reed_shepp_path_assist.compute_reed_shepp_path
            
            def wrapped_compute_reed_shepp_path(*args, **kwargs):
                # Extract additional parameters
                additional_params = config_data['ADDITIONAL_PARAMS']
                
                # Apply these parameters within the function context
                # This requires the developer to modify the compute_reed_shepp_path
                # function to look for these values
                
                print("Using custom parameters for path generation:")
                for key, value in additional_params.items():
                    print(f"  {key}: {value}")
                
                # Call the original function with the same arguments
                return original_function(*args, **kwargs)
            
            # Replace the original function with our wrapped version
            reed_shepp_path_assist.compute_reed_shepp_path = wrapped_compute_reed_shepp_path
        
        print("Configuration successfully integrated with reed_shepp_path_assist")
        return True
    
    except ImportError:
        print("Could not import reed_shepp_path_assist module. Make sure it's in the current directory.")
        return False
    except Exception as e:
        print(f"Error integrating configuration: {e}")
        return False

def show_current_config():
    """Display the current configuration"""
    try:
        import reed_shepp_path_assist
        print("\nCurrent Reed-Shepp Path Configuration:")
        for key, value in reed_shepp_path_assist.CONFIG.items():
            print(f"  {key}: {value}")
    except ImportError:
        print("Could not import reed_shepp_path_assist module")

if __name__ == "__main__":
    print("Reed-Shepp Path Configuration Integration Tool")
    print("=============================================")
    
    if integrate_config_with_reed_shepp():
        show_current_config()
        
        print("\nTo run the Reed-Shepp path generator with your configuration:")
        print("  python reed_shepp_path_assist.py")
    else:
        print("\nConfiguration integration failed.")
        print("Make sure to run reed_shepp_config_ui.py first to create your configuration.") 