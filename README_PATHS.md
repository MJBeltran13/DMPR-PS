# Smart Parking Path Assistant

This system helps generate optimal parking paths using machine learning.

## Overview

The Smart Parking Path Assistant is a tool for:
1. Simulating car positioning and paths
2. Collecting training data for path generation
3. Training a machine learning model to predict optimal paths
4. Using the trained model to automatically generate realistic parking paths

## Quick Start

1. Run the main application:
   ```
   python smart_path_assist.py
   ```

2. Generate and collect path data:
   - Use mouse to position the green (current position) and red (target position) boxes
   - Use 'z'/'x' keys to rotate the green box
   - Use 'c'/'v' keys to rotate the red box
   - Press 'p' to generate a parking path
   - Press 'a' to approve a good path or 'r' to reject a bad path
   - Repeat to collect multiple examples

3. Train the model:
   - After collecting at least 5-10 good path examples, press 'e' to train the model
   - The training will run and show results in the console

4. Use the trained model:
   - Once trained, the system will automatically use the ML model for path generation
   - Press 't' to toggle between ML and manual path generation
   - Continue approving/rejecting paths to improve the model over time

## Controls

- **Mouse**: Drag boxes to position them
- **z/x**: Rotate green box (current position) counter-clockwise/clockwise
- **c/v**: Rotate red box (target position) counter-clockwise/clockwise
- **p**: Generate parking path
- **t**: Toggle between ML and manual path generation
- **m**: Toggle control point editing mode (manual mode only)
- **a**: Approve path as correct (saves to training data)
- **r**: Reject path as incorrect (saves to training data)
- **e**: Train model on collected data
- **Space**: Save box parameters to CSV
- **q**: Quit

## Training the Model Separately

You can also train the model separately without the UI: 