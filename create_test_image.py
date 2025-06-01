import numpy as np
import cv2
import os

# Create a directory for the test image if it doesn't exist
os.makedirs('test_images', exist_ok=True)

# Create a blank image (800x600 white background)
image = np.ones((600, 800, 3), dtype=np.uint8) * 255

# Add some shapes to make it more interesting
# Add a rectangle
cv2.rectangle(image, (100, 100), (300, 200), (0, 0, 255), -1)
# Add a circle
cv2.circle(image, (500, 300), 80, (0, 255, 0), -1)
# Add some text
cv2.putText(image, "Test Image for Drag and Drop", (200, 400), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Save the image
cv2.imwrite('test_images/test_image.jpg', image)
print("Test image created at test_images/test_image.jpg") 