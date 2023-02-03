# Importing the libraries openCV & numpy
import cv2
import numpy as np
  
# Get green color
color = np.uint8([[[215, 0, 0]]])
  
# Convert Green color to Green HSV
hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
  
# Print HSV Value for Green color
print(hsv_color)