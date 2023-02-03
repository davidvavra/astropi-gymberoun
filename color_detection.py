# Importing the libraries OpenCV and numpy
import cv2
import numpy as np

#colors
ocean = [215, 0, 0]
river = [215, 255, 0]
clouds = [215, 181, 227]
forest = [0, 89, 7]
field = [0, 255, 7]
desert = [0, 255, 255]
unknownland = [0, 106, 255]
unknown = [220, 106, 90]
island = [129, 106, 255]
mountains = [129, 0, 67]

colors =[ocean, river, clouds, forest, field, desert, unknownland, unknown, island, mountains]

# Read the images
img = cv2.imread("bitmapa.png")
  
# Convert Image to Image HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Get color
color = np.uint8([[ocean]])
  
# Convert color to HSV
hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

# Defining lower and upper bound HSV values
h = hsv_color[0][0][0]
lower = np.array([h-10, 50, 50])
upper = np.array([h+10, 255, 255])

# Defining mask for detecting color
mask = cv2.inRange(hsv, lower, upper)

# Get averag color
array = np.array(mask)
average_color = np.mean(array)
print(average_color)

# Display Image and Mask
cv2.imshow("Image", img)
cv2.imshow("Mask", mask)
  
# Make python sleep for unlimited time
cv2.waitKey(0)

# for i in colors:
#     color = np.np.uint8([[i]])
#     hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
#     h = hsv_color[0][0][0]
#     lower = np.array([h-10, 50, 50])
#     upper = np.array([h+10, 255, 255])
#     mask = cv2.inRange(hsv, lower, upper)
