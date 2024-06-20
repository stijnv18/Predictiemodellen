import cv2
import numpy as np

# Load the image
img = cv2.imread('ressnar2.png')

# Define color range for the first line (blue in this case)
# The color values are reversed because OpenCV uses BGR format
offset = 40 # You can adjust this value

lower_blue = np.array([225-offset, 201-offset, 166-offset])
upper_blue = np.array([225+offset, 201+offset, 166+offset])

# Define color range for the second line (orange in this case)
# The color values are reversed because OpenCV uses BGR format
lower_orange = np.array([151-offset, 199-offset, 255-offset])
upper_orange = np.array([151+offset, 199+offset, 255+offset])

# Create masks for the colors
mask_blue = cv2.inRange(img, lower_blue, upper_blue)
mask_orange = cv2.inRange(img, lower_orange, upper_orange)

# Replace blue color with orange
img[mask_blue != 0] = [151, 199, 255]

# Replace orange color with blue
img[mask_orange != 0] = [225, 201, 166]

# Save the result
cv2.imwrite('Snar2update.png', img)