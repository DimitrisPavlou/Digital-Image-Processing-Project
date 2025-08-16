import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from harris import *
from skimage.feature import canny
import cv2

#read image
filename = "test_image.jpg"
img = Image.open(fp=filename)
#convert to gray scale
bw_image = img.convert("L")
#convert image to a numpy array
gray_image = np.array(bw_image)

# HARRIS CORNER DETECTOR
k = 0.05
sigma = 1.0
size = 5
harris_response = my_corner_harris(gray_image, k, sigma, size)

rel_threshold = 0.5
corner_locations = my_corner_peaks(harris_response, rel_threshold)
marked_img = mark_corners_on_image(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB), corner_locations)

# Display the result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(marked_img, cmap="gray")
plt.title('Harris Corner Detection')
plt.axis("off")
plt.show()

