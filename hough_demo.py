"""Hough transform line detection demo."""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from hough import my_hough_transform, plot_hough_lines
from skimage.feature import canny

"""Run Hough transform line detection on an image."""
# Load and preprocess image
filename = "test_images/test_image.jpg"
img = Image.open(filename)
bw_image = img.convert("L")

image_array = np.array(img)
gray_image = np.array(bw_image)

# Edge detection using Canny
print("Running edge detection...\n\n")
edges = canny(gray_image, sigma=4)

# Display edges
plt.figure(1)
plt.imshow(edges, cmap="gray")
plt.title("Edge Detection (Canny)")
plt.axis("off")

# Hough transform parameters
drho = 1
dtheta = np.pi / 360
num_peaks = 10

# Perform Hough transform
print("Running Hough Transform...\n\n")
H, peaks, res = my_hough_transform(edges, drho, dtheta, num_peaks)
print(f"Residual metric: {res}")

# Draw detected lines on image
image_with_lines = image_array.copy()
plot_hough_lines(image_with_lines, peaks, drho, dtheta)

# Display Hough accumulator
plt.figure(2)
plt.imshow(H, cmap="gray", aspect="auto")
plt.title("Hough Transform Accumulator")
plt.xlabel("Theta Index")
plt.ylabel("Rho Index")

# Display image with detected lines
plt.figure(3)
plt.imshow(image_with_lines)
plt.title(f"Detected Lines (n={num_peaks})")
plt.axis("off")

plt.show()