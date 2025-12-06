import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from harris import my_corner_harris, my_corner_peaks, mark_corners_on_image
import cv2


"""Run Harris corner detection on an image."""
# Load and preprocess image
filename = "test_image.jpg"
img = Image.open(filename)
bw_image = img.convert("L")
gray_image = np.array(bw_image)

# Harris corner detection parameters
k = 0.05  # Harris detector free parameter
sigma = 1.0  # Gaussian window standard deviation
size = 5  # Gaussian window size
rel_threshold = 0.5  # Relative threshold for peak detection

# Compute Harris response
harris_response = my_corner_harris(gray_image, k, sigma, size)

# Detect corner peaks
corner_locations = my_corner_peaks(harris_response, rel_threshold)

# Mark corners on RGB image
rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
marked_img = mark_corners_on_image(rgb_image, corner_locations)

# Display results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(marked_img)
plt.title(f'Harris Corners Detected (n={len(corner_locations)})')
plt.axis("off")

plt.tight_layout()
plt.show()

