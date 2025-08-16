import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from hough import *
from skimage.feature import canny

#read image
filename = "test_image.jpg"
img = Image.open(fp=filename)
#convert to gray scale
bw_image = img.convert("L")
#convert image to a a numpy array of 8bit integers

image_array = np.array(img)#.astype(np.uint8)
gray_image = np.array(bw_image)


#edge detection 
edges = canny(gray_image, sigma = 4)
plt.figure(1)
plt.imshow(edges, cmap = "gray")
plt.axis("off")

#hough transform 
#accumulator,rhos,thetas = my_hough_transform(edges, drho = 1, dtheta = np.pi/3600)
#find peak locations

#peaks = find_peaks(accumulator, num_peaks = 150)
#draw the lines onto the image
H, peaks, res = my_hough_transform(edges, drho = 1 , dtheta = np.pi/360, num_peaks = 150)
print(f"Res = {res}")
plot_hough_lines(image_array, peaks, drho = 1 , dtheta = np.pi/360)
plt.figure(2)
plt.imshow(H, cmap="gray", aspect = "auto")
plt.title("Hough Transform Matrix H")
#plot the image with the marked lines
plt.figure(3)
plt.imshow(image_array)
plt.title('Detected Lines')
plt.axis("off")
plt.show()