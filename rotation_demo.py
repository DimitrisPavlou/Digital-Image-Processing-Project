import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from rotation import my_img_rotation

filename = "test_image.jpg"
img = Image.open(fp=filename)
gray = img.convert("L")

#gray-scale image
gray_image = np.array(gray).astype(np.uint8)
#rgb-image
rgb_image = np.array(img)

#figure of the gray-scaled and rgb image  
fig1, ax1 = plt.subplots(1, 2)
ax1[0].imshow(gray_image, cmap = "gray")
ax1[1].imshow(rgb_image)

#first rotation 
angle1 = 54*np.pi/180 

rotated1_gray = my_img_rotation(gray_image, angle=angle1)
rotated1_rgb = my_img_rotation(rgb_image , angle = angle1)

fig2, ax2 = plt.subplots(1, 2)
ax2[0].imshow(rotated1_gray, cmap = "gray")
ax2[1].imshow(rotated1_rgb)

#second rotation
angle2 = 213*np.pi/180 

rotated2_gray = my_img_rotation(gray_image, angle = angle2)
rotated2_rgb = my_img_rotation(rgb_image , angle = angle2)

fig3, ax3 = plt.subplots(1, 2)
ax3[0].imshow(rotated2_gray, cmap = "gray")
ax3[1].imshow(rotated2_rgb)


plt.show()