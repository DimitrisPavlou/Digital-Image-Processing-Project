from scipy.ndimage import convolve
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from filtering.filters import my_wiener_filter, inverse_filter
from filtering.utils import preprocess, create_motion_blur_filter, plot_results, find_best_mse
# ======================================================================================

#==================================================

filename1 = "test_images/cameraman.jpg"
filename2 = "test_images/checkerboard.jpg"

image1 = Image.open(fp = filename1)
image2 = Image.open(fp = filename2)

bw_img1 = image1.convert("L")
bw_img2 = image2.convert("L")

images = (image1, image2)
filter_params = ( (10, 0), (20,30) )
sigmas = (0.02, 0.2)


#==================================================

#pick a combination image-filter_params-noise_power by changing the corresponding indices from 0 to 1 and vice versa
img_index = 0
params_index = 1
sigma_index = 0
# transform the image to a np.ndarray

x = np.array(images[img_index]).astype(np.double)

(m,n) = x.shape
#preprocess x 
x = preprocess(x)

# create white noise with level 0.02
sigma = sigmas[sigma_index]
v = sigma * np.random.randn(*x.shape)

# create motion blur filter
h = create_motion_blur_filter(length=filter_params[params_index][0], angle=filter_params[params_index][1])

# obtain the blured image
y0 = convolve(x, h, mode="wrap")

# generate the noisy image
y = y0 + v

#normalize y
y = preprocess(y)
#uncomment this if you want to find the x_hat that minimizes mse
best_x_hat, best_mse, mses, k_val = find_best_mse(x, y, h)

#plot of the mean square error
plt.figure() 
plt.plot(range(1,500,2), mses) 
plt.grid() 
plt.title("J = E[ (x(n1,n2) - x_hat(n1,n2))^2]") 
plt.xlabel("k values") 
plt.ylabel("MSE")

#uncomment this section if you want to plot the best_x_hat
#plt.figure() 
#plt.imshow(best_x_hat, cmap = "gray") 
#plt.axis("off") 

# perform the filtering 
x_hat = my_wiener_filter(y, h, 100)
x_inv0 = inverse_filter(y0,h)
x_inv = inverse_filter(y,h)

#uncomment this section if you want to plot x_hat with the specific k you used 
#plt.figure() 
#plt.imshow(x_hat, cmap = "gray") 
#plt.axis("off") 

#plot the results the final results
plot_results(x, y0, y, x_inv0, x_inv, x_hat)
plt.show()

