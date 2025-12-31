import numpy as np
import matplotlib.pyplot as plt
import cv2
from ..filters import my_wiener_filter
#======================================================================================

def create_motion_blur_filter(length, angle):
    """
    Generates a 2D kernel, which, when convolved with an image, simulates the motion blur effect.

    :param length: (int) the length of the camera motion, in pixels
    :param angle: (float) the direction of the camera motion, in degrees (measured from the upper left corner)

    :return: a 2D np array
    """

    theta_rad = np.deg2rad(angle)

    x, y = length * np.cos(theta_rad), length * np.sin(theta_rad)

    size_x = max(1, round(x))
    size_y = max(1, round(y))

    img = np.zeros(shape=(size_x, size_y))

    cv2.line(img, pt1=(0, 0), pt2=(size_y - 1, size_x - 1), color=1.0)
    img = img / np.sum(img)

    return img

def preprocess(x: np.ndarray) -> np.ndarray: 
    """
    Function that normalizes an image into the interval [0,1]

    Parameters : 
    x (np.ndarray) : the 2d matrix of an image
    
    Returns : 
    normalized_x (np.ndarray) : the image normalized in [0,1] 
    """ 

    #find the minimum of the image
    minn = np.min(x)
    #find the maximum of the image
    maxx = np.max(x) 
    #apply the normalization transform
    normalized_x = (x - minn)/(maxx - minn)
    return normalized_x


#======================================================================================

def plot_results(x: np.ndarray,
                 y0: np.ndarray,
                 y: np.ndarray,
                 x_inv0: np.ndarray,
                 x_inv: np.ndarray,
                 x_hat: np.ndarray):
    """
    Function that plots the results

    Parameters :
    x (np.ndarray) :  original image
    y0 (np.ndarray) : blurred image
    y (np.ndarray) : blurred and noisy image
    x_inv0 (np.ndarray) : the result of inverse filtering on y0 
    x_inv (np.ndarray) : the result of inverse filtering on y 
    x_hat (np.ndarray) : the result of wiener filtering on y


    note: the functions requires a plt.show() command on the main script to work
    """
    
    
    fig, axs = plt.subplots(nrows=2, ncols=3)
    axs[0][0].imshow(x, cmap='gray')
    axs[0][0].set_title("Original image x")
    axs[0][1].imshow(y0, cmap='gray')
    axs[0][1].set_title("Clean image y0")
    axs[0][2].imshow(y, cmap='gray')
    axs[0][2].set_title("Blurred and noisy image y")
    axs[1][0].imshow(x_inv0, cmap='gray')
    axs[1][0].set_title("Inverse filtering noiseless output x_inv0")
    axs[1][1].imshow(x_inv, cmap='gray')
    axs[1][1].set_title("Inverse filtering noisy output x_inv")
    axs[1][2].imshow(x_hat, cmap='gray')
    axs[1][2].set_title("Wiener filtering output x_hat")


#======================================================================================


def find_best_mse(x_original: np.ndarray,
                  y: np.ndarray,
                  h: np.ndarray,
                  range_k: tuple = (1, 500),
                  step: int = 2) -> tuple[np.ndarray, float, list, float]:
    """
    function to find the best reconstruction based on the minimum mse

    Parameters :
    x_original (np.ndarray) : the original image
    y (np.ndarray) : noisy image
    range_k (tuple) : a tuple with starting and ending values for the hypermeter K of the filtering filter
    step (int) : step size for the range_k

    Returns :
    best_x_hat (np.ndarray): the reconstructed image with the minimum mse(x_original - x_hat)
    best_mse_val (float) : the minimum mse
    mse_vals (list) : the list of computed mse values for plotting purposes
    best_k_val (float) : the K value that achieves the minimum MSE
    """

    # get the shape of the image
    (m, n) = x_original.shape
    # initialize lists
    reconstructed = []
    mse_vals = []
    k_vals = []

    # for each k
    for k in range(range_k[0], range_k[1], step):
        # find the reconstruction with the specific k
        x_hat = my_wiener_filter(y, h, k)
        reconstructed.append(x_hat)
        # find the mse
        mse = 1 / (m * n) * (np.sum((x_original - x_hat) ** 2))
        mse_vals.append(mse)

        k_vals.append(k)

    # best index
    idx = np.argmin(mse_vals)
    # find the best reconstruction, k value and the minimum mse
    best_x_hat = reconstructed[idx]
    best_k_val = k_vals[idx]
    best_mse_val = mse_vals[idx]

    return best_x_hat, best_mse_val, mse_vals, best_k_val
