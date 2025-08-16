import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    X, Y = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(X) + np.square(Y)) / np.square(sigma))
    return kernel / np.sum(kernel)

def gaussian_filter(img: np.ndarray, size: int, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(img, kernel, mode='same', boundary='symm')


def sobel_filter(img: np.ndarray) :
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # Apply the Sobel kernels to the image using convolve2d
    Ix = convolve2d(img, sobel_x, mode='same', boundary='symm')
    Iy = convolve2d(img, sobel_y, mode='same', boundary='symm')

    return Ix, Iy

def my_corner_harris(img: np.ndarray, k: float, sigma: float, size: int) -> np.ndarray:
    # Ensure the image is in float format
    img = np.float32(img)
    #compute gradients in each direction
    Ix, Iy = sobel_filter(img)

    # Compute products of gradients
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
 
    # Apply Gaussian filter to smooth the products of derivatives
    Sxx = gaussian_filter(Ixx, size = size, sigma=sigma)
    Sxy = gaussian_filter(Ixy, size = size, sigma=sigma)
    Syy = gaussian_filter(Iyy, size = size, sigma=sigma)
    
    # Compute the Harris response
    detM = (Sxx * Syy) - (Sxy * Sxy)
    traceM = Sxx + Syy
    R = detM - k * (traceM ** 2)
    
    return R


def my_corner_peaks(harris_response: np.ndarray, rel_threshold: float) -> np.ndarray:
    # Normalize the Harris response
    harris_response = harris_response / np.max(harris_response)
    
    # Apply the relative threshold
    corner_mask = harris_response > rel_threshold

    # Perform non-maximum suppression
    dilated_response = cv2.dilate(harris_response, None)
    corner_peaks = (harris_response == dilated_response) & corner_mask
    
    # Get the coordinates of the corners
    corners = np.argwhere(corner_peaks)
    
    return corners


def mark_corners_on_image(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    output_img = np.copy(image)
    for y, x in corners:
        cv2.circle(output_img, (x, y), 3, (255, 0, 0), 50)  # Red dot with radius 3 and thickness 50
    return output_img
