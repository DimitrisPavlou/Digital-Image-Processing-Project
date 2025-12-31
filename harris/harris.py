"""Harris corner detection implementation."""

import cv2
import numpy as np
from scipy.signal import convolve2d
from typing import Tuple


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.
    
    Args:
        size: Kernel size (should be odd)
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Normalized 2D Gaussian kernel
    """
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    X, Y = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(X) + np.square(Y)) / np.square(sigma))
    return kernel / np.sum(kernel)


def gaussian_filter(img: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian filter to an image.
    
    Args:
        img: Input image
        size: Kernel size
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Filtered image
    """
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(img, kernel, mode='same', boundary='symm')


def sobel_filter(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients using Sobel filters.
    
    Args:
        img: Input grayscale image
        
    Returns:
        Tuple of (Ix, Iy) gradient images in x and y directions
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    
    Ix = convolve2d(img, sobel_x, mode='same', boundary='symm')
    Iy = convolve2d(img, sobel_y, mode='same', boundary='symm')
    
    return Ix, Iy


def my_corner_harris(
    img: np.ndarray,
    k: float,
    sigma: float,
    size: int
) -> np.ndarray:
    """
    Compute Harris corner response for an image.
    
    Args:
        img: Input grayscale image
        k: Harris detector free parameter (typically 0.04-0.06)
        sigma: Gaussian window standard deviation
        size: Gaussian window size
        
    Returns:
        Harris response matrix (same shape as input image)
    """
    img = np.float32(img)
    
    # Compute image gradients
    Ix, Iy = sobel_filter(img)
    
    # Compute gradient products
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    
    # Apply Gaussian smoothing to gradient products
    Sxx = gaussian_filter(Ixx, size=size, sigma=sigma)
    Sxy = gaussian_filter(Ixy, size=size, sigma=sigma)
    Syy = gaussian_filter(Iyy, size=size, sigma=sigma)
    
    # Compute Harris response
    detM = (Sxx * Syy) - (Sxy * Sxy)
    traceM = Sxx + Syy
    R = detM - k * (traceM ** 2)
    
    return R


def my_corner_peaks(
    harris_response: np.ndarray,
    rel_threshold: float
) -> np.ndarray:
    """
    Detect corner peaks in Harris response using non-maximum suppression.
    
    Args:
        harris_response: Harris response matrix
        rel_threshold: Relative threshold (0-1) for peak detection
        
    Returns:
        Array of corner coordinates (N x 2), where each row is [y, x]
    """
    # Normalize response to [0, 1]
    max_response = np.max(harris_response)
    if max_response == 0:
        return np.array([])
    
    normalized_response = harris_response / max_response
    
    # Apply threshold
    corner_mask = normalized_response > rel_threshold
    
    # Non-maximum suppression
    dilated_response = cv2.dilate(normalized_response, None)
    corner_peaks = (normalized_response == dilated_response) & corner_mask
    
    # Extract coordinates
    corners = np.argwhere(corner_peaks)
    
    return corners


def mark_corners_on_image(
    image: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
    thickness: int = -1
) -> np.ndarray:
    """
    Draw circles at corner locations on an image.
    
    Args:
        image: Input RGB image
        corners: Array of corner coordinates (N x 2), where each row is [y, x]
        color: Circle color in BGR format (default: red)
        radius: Circle radius in pixels
        thickness: Circle thickness (-1 for filled)
        
    Returns:
        Image with marked corners
    """
    output_img = np.copy(image)
    
    for y, x in corners:
        cv2.circle(output_img, (x, y), radius, color, thickness)
    
    return output_img
