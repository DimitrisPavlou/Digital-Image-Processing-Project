import numpy as np
import cv2
from typing import Tuple


def my_hough_transform(
    edge_image: np.ndarray,
    drho: float,
    dtheta: float,
    num_peaks: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Hough transform on an edge-detected image to find lines.
    
    Args:
        edge_image: Binary edge image (2D numpy array)
        drho: Resolution of rho in pixels
        dtheta: Resolution of theta in radians
        num_peaks: Number of peaks to detect in the accumulator
        
    Returns:
        Tuple containing:
            - H: Hough accumulator array
            - peak_indices: Array of (rho_idx, theta_idx) for detected peaks
            - res: Residual value (diagnostic metric)
    """
    rows, cols = edge_image.shape
    max_rho = int(np.hypot(rows, cols))
    
    # Define parameter space
    rhos = np.arange(-max_rho, max_rho, drho)
    thetas = np.arange(0, np.pi, dtheta)
    
    # Initialize accumulator
    H = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    
    # Get edge point coordinates
    y_idxs, x_idxs = np.nonzero(edge_image)
    
    # Precompute trigonometric values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    # Vote in accumulator space
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        rho_vals = x * cos_t + y * sin_t
        rho_idxs = np.round((rho_vals + max_rho) / drho).astype(int)
        
        # Ensure indices are within bounds
        valid_mask = (rho_idxs >= 0) & (rho_idxs < len(rhos))
        H[rho_idxs[valid_mask], np.arange(len(thetas))[valid_mask]] += 1
    
    # Find top peaks
    flat_indices = np.argpartition(H.ravel(), -num_peaks)[-num_peaks:]
    peak_indices = np.column_stack(np.unravel_index(flat_indices, H.shape))
    
    # Sort peaks by accumulator value (descending)
    peak_indices = peak_indices[
        np.argsort(H[peak_indices[:, 0], peak_indices[:, 1]])[::-1]
    ]
    
    res = rows * cols - H.size
    return H, peak_indices, res


def plot_hough_lines(
    image: np.ndarray,
    peaks: np.ndarray,
    drho: float,
    dtheta: float
) -> None:
    """
    Draw detected lines on an image.
    
    Args:
        image: Input image (modified in-place)
        peaks: Array of (rho_idx, theta_idx) peak locations
        drho: Resolution of rho in pixels
        dtheta: Resolution of theta in radians
    """
    rows, cols = image.shape[:2]
    max_rho = int(np.hypot(rows, cols))
    
    rhos = np.arange(-max_rho, max_rho, drho)
    thetas = np.arange(0, np.pi, dtheta)
    
    for rho_idx, theta_idx in peaks:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # Calculate line endpoints (extended far beyond image bounds)
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * a)
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * a)
        
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 10)
