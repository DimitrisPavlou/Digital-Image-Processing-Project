import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple


def my_hough_transform(edge_image: np.ndarray, drho: float, dtheta: float, num_peaks: int)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Image dimensions
    rows, cols = edge_image.shape

    # Rho and Theta ranges
    max_rho = int(np.hypot(rows, cols))
    rhos = np.arange(-max_rho, max_rho, drho)
    thetas = np.arange(0, np.pi, dtheta)

    # Hough accumulator array
    H = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    # Edge points (non-zero points in the edge image)
    y_idxs, x_idxs = np.nonzero(edge_image)

    # Precompute cos and sin values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Vectorized voting in the accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        rho_vals = x * cos_t + y * sin_t
        rho_idxs = np.round((rho_vals + max_rho) / drho).astype(int)
        H[rho_idxs, np.arange(len(thetas))] += 1

    # Flatten the accumulator array and get the indices of the top num_peaks values
    flat_indices = np.argpartition(H.ravel(), -num_peaks)[-num_peaks:]
    peak_indices = np.column_stack(np.unravel_index(flat_indices, H.shape))
    
    # Sort the peaks based on accumulator values
    peak_indices = peak_indices[np.argsort(H[peak_indices[:, 0], peak_indices[:, 1]])[::-1]]
    
    res = rows*cols - len(H)
    return H, peak_indices, res


def plot_hough_lines(image: np.ndarray, peaks: np.ndarray, drho: int , dtheta: int):
    
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
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 10)


