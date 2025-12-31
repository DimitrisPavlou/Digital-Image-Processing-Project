import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


# ======================================================================================
def inverse_filter(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    '''
    Function that performs inverse filtering of an image

    Parameters:
    y (np.ndarray) : input image blurred and/or noisy
    h (np.ndarray) : the response of the filter

    Returns:
    x_hat (np.ndarray) : the filtered image
    '''

    # get the shapes of the image y and the lsi system
    (m, n) = y.shape
    (l, p) = h.shape

    # Zero-pad the kernel to the size of the image
    h_padded = np.zeros((m, n))

    # Place the kernel at the center of the zero-padded array
    start_m = (m - l) // 2
    start_n = (n - p) // 2
    h_padded[start_m:start_m + l, start_n:start_n + p] = h

    # REPLACE shift2d with fftshift
    h_padded_shifted = fftshift(h_padded)  # Changed from shift2d(h_padded)

    # Compute the Fourier transform of the image and the padded kernel
    Yf = fft2(y)
    Hf = fft2(h_padded_shifted)

    # Avoid division by zero by adding a small constant to Hf
    epsilon = 1e-10
    # find where |Hf| is less than epsilon
    indices = np.abs(Hf) < epsilon
    # add epsilon to avoid division by 0
    Bf = 1 / (Hf + epsilon)
    # set Bf = epsilon at the indices we found above
    Bf[indices] = epsilon
    # Perform the inverse filtering
    X_hat = Yf * Bf
    # return to the spatial domain
    x_hat = np.real(ifft2(X_hat))

    return x_hat

