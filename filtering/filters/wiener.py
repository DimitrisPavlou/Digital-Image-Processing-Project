import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def my_wiener_filter(y: np.ndarray, h: np.ndarray, K: float) -> np.ndarray:
    '''
    Function that performs wiener filtering

    INPUT:
    y (np.ndarray) : input image blurred and/or noisy
    h (np.ndarray) : the response of the filter
    K (float) : represents the signal to noise ratio. It is a tunable hyperparameter

    OUTPUT:
    x_hat (np.ndarray) : the filtered image
    '''

    (m, n) = y.shape
    (l, p) = h.shape

    # Zero-pad the kernel to the size of the image
    h_padded = np.zeros((m, n))

    # Center the kernel in the zero-padded array
    start_m = (m - l) // 2
    start_n = (n - p) // 2
    h_padded[start_m:start_m + l, start_n:start_n + p] = h

    # REPLACE shift2d with fftshift
    h_padded_shifted = fftshift(h_padded)  # Changed from shift2d(h_padded)

    # Compute the Fourier transform of the image and the padded and shifted kernel
    Yf = fft2(y)
    Hf = fft2(h_padded_shifted)

    # Wiener filter
    Gf = np.conj(Hf) / (np.abs(Hf) ** 2 + 1 / K)
    X_hat = Gf * Yf
    # return to the spatial domain
    x_hat = np.real(ifft2(X_hat))

    return x_hat