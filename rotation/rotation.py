import numpy as np


def my_img_rotation(img: np.ndarray, angle: float) -> np.ndarray:
    # Calculate the dimensions of the input image
    h, w = img.shape[:2]
    
    # Calculate the dimensions of the output image
    angle_cos = np.abs(np.cos(angle))
    angle_sin = np.abs(np.sin(angle))
    new_w = int(h * angle_sin + w * angle_cos)
    new_h = int(h * angle_cos + w * angle_sin)

    # Calculate the rotation matrix for clockwise rotation
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    # Calculate the translation matrix to center the original image
    trans_mat1 = np.array([
        [1, 0, -new_w / 2],
        [0, 1, -new_h / 2],
        [0, 0, 1]
    ])

    # Calculate the translation matrix to center the rotated image
    trans_mat2 = np.array([
        [1, 0, w / 2],
        [0, 1, h / 2],
        [0, 0, 1]
    ])

    # Combine the transformations: T2 * R * T1
    transform_mat = trans_mat2 @ rot_mat @ trans_mat1

    # Generate coordinates of the output image
    x, y = np.meshgrid(np.arange(new_w), np.arange(new_h))
    coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()])

    # Apply the transformation
    new_coords = transform_mat @ coords
    new_coords = new_coords[:2, :].round().astype(int)

    # Initialize the output image
    if img.ndim == 2:  # Grayscale image
        rot_img = np.zeros((new_h, new_w), dtype=img.dtype)
    else:  # RGB image
        rot_img = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)

    # Get valid coordinates
    valid_x = (0 <= new_coords[0]) & (new_coords[0] < w)
    valid_y = (0 <= new_coords[1]) & (new_coords[1] < h)
    valid_coords = valid_x & valid_y

    new_x, new_y = new_coords[:, valid_coords]
    x, y = coords[:2, valid_coords]

    # Copy the pixels from the original image to the rotated image
    if img.ndim == 2:
        rot_img[y, x] = img[new_y, new_x]
    else:
        rot_img[y, x] = img[new_y, new_x, :]

    return rot_img


# slower implementation , not used in the demo

#def my_img_rotation_slow(img: np.ndarray, angle: float) -> np.ndarray:
#    # Get the dimensions of the input image
#    h, w = img.shape[:2]
#    
#    # Calculate the center of the image
#    cx, cy = w / 2, h / 2
#    
#    # Calculate the new image dimensions
#    new_w = int(abs(h * np.sin(angle)) + abs(w * np.cos(angle)))
#    new_h = int(abs(h * np.cos(angle)) + abs(w * np.sin(angle)))
#    
#    # Create the output image with a black background
#    if len(img.shape) == 3:
#        rot_img = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
#    else:
#        rot_img = np.zeros((new_h, new_w), dtype=img.dtype)
#    
#    # Calculate the coordinates transformation
#    for y in range(new_h):
#        for x in range(new_w):
#            # Transform coordinates to the original image space
#            x_orig = (x - new_w / 2) * np.cos(angle) + (y - new_h / 2) * np.sin(angle) + cx
#            y_orig = -(x - new_w / 2) * np.sin(angle) + (y - new_h / 2) * np.cos(angle) + cy
#            
#            # Check if the coordinates are within the bounds of the original image
#            if 0 <= x_orig < w and 0 <= y_orig < h:
#                # Bilinear interpolation
#                x1, y1 = int(x_orig), int(y_orig)
#                x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
#                
#                r1 = (x2 - x_orig) * img[y1, x1] + (x_orig - x1) * img[y1, x2]
#                r2 = (x2 - x_orig) * img[y2, x1] + (x_orig - x1) * img[y2, x2]
#                pixel_value = (y2 - y_orig) * r1 + (y_orig - y1) * r2
#                
#                rot_img[y, x] = pixel_value
#    
#    return rot_img
