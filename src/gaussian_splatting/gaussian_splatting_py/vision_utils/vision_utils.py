import numpy as np
import cv2

from torchvision import transforms

def convert_intrinsics(img, old_intrinsics = (360.01, 360.01, 243.87, 137.92), new_intrinsics = (1297.67, 1298.63, 620.91, 238.28), new_size=(1280, 720)):
    """
    Convert a set of images to a different set of camera intrinsics.
    Parameters:
    - images: List of input images.
    - old_intrinsics: Tuple (fx, fy, cx, cy) of the old camera intrinsics.
    - new_intrinsics: Tuple (fx, fy, cx, cy) of the new camera intrinsics.
    - new_size: Tuple (width, height) defining the size of the output images.
    Returns:
    - List of images converted to the new camera intrinsics.
    """
    old_fx, old_fy, old_cx, old_cy = old_intrinsics
    new_fx, new_fy, new_cx, new_cy = new_intrinsics
    width, height = new_size
    
    # Constructing the old and new intrinsics matrices
    K_old = np.array([[old_fx, 0, old_cx], [0, old_fy, old_cy], [0, 0, 1]])
    K_new = np.array([[new_fx, 0, new_cx], [0, new_fy, new_cy], [0, 0, 1]])
    # Compute the inverse of the new intrinsics matrix for remapping
    K_new_inv = np.linalg.inv(K_new)
    
    # Construct a grid of points representing the new image coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    homogenous_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=-1).T
    
    # Convert to the old image coordinates
    old_coords = K_old @ K_new_inv @ homogenous_coords
    old_coords /= old_coords[2, :]  # Normalize to make homogeneous
    
    # Reshape for remapping
    map_x = old_coords[0, :].reshape(height, width).astype(np.float32)
    map_y = old_coords[1, :].reshape(height, width).astype(np.float32)
    
    # Remap the image to the new intrinsics
    converted_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return converted_img


def warp_image(image, K, R, t):
    """
    Warp an image from the perspective of camera 1 to camera 2.

    :param image: Input image from camera 1
    :param K: Intrinsic matrix of both cameras
    :param R: Rotation matrix from camera 1 to camera 2
    :param t: Translation vector from camera 1 to camera 2
    :return: Warped image as seen from camera 2
    """
    # Compute the homography matrix
    H = compute_homography(K, R, t)

    # Warp the image using the homography
    height, width = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (width, height))

    return warped_image


def compute_homography(K, R, t):
    """
    Compute the homography matrix given intrinsic matrix K, rotation matrix R, and translation vector t.
    """
    K_inv = np.linalg.inv(K)
    H = np.dot(K, np.dot(R - np.dot(t.reshape(-1, 1), K_inv[-1, :].reshape(1, -1)), K_inv))
    return H


def resize_if_too_large(image):
    if image.shape[0] > 1600 and image.shape[1] > 1600:
        image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
    return image


def preprocess_image(image):
    # Transform to convert image to tensor and normalize
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

def learn_scale_and_offset_raw(dense_depth, sparse_depth):
    dense_depth_flat = dense_depth.flatten()
    sparse_depth_flat = sparse_depth.flatten()

    valid_mask = sparse_depth_flat > 0
    dense_depth_valid = dense_depth_flat[valid_mask]
    sparse_depth_valid = sparse_depth_flat[valid_mask]

    A = np.vstack([dense_depth_valid, np.ones_like(dense_depth_valid)]).T
    b = sparse_depth_valid

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    scale, offset = x
    return scale, offset