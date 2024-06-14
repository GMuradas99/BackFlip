import cv2
import random
import numpy as np

def hor_flip(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Perform horizontal flip on the input image and mask.

    Args:
        img (np.ndarray): The input image.
        mask (np.ndarray): The input mask.

    Returns:
        dict: A dictionary containing the flipped image and mask.
    """
    return {
        'img': cv2.flip(img, 1),
        'mask': cv2.flip(mask, 1)
    }

def ver_flip(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Perform vertical flip on the input image and mask.

    Args:
        img (np.ndarray): The input image.
        mask (np.ndarray): The input mask.

    Returns:
        dict: A dictionary containing the flipped image and mask.
    """
    return {
        'img': cv2.flip(img, 0),
        'mask': cv2.flip(mask, 0)
    }

def rotation(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Rotate the input image and mask by a random angle.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        mask (np.ndarray): The input mask as a NumPy array.

    Returns:
        dict: A dictionary containing the rotated image and mask.
            - 'img' (np.ndarray): The rotated image as a NumPy array.
            - 'mask' (np.ndarray): The rotated mask as a NumPy array.
    """
    
    # Adding padding for rotation
    diagonal = int(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2))
    padded_img = np.zeros((diagonal, diagonal, 3), dtype=np.uint8)
    padded_img[(diagonal - img.shape[0]) // 2:(diagonal - img.shape[0]) // 2 + img.shape[0], (diagonal - img.shape[1]) // 2:(diagonal - img.shape[1]) // 2 + img.shape[1]] = img

    padded_mask = np.zeros((diagonal, diagonal, 3), dtype=np.uint8)
    padded_mask[(diagonal - mask.shape[0]) // 2:(diagonal - mask.shape[0]) // 2 + mask.shape[0], (diagonal - mask.shape[1]) // 2:(diagonal - mask.shape[1]) // 2 + mask.shape[1]] = mask
    
    # Random rotation angle
    angle = random.randint(-45, 45)
    rotation_matrix = cv2.getRotationMatrix2D((diagonal // 2, diagonal // 2), angle, 1)

    # Rotate the image
    rotated_img = cv2.warpAffine(padded_img, rotation_matrix, (diagonal, diagonal))
    rotated_mask = cv2.warpAffine(padded_mask, rotation_matrix, (diagonal, diagonal))

    return {
        'img': rotated_img,
        'mask': rotated_mask
    }

def brightness_jitter(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Adjusts the brightness of the input image and mask by a random factor.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        mask (np.ndarray): The input mask as a NumPy array.

    Returns:
        dict: A dictionary containing the image and mask with adjusted brightness.
            - 'img' (np.ndarray): The image with adjusted brightness as a NumPy array.
            - 'mask' (np.ndarray): The mask with adjusted brightness as a NumPy array.
    """
    # Random brightness factor
    brightness_factor = random.uniform(0.5, 1.5)

    # Adjust the brightness of the image
    adjusted_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    adjusted_img = np.clip(adjusted_img, 0, 255)

    return {
        'img': adjusted_img,
        'mask': mask
    }

# TRANSLATION (todo)

def downscale(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Downscale the input image and mask by a random factor.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        mask (np.ndarray): The input mask as a NumPy array.

    Returns:
        dict: A dictionary containing the downscaled image and mask.
            - 'img' (np.ndarray): The downscaled image as a NumPy array.
            - 'mask' (np.ndarray): The downscaled mask as a NumPy array.
    """
    # Random scaling factor
    scaling_factor = random.uniform(0.5, 0.8)

    # Downscale the image
    downscaled_img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    downscaled_mask = cv2.resize(mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

    return {
        'img': downscaled_img,
        'mask': downscaled_mask
    }

def upscale(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Upscale the input image and mask by a random factor.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        mask (np.ndarray): The input mask as a NumPy array.

    Returns:
        dict: A dictionary containing the upscaled image and mask.
            - 'img' (np.ndarray): The upscaled image as a NumPy array.
            - 'mask' (np.ndarray): The upscaled mask as a NumPy array.
    """
    # Random scaling factor
    scaling_factor = random.uniform(1.1, 1.4)

    # Upscale the image
    upscaled_img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    upscaled_mask = cv2.resize(mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

    return {
        'img': upscaled_img,
        'mask': upscaled_mask
    }

def removal(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Removes the object from the input image using the provided mask.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        mask (np.ndarray): The input mask as a NumPy array.

    Returns:
        dict: A dictionary containing the empty image
    """

    return {
        'img': img,
        'mask': np.zeros_like(mask)
    }