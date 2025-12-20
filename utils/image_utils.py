"""
Utility Functions for Image Processing and Conversion

Provides helper functions for:
- Image format conversion
- Mask overlay visualization
- Base64 encoding for API responses
"""
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO


def numpy_to_pil(image):
    """
    Convert numpy array to PIL Image.
    
    Args:
        image (np.ndarray): Numpy array (H, W) or (H, W, C)
    
    Returns:
        PIL.Image: PIL Image object
    """
    if len(image.shape) == 2:
        # Grayscale
        return Image.fromarray(image)
    else:
        # RGB
        return Image.fromarray(image.astype(np.uint8))


def pil_to_numpy(image):
    """
    Convert PIL Image to numpy array.
    
    Args:
        image (PIL.Image): PIL Image object
    
    Returns:
        np.ndarray: Numpy array
    """
    return np.array(image)


def create_overlay(original_image, mask, color=(0, 255, 0), thickness=2):
    """
    Create visualization with boundary contour line on original image.
    
    Args:
        original_image (np.ndarray): Original grayscale or RGB image
        mask (np.ndarray): Binary mask (0 or 255)
        color (tuple): RGB color for boundary line (default: green)
        thickness (int): Thickness of boundary line in pixels (default: 2)
    
    Returns:
        np.ndarray: RGB image with boundary contour line
    """
    # Ensure original image is RGB
    if len(original_image.shape) == 2:
        # Grayscale to RGB
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 1:
        # Single channel to RGB
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_image.copy()
    
    # Ensure mask is binary (0 or 1)
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Find contours of the segmented region
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw only the boundary contour line (no filled overlay)
    result = original_rgb.copy()
    cv2.drawContours(result, contours, -1, color, thickness)
    
    return result


def image_to_base64(image):
    """
    Convert image (numpy or PIL) to base64 string.
    
    Args:
        image (np.ndarray or PIL.Image): Image to encode
    
    Returns:
        str: Base64 encoded string
    """
    # Convert numpy to PIL if needed
    if isinstance(image, np.ndarray):
        image = numpy_to_pil(image)
    
    # Save to BytesIO buffer
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return img_base64


def base64_to_image(base64_string):
    """
    Convert base64 string to PIL Image.
    
    Args:
        base64_string (str): Base64 encoded image string
    
    Returns:
        PIL.Image: Decoded image
    """
    # Decode base64
    img_data = base64.b64decode(base64_string)
    
    # Load from BytesIO buffer
    buffer = BytesIO(img_data)
    image = Image.open(buffer)
    
    return image


def resize_image(image, size):
    """
    Resize image to target size.
    
    Args:
        image (np.ndarray or PIL.Image): Input image
        size (tuple): Target size (width, height)
    
    Returns:
        np.ndarray: Resized image
    """
    if isinstance(image, Image.Image):
        image = pil_to_numpy(image)
    
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    return resized


def normalize_image(image):
    """
    Normalize image to [0, 255] range.
    
    Args:
        image (np.ndarray): Input image
    
    Returns:
        np.ndarray: Normalized image (uint8)
    """
    # Normalize to [0, 1]
    normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Scale to [0, 255]
    normalized = (normalized * 255).astype(np.uint8)
    
    return normalized


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image.
    
    Args:
        image (np.ndarray): Grayscale image
        clip_limit (float): Threshold for contrast limiting
        tile_grid_size (tuple): Size of grid for histogram equalization
    
    Returns:
        np.ndarray: Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced


def calculate_confidence(probability_map, mask):
    """
    Calculate segmentation confidence based on probability values.
    
    Computes mean probability of pixels within the segmented region.
    Higher values indicate more confident predictions.
    
    Args:
        probability_map (np.ndarray): Probability map from model (H, W) with values [0, 1]
        mask (np.ndarray): Binary mask (H, W) with values 0 or 255
    
    Returns:
        float: Confidence score [0, 1]
    """
    # Convert mask to binary (0 or 1)
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Get probabilities within segmented region
    if mask_binary.sum() == 0:
        # No segmentation found
        return 0.0
    
    segmented_probs = probability_map[mask_binary == 1]
    
    # Calculate mean probability (confidence)
    confidence = float(np.mean(segmented_probs))
    
    return confidence
