"""
Preprocessing Service Module
Handles image cleaning and enhancement before OCR.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Applies filters, binarization, and noise reduction to the image.
    
    Args:
        image: Input PIL Image.
        
    Returns:
        Image.Image: Processed image ready for OCR.
    """
    # Convert to numpy array
    if image.mode != 'L':
        image = image.convert('L')
    
    img_array = np.array(image)
    
    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(img_array, (3, 3), 0)
    
    # Adaptive thresholding for better symbol recognition
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # C constant
    )
    
    # Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return Image.fromarray(cleaned)

def check_image_quality(image: Image.Image) -> float:
    """
    Assess if image is clear enough for processing.
    Returns a score between 0.0 and 1.0.
    """
    # Simple heuristic: Contrast and Sharpness
    img_array = np.array(image.convert('L'))
    
    # Contrast
    contrast = img_array.std()
    
    # Sharpness (Variance of Laplacian)
    laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Normalize (heuristics based on typical document images)
    norm_contrast = min(1.0, contrast / 50.0)
    norm_sharpness = min(1.0, sharpness / 500.0)
    
    return (norm_contrast + norm_sharpness) / 2.0
