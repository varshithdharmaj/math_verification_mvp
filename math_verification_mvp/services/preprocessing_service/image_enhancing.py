import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class ImageEnhancer:
    """
    Applies neuro-symbolic preprocessing directly from the MVM² architecture.
    Enhances mathematical images for optimal OCR extraction.
    """
    def __init__(self, sigma: float = 1.2):
        self.sigma = sigma
    
    def calculate_contrast(self, gray_img: np.ndarray) -> float:
        """Calculate RMS contrast."""
        if gray_img is None or gray_img.size == 0:
            return 0.0
        return float(gray_img.std())

    def enhance(self, image_source: Union[str, Path, bytes, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhance image using Gaussian Blur, CLAHE, and Adaptive Binarization.
        Returns the enhanced image (as numpy array) and metadata tagged with quality metrics.
        """
        if isinstance(image_source, (str, Path)):
            img = cv2.imread(str(image_source))
            if img is None:
                raise ValueError(f"Could not load image at {image_source}")
        elif isinstance(image_source, bytes):
            nparr = np.frombuffer(image_source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from bytes")
        elif isinstance(image_source, np.ndarray):
            img = image_source
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")
            
        height, width = img.shape[:2]
        
        # 1. Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        initial_contrast = self.calculate_contrast(gray)
            
        # 2. Gaussian Blur (sigma=1.2)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        
        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(blurred)
        
        # 4. Adaptive Binarization (Lighting normalization)
        binarized = cv2.adaptiveThreshold(
            clahe_img, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        final_contrast = self.calculate_contrast(binarized)
        
        metadata = {
            "resolution": {"width": width, "height": height},
            "metrics": {
                "initial_contrast": round(initial_contrast, 2),
                "final_contrast": round(final_contrast, 2),
                "blur_sigma_used": self.sigma
            },
            "processing_steps": ["grayscale", f"gaussian_blur_sigma_{self.sigma}", "clahe", "adaptive_binarization"]
        }
        
        return binarized, metadata
