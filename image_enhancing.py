import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Union
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Applies the handwritten-math-optimized preprocessing stack from the MVM² architecture.

    Pipeline:
    1. Robust loading from multiple input types (path / bytes / numpy / PIL).
    2. Convert to grayscale and measure initial contrast.
    3. Apply Gaussian blur (stabilizes stroke noise for handwriting).
    4. Apply CLAHE to locally boost contrast on notebook paper.
    5. Optionally apply adaptive binarization if the page is low contrast.
    """

    def __init__(self, sigma: float = 1.2):
        # Gaussian standard deviation; tuned for typical notebook handwriting.
        self.sigma = sigma

    def calculate_contrast(self, gray_img: np.ndarray) -> float:
        """
        Simple contrast proxy: standard deviation of grayscale intensities.
        """
        if gray_img is None or gray_img.size == 0:
            return 0.0
        return float(gray_img.std())

    def enhance(
        self,
        image_source: Union[str, Path, bytes, np.ndarray, Image.Image],
        skip_binarization: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Core handwritten-math enhancement routine (CLAHE + Gaussian blur + optional binarization).

        Supports:
        - str / Path: filesystem path to an image.
        - bytes: raw encoded image bytes.
        - np.ndarray: BGR / grayscale OpenCV image.
        - PIL.Image.Image: Gradio / HF directly supplies PIL objects.
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
        elif isinstance(image_source, Image.Image):
            # Gradio hands us a PIL Image when type="pil"; convert to OpenCV BGR.
            img = cv2.cvtColor(np.array(image_source.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

        height, width = img.shape[:2]

        # Always work in grayscale for the enhancer.
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        initial_contrast = self.calculate_contrast(gray)

        # Gaussian Blur (sigma tuned for handwriting strokes).
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(blurred)

        # Adaptive binarization only when the notebook page is low-contrast.
        if skip_binarization or initial_contrast > 60:
            final_img = clahe_img
            bin_applied = False
        else:
            final_img = cv2.adaptiveThreshold(
                clahe_img,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )
            bin_applied = True

        final_contrast = self.calculate_contrast(final_img)

        metadata = {
            "resolution": {"width": width, "height": height},
            "metrics": {
                "initial_contrast": round(initial_contrast, 2),
                "final_contrast": round(final_contrast, 2),
                "blur_sigma_used": self.sigma,
                "binarization_applied": bin_applied,
            },
        }
        return final_img, metadata
