"""
Stroke Extraction Module
Converts raster images to vector strokes for handwritten math OCR
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
import torch


class StrokeExtractor:
    """
    Extracts pen strokes from handwritten math images
    Converts raster to vector representation similar to InkML
    """
    
    def __init__(self):
        self.min_stroke_length = 5  # Minimum points per stroke
        
    def extract_strokes(self, image: Image.Image) -> List[np.ndarray]:
        """
        Extract strokes from image using skeletonization and contour tracing
        Returns list of strokes, each stroke is Nx2 array of (x, y) coordinates
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        img_array = np.array(image)
        
        # Invert if needed (we want black ink on white background)
        if np.mean(img_array) < 128:
            img_array = 255 - img_array
        
        # Binarize
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Skeletonize to get thin strokes
        skeleton = self._skeletonize(binary)
        
        # Find connected components (individual strokes)
        strokes = self._trace_strokes(skeleton)
        
        # Filter out very short strokes (noise)
        strokes = [s for s in strokes if len(s) >= self.min_stroke_length]
        
        return strokes
    
    def _skeletonize(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Skeletonize binary image using morphological thinning
        """
        # Use Zhang-Suen thinning algorithm
        skeleton = binary_image.copy()
        skeleton = cv2.ximgproc.thinning(skeleton, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return skeleton
    
    def _trace_strokes(self, skeleton: np.ndarray) -> List[np.ndarray]:
        """
        Trace individual strokes from skeletonized image
        Uses endpoint detection and ordered tracing
        """
        strokes = []
        visited = np.zeros_like(skeleton, dtype=bool)
        
        # Find endpoints and junctions
        endpoints = self._find_endpoints(skeleton)
        
        # Start tracing from endpoints for better stroke ordering
        for y, x in endpoints:
            if visited[y, x]:
                continue
            
            stroke = self._trace_ordered_stroke(skeleton, (x, y), visited)
            if len(stroke) >= self.min_stroke_length:
                strokes.append(np.array(stroke))
        
        # Trace any remaining unvisited pixels
        stroke_pixels = np.argwhere(skeleton > 0)
        for y, x in stroke_pixels:
            if visited[y, x]:
                continue
            
            stroke = self._trace_ordered_stroke(skeleton, (x, y), visited)
            if len(stroke) >= self.min_stroke_length:
                strokes.append(np.array(stroke))
        
        # Sort strokes by reading order (left-to-right, top-to-bottom)
        strokes = sorted(strokes, key=lambda s: (np.mean(s[:, 1]), np.mean(s[:, 0])))
        
        return strokes
    
    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoints (pixels with only one neighbor)"""
        endpoints = []
        for y in range(1, skeleton.shape[0] - 1):
            for x in range(1, skeleton.shape[1] - 1):
                if skeleton[y, x] == 0:
                    continue
                
                # Count neighbors
                neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        if skeleton[y + dy, x + dx] > 0:
                            neighbors += 1
                
                # Endpoint has exactly 1 neighbor
                if neighbors == 1:
                    endpoints.append((y, x))
        
        return endpoints
    
    def _trace_ordered_stroke(self, skeleton: np.ndarray, start: Tuple[int, int],
                              visited: np.ndarray) -> List[Tuple[int, int]]:
        """
        Trace stroke in order from start point
        """
        stroke = []
        current = start
        
        while current is not None:
            x, y = current
            
            if visited[y, x]:
                break
            
            visited[y, x] = True
            stroke.append((x, y))
            
            # Find next unvisited neighbor
            next_point = None
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    
                    if (0 <= ny < skeleton.shape[0] and
                        0 <= nx < skeleton.shape[1] and
                        skeleton[ny, nx] > 0 and
                        not visited[ny, nx]):
                        next_point = (nx, ny)
                        break
                
                if next_point:
                    break
            
            current = next_point
        
        return stroke
    
    def strokes_to_features(self, strokes: List[np.ndarray], image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Convert strokes to feature tensor matching InkML format
        Features: [x, y, dx, dy, speed, curvature, pressure, pen_up, ...]
        """
        all_features = []
        width, height = image_size
        
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            
            # Normalize coordinates to [0, 1]
            stroke_norm = stroke.astype(float)
            stroke_norm[:, 0] /= width
            stroke_norm[:, 1] /= height
            
            # Calculate derivatives (velocity)
            dx = np.diff(stroke_norm[:, 0], prepend=stroke_norm[0, 0])
            dy = np.diff(stroke_norm[:, 1], prepend=stroke_norm[0, 1])
            
            # Calculate speed
            speed = np.sqrt(dx**2 + dy**2)
            
            # Calculate curvature (angle change)
            angles = np.arctan2(dy, dx)
            curvature = np.diff(angles, prepend=angles[0])
            
            # Simulate pressure (constant for extracted strokes)
            pressure = np.ones(len(stroke_norm))
            
            # Pen state (0 for down, 1 for up at end of stroke)
            pen_state = np.zeros(len(stroke_norm))
            pen_state[-1] = 1  # Pen up at end of stroke
            
            # Time (simulated as cumulative distance)
            time = np.cumsum(speed)
            time = time / (time[-1] + 1e-6)  # Normalize
            
            # Combine features (11 features to match model input)
            features = np.stack([
                stroke_norm[:, 0],  # x
                stroke_norm[:, 1],  # y
                dx,                  # dx
                dy,                  # dy
                speed,               # speed
                curvature,           # curvature
                pressure,            # pressure
                pen_state,           # pen state
                time,                # time
                np.zeros(len(stroke_norm)),  # placeholder
                np.zeros(len(stroke_norm)),  # placeholder
            ], axis=1)
            
            all_features.append(features)
        
        # Concatenate all strokes
        if all_features:
            all_features = np.vstack(all_features)
            return torch.FloatTensor(all_features)
        else:
            return torch.zeros((1, 11))


def extract_features_from_image(image: Image.Image) -> torch.Tensor:
    """
    Main function: Extract InkML-like features from image
    """
    extractor = StrokeExtractor()
    strokes = extractor.extract_strokes(image)
    features = extractor.strokes_to_features(strokes, image.size)
    return features
