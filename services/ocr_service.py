"""
Enhanced OCR Service with Multi-Backend Support
Integrates Tesseract + Handwritten Math OCR (johnkimdw/handwritten-math-transcription)
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import pytesseract
import cv2
import numpy as np
import io
from typing import List, Dict, Optional
import time

# Import handwritten math OCR
try:
    from services.handwritten_math_ocr import HandwrittenMathOCR
    HANDWRITTEN_OCR_AVAILABLE = True
except ImportError:
    HANDWRITTEN_OCR_AVAILABLE = False
    print("[WARN] Handwritten Math OCR not available")

app = FastAPI(
    title="Enhanced OCR Service with Math Support",
    description="Multi-backend OCR with specialized math handwriting recognition",
    version="3.0.0"
)

class OCRResponse(BaseModel):
    extracted_text: str
    confidence: float
    backend_used: str
    processing_time: float
    normalized_text: str
    problem: str
    steps: List[str]
    ocr_confidence: float
    latex: Optional[str] = None  # LaTeX output from handwritten OCR

class EnhancedMathOCR:
    """
    Enhanced OCR with multiple backend support
    - Tesseract (for printed text)
    - Handwritten Math OCR (for handwritten equations)
    - Math-specific preprocessing
    """
    
    def __init__(self):
        self.backends = {
            'tesseract': self._tesseract_ocr,
            'handwritten_math': self._handwritten_math_ocr
        }
        self.handwritten_ocr = HandwrittenMathOCR() if HANDWRITTEN_OCR_AVAILABLE else None
        
    def extract_text(self, image: Image.Image, backend: str = 'auto') -> Dict:
        """
        Extract text from image using specified backend
        """
        start = time.time()
        
        # Preprocess image
        processed = self._preprocess_for_math(image)
        
        # Auto-select backend based on content
        if backend == 'auto':
            backend = self._select_backend(processed)
        
        # Extract text
        if backend in self.backends:
            result = self.backends[backend](processed)
        else:
            result = self._tesseract_ocr(processed)  # Fallback
        
        # Normalize mathematical notation
        result['normalized_text'] = self._normalize_math(result['extracted_text'])
        result['processing_time'] = time.time() - start
        result['backend_used'] = backend
        
        # Parse problem and steps (simple heuristic for now)
        lines = [line.strip() for line in result['normalized_text'].split('\n') if line.strip()]
        result['problem'] = lines[0] if lines else ""
        result['steps'] = lines[1:] if len(lines) > 1 else []
        result['ocr_confidence'] = result['confidence']
        
        return result
    
    def _preprocess_for_math(self, image: Image.Image) -> Image.Image:
        """
        Enhanced preprocessing for mathematical content
        - Binarization
        - Noise reduction
        - Contrast enhancement
        """
        # Convert to numpy array
        img_array = np.array(image.convert('L'))
        
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
    
    def _select_backend(self, image: Image.Image) -> str:
        """
        Auto-select OCR backend based on image characteristics
        """
        # For now, always use Tesseract
        # Future: Detect handwriting vs printed, complexity, etc.
        return 'tesseract'
    
    def _tesseract_ocr(self, image: Image.Image) -> Dict:
        """
        Tesseract OCR with math-optimized configuration
        """
        try:
            # Configure Tesseract for better math recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-×÷=().,abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
            
            # Extract text
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Get confidence
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data['conf'] if c != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'extracted_text': text.strip(),
                'confidence': avg_confidence / 100.0,  # Normalize to 0-1
                'method': 'Tesseract with math config'
            }
        
        except Exception as e:
            return {
                'extracted_text': '',
                'confidence': 0.0,
                'error': str(e),
                'method': 'Tesseract (failed)'
            }
    
    def _handwritten_math_ocr(self, image: Image.Image) -> Dict:
        """
        Handwritten Math OCR using johnkimdw/handwritten-math-transcription
        Converts handwritten math to LaTeX
        """
        if not self.handwritten_ocr:
            # Fallback to Tesseract if model unavailable
            result = self._tesseract_ocr(image)
            result['method'] = 'Tesseract (Handwritten OCR unavailable)'
            return result
        
        try:
            # Use handwritten math OCR model
            result = self.handwritten_ocr.transcribe(image)
            
            # Convert LaTeX to plain text if available
            latex = result.get('latex', '')
            if latex:
                # Simple LaTeX to text conversion
                plain_text = latex.replace('\\frac', '').replace('{', '').replace('}', '')
                plain_text = plain_text.replace('\\', '')
                result['extracted_text'] = plain_text
            else:
                result['extracted_text'] = ''
            
            return result
            
        except Exception as e:
            # Fallback to Tesseract on error
            result = self._tesseract_ocr(image)
            result['method'] = f'Tesseract (Handwritten OCR failed: {str(e)})'
            return result
    
    def _normalize_math(self, text: str) -> str:
        """
        Normalize mathematical symbols and notation
        """
        # Symbol replacements
        replacements = {
            '×': '*',
            '÷': '/',
            '−': '-',
            '·': '*',
            ' x ': ' * ',
            ' X ': ' * ',
            '**': '^',
        }
        
        normalized = text
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # Clean up whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized


# Global OCR instance
ocr_engine = EnhancedMathOCR()


@app.post("/extract", response_model=OCRResponse)
async def extract_text(
    file: UploadFile = File(...),
    backend: str = 'auto'
):
    """
    Extract text from uploaded image
    Supports: auto, tesseract, math_specialized
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Extract text
        result = ocr_engine.extract_text(image, backend=backend)
        
        return OCRResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "enhanced_ocr",
        "version": "2.0",
        "backends": list(ocr_engine.backends.keys())
    }


@app.get("/info")
async def service_info():
    return {
        "service": "Enhanced Math OCR Service",
        "capabilities": [
            "Tesseract OCR",
            "Math-specific preprocessing",
            "Symbol normalization",
            "Multi-backend support (planned)"
        ],
        "future_integrations": [
            "MathAI specialized model",
            "Custom handwriting recognition",
            "LaTeX generation"
        ],
        "references": [
            "Math_Handwriting_OCR resources",
            "MathAI (Tensorflow)",
            "Advanced OCR methods"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("[START] Enhanced OCR Service on port 8001...")
    print("[OK] Tesseract backend loaded")
    print("[INFO] Math-specialized backends: planned")
    uvicorn.run(app, host="0.0.0.0", port=8001)
