"""
OCR Service Module
Responsible for extracting text from images using Tesseract or handwritten models.
"""
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image
import pytesseract
import cv2
import numpy as np
import base64
import io
import time

# Import handwritten math OCR
try:
    from .handwritten_math_ocr import HandwrittenMathOCR
    HANDWRITTEN_OCR_AVAILABLE = True
except ImportError:
    HANDWRITTEN_OCR_AVAILABLE = False
    print("[WARN] Handwritten Math OCR not available")

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
        
        # Preprocess image (External Service)
        from backend.core.preprocessing_service import preprocess_image
        processed = preprocess_image(image)
        
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
            
            # Identify tokens and their confidences (Simplistic mapping)
            tokens = [t for t in data['text'] if t.strip()]
            token_confs = [int(c)/100.0 for t, c in zip(data['text'], data['conf']) if t.strip() and c != '-1']
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'extracted_text': text.strip(),
                'confidence': avg_confidence / 100.0,  # Normalize to 0-1
                'method': 'Tesseract with math config',
                'tokens': tokens,
                'token_confidences': token_confs
            }
        
        except Exception as e:
            return {
                'extracted_text': '',
                'confidence': 0.0,
                'error': str(e),
                'method': 'Tesseract (failed)',
                'tokens': [],
                'token_confidences': []
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
            
            # Mocking tokens for this backend as it returns whole latex
            return {
                **result,
                'tokens': result['extracted_text'].split(),
                'token_confidences': [result.get('confidence', 0.8)] * len(result['extracted_text'].split())
            }
            
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

# Global instance
ocr_engine = EnhancedMathOCR()

def compute_ocr_confidence(tokens: List[str], confidences: List[float]) -> float:
    """
    Computes weighted OCR confidence based on token importance (MVM² Eq 2-4).
    
    Weights:
    - High (2.0): Operators, Brackets (Crucial for structure)
    - Medium (1.0): Digits, Variables (Content)
    - Low (0.5): Ambiguous/Noise (l, 1, O, 0, etc.) or unknown
    """
    if not tokens or not confidences:
        return 0.0
    
    if len(tokens) != len(confidences):
        # Fallback if mismatch
        return sum(confidences) / len(confidences)
        
    # Heuristics
    OPERATORS_BRACKETS = set("+-*/=()[]{}<>^√∫∑∏")
    AMBIGUOUS = set("l1O0S5Z2g9q")
    
    total_weighted_conf = 0.0
    total_weight = 0.0
    
    for token, conf in zip(tokens, confidences):
        w = 1.0 # Default (Medium)
        
        # Check first char or heuristic for whole token
        t_clean = token.strip()
        if not t_clean: 
            continue
            
        first_char = t_clean[0]
        
        if first_char in OPERATORS_BRACKETS or any(c in OPERATORS_BRACKETS for c in t_clean):
            w = 2.0
        elif first_char in AMBIGUOUS and len(t_clean) == 1:
            w = 0.5
        elif t_clean.isalpha() or t_clean.isdigit():
            w = 1.0
        else:
            w = 0.5 # Unknown symbols/noise
            
        total_weighted_conf += w * conf
        total_weight += w
        
    if total_weight == 0:
        return 0.0
        
    return total_weighted_conf / total_weight

async def run_math_ocr(image_bytes: bytes) -> Tuple[List[str], List[float], str]:
    """
    Runs OCR on the provided image bytes.
    
    Returns:
        tokens (List[str]): List of detected tokens/words.
        confidences (List[float]): Confidence score for each token (0-1).
        raw_text (str): Full extracted text.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        result = ocr_engine.extract_text(image)
        
        raw_text = result.get('extracted_text', '')
        
        # Get enriched token data if available, else derive
        tokens = result.get('tokens', raw_text.split())
        
        # Ensure confidences match tokens
        if 'token_confidences' in result and len(result['token_confidences']) == len(tokens):
            confidences = result['token_confidences']
        else:
            # Fallback uniform confidence
            confidences = [result.get('confidence', 0.0)] * len(tokens)
        
        return tokens, confidences, raw_text
    except Exception as e:
        print(f"OCR Failed: {e}")
        return [], [], ""

async def extract_text(image_bytes: bytes) -> str:
    """Wrapper for backward compatibility or simple calls"""
    _, _, text = await run_math_ocr(image_bytes)
    return text

async def get_ocr_confidence(image_bytes: bytes) -> float:
    """Wrapper using the new weighted logic"""
    tokens, confs, _ = await run_math_ocr(image_bytes)
    return compute_ocr_confidence(tokens, confs)

if __name__ == "__main__":
    # Unit Tests for OCR Confidence
    print("Running Unit Tests for compute_ocr_confidence...")
    
    # CASE 1: Perfect confidence, mixed types
    # 3x + 5 = 20
    t1 = ["3", "x", "+", "5", "=", "20"]
    c1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # Exp: All 1.0 -> 1.0
    assert abs(compute_ocr_confidence(t1, c1) - 1.0) < 0.001, "Test 1 Failed"
    print("Test 1 (Perfect) Passed")
    
    # CASE 2: Low confidence on Operator (High Weight)
    # + is 0.0, others 1.0
    # Note: '5' is in AMBIGUOUS set, so w=0.5
    t2 = ["3", "+", "5"]
    c2 = [1.0, 0.0, 1.0] 
    # Weights: 3(1.0), +(2.0), 5(0.5) -> Total W=3.5
    # Score: (1*1 + 2*0 + 0.5*1) / 3.5 = 1.5/3.5 ≈ 0.4286
    res2 = compute_ocr_confidence(t2, c2)
    assert abs(res2 - 0.4286) < 0.001, f"Test 2 Failed: Got {res2}"
    print("Test 2 (Operator Penality) Passed")
    
    # CASE 3: Low confidence on Ambiguous char (Low Weight)
    # l (ambiguous) is 0.0, others 1.0
    # Note: 'l' and '5' are in AMBIGUOUS set, w=0.5 each
    t3 = ["l", "+", "5"] # typo for 1
    c3 = [0.0, 1.0, 1.0]
    # Weights: l(0.5), +(2.0), 5(0.5) -> Total W=3.0
    # Score: (0.5*0 + 2*1 + 0.5*1) / 3.0 = 2.5/3.0 ≈ 0.8333
    res3 = compute_ocr_confidence(t3, c3)
    assert abs(res3 - 0.8333) < 0.001, f"Test 3 Failed: Got {res3}"
    print("Test 3 (Ambiguous Tolerance) Passed")
    
    print("All Unit Tests Passed!")
