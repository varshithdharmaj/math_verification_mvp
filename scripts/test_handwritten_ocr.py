"""
Test script for Handwritten Math OCR
Verifies model loading and inference capability
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from services.handwritten_math_ocr import HandwrittenMathOCR
from PIL import Image

def test_model_loading():
    """Test if the model loads correctly"""
    print("=" * 60)
    print("Testing Handwritten Math OCR Model")
    print("=" * 60)
    
    # Initialize OCR
    ocr = HandwrittenMathOCR()
    
    # Try to load model
    print("\n[1/3] Loading model...")
    ocr.load_model()
    
    if ocr.model_loaded:
        print("[OK] Model loaded successfully!")
        print(f"   Device: {ocr.device}")
        print(f"   Model type: {type(ocr.model).__name__}")
    else:
        print("[ERROR] Model failed to load")
        return False
    
    # Test with example image from repository
    print("\n[2/3] Testing inference...")
    example_image_path = os.path.join(
        "handwritten-math-transcription",
        "example-testing.png"
    )
    
    if os.path.exists(example_image_path):
        try:
            image = Image.open(example_image_path)
            print(f"   Loaded test image: {example_image_path}")
            print(f"   Image size: {image.size}")
            
            # Run inference
            result = ocr.transcribe(image)
            
            print("\n[3/3] Results:")
            print(f"   LaTeX: {result.get('latex', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            print(f"   Method: {result.get('method', 'N/A')}")
            
            if 'error' in result:
                print(f"   [WARN] Error: {result['error']}")
            else:
                print("   [OK] Inference successful!")
            
        except Exception as e:
            print(f"   [ERROR] Inference failed: {e}")
            return False
    else:
        print(f"   [WARN] Example image not found at {example_image_path}")
        print("   Skipping inference test")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
