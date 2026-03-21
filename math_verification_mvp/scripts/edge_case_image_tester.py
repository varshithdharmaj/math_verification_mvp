import os
import sys
from PIL import Image
import json

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "services"))

from services.local_ocr.mvm2_ocr_engine import MVM2OCREngine
from services.preprocessing_service.image_enhancing import ImageEnhancer

def test_edge_cases():
    engine = MVM2OCREngine()
    enhancer = ImageEnhancer(sigma=1.2)
    
    cases = {
        "empty": "empty.png",
        "small": "small.png",
        "large": "large.png",
        "no_math": "no_math.png",
        "corrupt": "corrupt.txt"
    }

    # Generate edge case images
    try:
        # Small image (1x1)
        Image.new('RGB', (1, 1), color='white').save(cases["small"]) 
        # Large image (2000x2000 is enough for test, 10k might be too much for memory in some envs)
        Image.new('RGB', (2000, 2000), color='white').save(cases["large"]) 
        # No math (pitch black)
        Image.new('RGB', (200, 100), color='black').save(cases["no_math"]) 
        # Corrupt file
        with open(cases["corrupt"], "w") as f:
            f.write("this is not an image")
        # Empty file (0 bytes) - renamed from empty.png because PIL can't save 0x0
        open(cases["empty"], 'a').close()
        
    except Exception as e:
        print(f"Setup Error: {e}")

    for name, path in cases.items():
        print(f"\n🧪 Testing Edge Case: {name.upper()} ({path})")
        try:
            # Test Enhancer
            print(f" - Running Enhancer...")
            enhanced, meta = enhancer.enhance(path)
            print(f"   [OK] Enhanced. Shape: {enhanced.shape}, Meta: {meta}")
            
            # Test OCR Engine
            print(f" - Running OCR Engine...")
            res = engine.process_image(path)
            print(f"   [OK] OCR Result: {res.get('backend')} - LaTeX: {res.get('latex_output')[:50]}...")
            
        except Exception as e:
            print(f" ❌ FAILED on {name}: {e}")
        finally:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    test_edge_cases()
