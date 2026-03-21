import os
import sys
import json
import cv2
import numpy as np
from PIL import Image

# Ensure paths are correct
PROJECT_ROOT = r"c:\Users\Varshith Dharmaj\Downloads\major\math_verification_mvp"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

local_ocr_path = os.path.join(PROJECT_ROOT, "services", "local_ocr")
if local_ocr_path not in sys.path:
    sys.path.append(local_ocr_path)

from mvm2_ocr_engine import MVM2OCREngine

def run_diagnostics():
    print("MVM2 OCR DIAGNOSTIC TOOL")
    print("========================")
    
    engine = MVM2OCREngine()
    print(f"Engine Model Loaded: {engine.model_loaded}")
    
    test_images = [
        "test_math.png",
        "services/dashboard/test_math.png" # Sometimes it's duplicated
    ]
    
    # Create a synthetic complex math image if others don't exist
    synth_path = "synth_math.png"
    img = Image.new('RGB', (800, 200), color = 'white')
    # Since I can't draw complex LaTeX easily here, I'll just check if existing ones work
    # But I can generate an image with text via CV2
    synth_img = np.ones((200, 800, 3), dtype=np.uint8) * 255
    cv2.putText(synth_img, "f(x) = x^2 + 2x + 1", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.imwrite(synth_path, synth_img)
    test_images.append(synth_path)

    from services.preprocessing_service.image_enhancing import ImageEnhancer
    enhancer = ImageEnhancer(sigma=1.2)

    for img_path in test_images:
        abs_path = os.path.abspath(img_path)
        if not os.path.exists(abs_path):
            print(f"\n[SKIP] {img_path} not found.")
            continue
            
        print(f"\n[TESTING RAW] {img_path}")
        result_raw = engine.process_image(abs_path)
        print(f"Raw Result: {result_raw.get('latex_output')}")
        
        print(f"\n[TESTING ENHANCED] {img_path}")
        try:
            enhanced_img, _ = enhancer.enhance(abs_path)
            enhanced_tmp = f"enhanced_{os.path.basename(img_path)}"
            cv2.imwrite(enhanced_tmp, enhanced_img)
            result_enh = engine.process_image(enhanced_tmp)
            print(f"Enhanced Result: {result_enh.get('latex_output')}")
            os.remove(enhanced_tmp)
        except Exception as e:
            print(f"Enhancement failed: {e}")

if __name__ == "__main__":
    run_diagnostics()
