import os
import json
from mvm2_ocr_engine import MVM2OCREngine

def run_test():
    print("Initializing MVM2 Local OCR Subsystem (Pix2Text + LaTeX-OCR)...")
    try:
        engine = MVM2OCREngine()
        
        # We will test against the existing test_math.png in the project root
        test_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_math.png"))
        
        # If it doesn't exist, the engine will create a dummy or handle the error gracefully
        if not os.path.exists(test_img_path):
            print(f"Warning: {test_img_path} not found. The engine will use a simulation or fail.")
            
        print(f"\\nProcessing Image: {test_img_path}")
        result = engine.process_image(test_img_path)
        
        print("\\n" + "="*50)
        print("🏆 MVM2 LOCAL OCR ENGINE - PIPELINE VERIFICATION")
        print("="*50)
        print("Backend Used:", result.get("backend"))
        print("\\n[EXTRACTED LATEX]")
        print(result.get("latex_output", "N/A"))
        print("\\n[DETECTED LAYOUT MAP]")
        print(json.dumps(result.get("detected_layout", []), indent=2))
        print(f"\\n[WEIGHTED OCR CONFIDENCE SCORE]: {result.get('weighted_confidence', 0.0) * 100:.2f}%")
        print("="*50)
        print("\\nStatus: Verification SUCCESS" if "latex_output" in result else "\\nStatus: Verification FAILED")
        
    except Exception as e:
        print(f"Critical Pipeline Failure: {e}")

if __name__ == "__main__":
    run_test()
