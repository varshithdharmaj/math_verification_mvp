"""
Download and test with real MathWriting InkML dataset
This will give us the 92% accuracy the model was trained for
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "handwritten-math-transcription"))

from utils import download_data
from services.handwritten_math_ocr import HandwrittenMathOCR
from dataset.hme_ink import read_inkml_file
from PIL import Image

def test_with_real_inkml():
    """Test with real InkML data from MathWriting dataset"""
    print("=" * 60)
    print("Testing with Real MathWriting InkML Dataset")
    print("=" * 60)
    
    # Download the dataset (this will take a few minutes)
    print("\n[1/4] Downloading MathWriting dataset...")
    print("(This is a large dataset ~1GB, will take a few minutes)")
    data_root = download_data("https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz")
    print(f"Dataset downloaded to: {data_root}")
    
    # Load OCR model
    print("\n[2/4] Loading handwritten math OCR model...")
    ocr = HandwrittenMathOCR()
    ocr.load_model()
    
    if not ocr.model_loaded:
        print("[ERROR] Model failed to load")
        return False
    
    print("[OK] Model loaded successfully!")
    
    # Test with a real InkML file
    print("\n[3/4] Testing with real InkML file...")
    test_inkml_path = os.path.join(data_root, "test", "00c46c9b07b39bb7.inkml")
    
    if not os.path.exists(test_inkml_path):
        # Try to find any inkml file
        import glob
        inkml_files = glob.glob(os.path.join(data_root, "test", "*.inkml"))
        if inkml_files:
            test_inkml_path = inkml_files[0]
        else:
            print(f"[ERROR] No InkML files found in {data_root}/test/")
            return False
    
    print(f"Using InkML file: {test_inkml_path}")
    
    # Read InkML file
    ink = read_inkml_file(test_inkml_path)
    ground_truth = ink.annotations.get('normalizedLabel', 'N/A')
    
    print(f"Ground truth LaTeX: {ground_truth}")
    
    # Run inference using the model's native inference function
    print("\n[4/4] Running inference...")
    from handwritten-math-transcription.main import inference
    
    try:
        predicted, actual, _ = inference(ocr.model, ink_file_path=test_inkml_path, apply_correction=False)
        
        print("\n" + "=" * 60)
        print("RESULTS WITH REAL InkML DATA")
        print("=" * 60)
        print(f"Predicted: {predicted}")
        print(f"Actual:    {actual}")
        print(f"Match:     {'YES!' if predicted == actual else 'No'}")
        print("=" * 60)
        
        return predicted == actual
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_real_inkml()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
