import cv2
import sys
import os

# Add services to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "services")))
from preprocessing_service.image_enhancing import ImageEnhancer

def test_enhancement(img_path):
    enhancer = ImageEnhancer(sigma=1.2)
    try:
        binarized, meta = enhancer.enhance(img_path)
        out_path = "enhanced_test_math.png"
        cv2.imwrite(out_path, binarized)
        print(f"Success! Enhanced image saved to {out_path}")
        print(f"Metadata: {meta}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_enhancement("test_math.png")
