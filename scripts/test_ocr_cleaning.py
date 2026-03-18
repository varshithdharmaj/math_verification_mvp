
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from math_verification_mvp.services.local_ocr.mvm2_ocr_engine import MVM2OCREngine

def test_cjk_cleaning():
    engine = MVM2OCREngine()
    
    # Test case: Mixed LaTeX and Chinese/Japanese/Korean characters
    input_text = "The equation is 2x + 4 = 10. 它是在 LaTeX 中提取的。 数式 2x + 4 = 10 が抽出されました。"
    expected_output = "The equation is 2x + 4 = 10.   LaTeX  .  2x + 4 = 10  ."
    
    cleaned = engine.clean_latex_output(input_text)
    
    print(f"Original: {input_text}")
    print(f"Cleaned:  {cleaned}")
    
    # Check if any CJK characters remain
    import re
    cjk_re = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
    if cjk_re.search(cleaned):
        print("FAIL: CJK characters still present in output!")
        return False
    else:
        print("SUCCESS: CJK characters successfully stripped.")
        return True

if __name__ == "__main__":
    if test_cjk_cleaning():
        sys.exit(0)
    else:
        sys.exit(1)
