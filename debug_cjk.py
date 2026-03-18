import sys
import os

# Set paths
PROJ_ROOT = r'c:\Users\Varshith Dharmaj\Downloads\major\math_verification_mvp'
sys.path.append(PROJ_ROOT)

from services.local_ocr.mvm2_ocr_engine import clean_latex_output, extract_latex_from_pix2text

test_input = """ $y$ 
 $R$ 1
 $o$ I 4 $x$ 
Figure 3
ige sos akth of artofth ere wih eqution $y=x^{\frac{1} {2}} \ln2 x .$ 
Me icaorg $R$ ,shown shaded in Figure 3 is bunded b the curve,
 $x$ -axis and the lines $x=1$ and $x=4$ 
沪州批发发大米有发中司中。
 $R$ , giving your answer to 2 decimal places.
O)Fnd $\int x^{\frac{1} {2}} \ln2 x$ ds (4) (c) Hence find the exact area of $R$ giving your answer in the form $a \operatorname{l n} 2+b ,$ .
where $a$ and $b$ are exact constants.
(3)"""

print("--- 1. Testing clean_latex_output directly ---")
cleaned_direct = clean_latex_output(test_input)
print(f"Contains '沪': {'沪' in cleaned_direct}")
print(f"Contains '中': {'中' in cleaned_direct}")
print(f"Remaining CJK: {[c for c in cleaned_direct if ord(c) > 127]}")

print("\n--- 2. Testing extract_latex_from_pix2text with list of dicts (Mixed Mode) ---")
mock_out = [
    {"type": "text", "text": "Me icaorg R"},
    {"type": "formula", "text": "$y=x^{\\frac{1} {2}} \\ln2 x$"},
    {"type": "text", "text": "沪州批发发大米有发中司中。"}
]
result_list = extract_latex_from_pix2text(mock_out)
print(f"Result List: {result_list}")
print(f"Contains CJK: {'沪' in result_list or '中' in result_list}")

print("\n--- 3. Testing extract_latex_from_pix2text with raw string ---")
result_str = extract_latex_from_pix2text(test_input)
print(f"Contains CJK in Str Result: {'沪' in result_str}")
