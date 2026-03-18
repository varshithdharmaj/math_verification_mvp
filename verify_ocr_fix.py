import sys
import os
sys.path.append(r'c:\Users\Varshith Dharmaj\Downloads\major\math_verification_mvp')

from services.local_ocr.mvm2_ocr_engine import clean_latex_output

test_string = """ $y$ 
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

print("--- RAW TEXT ---")
print(test_string)
print("\n--- CLEANED TEXT ---")
cleaned = clean_latex_output(test_string)
print(cleaned)

if "沪州" in cleaned:
    print("\n[FAIL] Chinese characters still present.")
else:
    print("\n[SUCCESS] Chinese characters removed successfully.")
