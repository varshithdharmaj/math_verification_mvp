import sys
import os
sys.path.append(os.path.join(os.getcwd(), "Math-Verify", "src"))
from math_verify import parse, verify

print("Testing Math-Verify...")

# Test case 1: Simple set union
gold = parse("${1,3} \\cup {2,4}$")
answer = parse("${1,2,3,4}$")
result = verify(gold, answer)
print(f"Test 1 (Set Union): {result}")

# Test case 2: Arithmetic
gold = parse("4")
answer = parse("2 + 2")
result = verify(gold, answer)
print(f"Test 2 (Arithmetic): {result}")

# Test case 3: Incorrect
gold = parse("5")
answer = parse("2 + 2")
result = verify(gold, answer)
print(f"Test 3 (Incorrect): {result}")

if result == False:
    print("✅ Math-Verify is working correctly!")
else:
    print("❌ Math-Verify failed validation!")
