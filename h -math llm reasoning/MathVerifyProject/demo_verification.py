"""
Demo script for MathVerifyProject
Tests the core verification functionality
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from core_verification import MathVerifier


def demo_verification():
    """Demonstrate core verification functionality."""
    print("=" * 60)
    print("MathVerifyProject - Verification Demo")
    print("=" * 60)
    print()
    
    # Initialize verifier
    print("Initializing MathVerifier...")
    verifier = MathVerifier()
    print("✓ Verifier initialized\n")
    
    # Test cases
    test_cases = [
        {
            "name": "Simple fraction equivalence",
            "gold": "1/2",
            "pred": "0.5",
            "expected": True
        },
        {
            "name": "LaTeX fraction",
            "gold": "$\\frac{1}{2}$",
            "pred": "0.5",
            "expected": True
        },
        {
            "name": "Set union",
            "gold": "${1,3} \\cup {2,4}$",
            "pred": "${1,2,3,4}$",
            "expected": True
        },
        {
            "name": "Incorrect answer",
            "gold": "42",
            "pred": "43",
            "expected": False
        },
        {
            "name": "Square root",
            "gold": "$\\sqrt{4}$",
            "pred": "2",
            "expected": True
        }
    ]
    
    print("Running test cases...")
    print("-" * 60)
    
    passed = 0
    total = len(test_cases)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"  Gold: {test['gold']}")
        print(f"  Pred: {test['pred']}")
        
        try:
            result = verifier.verify_answer(test['gold'], test['pred'])
            status = "✓" if result == test['expected'] else "✗"
            print(f"  Result: {result} {status}")
            
            if result == test['expected']:
                passed += 1
            else:
                print(f"  Expected: {test['expected']}, Got: {result}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print()
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == '__main__':
    try:
        demo_verification()
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        print("  pip install math-verify[antlr4_13_2]")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

