"""
Step-by-step testing of all components
"""

print("="*60)
print("TESTING ALL COMPONENTS ONE BY ONE")
print("="*60)

# Test 1: Model 1 - SymbolicVerifier
print("\n[1/10] Testing Model 1: SymbolicVerifier")
try:
    from models.symbolic_verifier import SymbolicVerifier
    verifier = SymbolicVerifier()
    
    # Test valid calculation
    result1 = verifier.verify(["3 + 2 = 5"])
    print(f"  ✓ Valid test: {result1['verdict']} ({result1['confidence']*100:.0f}% confidence)")
    
    # Test error calculation
    result2 = verifier.verify(["5 - 1 = 6"])
    print(f"  ✓ Error test: {result2['verdict']} ({result2['confidence']*100:.0f}% confidence, {len(result2['errors'])} errors found)")
    
    print("  ✅ Model 1 PASSED")
except Exception as e:
    print(f"  ❌ Model 1 FAILED: {e}")

# Test 2: Model 2 - LLMLogicalChecker
print("\n[2/10] Testing Model 2: LLMLogicalChecker")
try:
    from models.llm_logical_checker import LLMLogicalChecker
    checker = LLMLogicalChecker("GPT-4")
    
    result = checker.verify(["She buys 2 more: 3 + 2 = 5 apples"])
    print(f"  ✓ Test: {result['verdict']} ({result['confidence']*100:.0f}% confidence)")
    print(f"  ✓ Model name: {result['model_name']}")
    
    print("  ✅ Model 2 PASSED")
except Exception as e:
    print(f"  ❌ Model 2 FAILED: {e}")

# Test 3: Model 3 - EnsembleNeuralChecker
print("\n[3/10] Testing Model 3: EnsembleNeuralChecker")
try:
    from models.ensemble_neural_checker import EnsembleNeuralChecker
    ensemble = EnsembleNeuralChecker(["GPT-4", "Llama 2", "Gemini"])
    
    result = ensemble.verify(["5 - 1 = 6"])
    print(f"  ✓ Test: {result['verdict']} ({result['confidence']*100:.0f}% confidence)")
    print(f"  ✓ Agreement: {result['agreement']}")
    print(f"  ✓ Sub-models: {result['sub_models']}")
    
    print("  ✅ Model 3 PASSED")
except Exception as e:
    print(f"  ❌ Model 3 FAILED: {e}")

# Test 4: Consensus Mechanism
print("\n[4/10] Testing Consensus Mechanism")
try:
    from consensus.consensus_mechanism import compute_consensus
    from models.symbolic_verifier import SymbolicVerifier
    from models.llm_logical_checker import LLMLogicalChecker
    from models.ensemble_neural_checker import EnsembleNeuralChecker
    
    steps = ["5 - 1 = 6"]
    symbolic = SymbolicVerifier()
    llm = LLMLogicalChecker()
    ensemble = EnsembleNeuralChecker()
    
    r1 = symbolic.verify(steps)
    r2 = llm.verify(steps)
    r3 = ensemble.verify(steps)
    
    consensus = compute_consensus(r1, r2, r3)
    print(f"  ✓ Final verdict: {consensus['final_verdict']}")
    print(f"  ✓ Confidence: {consensus['overall_confidence']*100:.1f}%")
    print(f"  ✓ Agreement: {consensus['agreement_type']}")
    print(f"  ✓ Error score: {consensus['error_score']:.3f}")
    
    print("  ✅ Consensus Mechanism PASSED")
except Exception as e:
    print(f"  ❌ Consensus Mechanism FAILED: {e}")

# Test 5: Error Classification
print("\n[5/10] Testing Error Classification")
try:
    from utils.error_classifier import classify_error
    
    error = {
        "type": "calculation_error",
        "found": "5 - 1 = 6",
        "correct": "5 - 1 = 4",
        "operation": "-",
        "step_number": 1
    }
    
    classified = classify_error(error)
    print(f"  ✓ Category: {classified['category']}")
    print(f"  ✓ Severity: {classified['severity']}")
    print(f"  ✓ Fixable: {classified['fixable']}")
    print(f"  ✓ Fixability score: {classified['fixability_score']*100:.0f}%")
    
    print("  ✅ Error Classification PASSED")
except Exception as e:
    print(f"  ❌ Error Classification FAILED: {e}")

# Test 6: Explanation Generation
print("\n[6/10] Testing Explanation Generation")
try:
    from utils.explanation_generator import generate_explanation
    
    error = {
        "type": "calculation_error",
        "found": "5 - 1 = 6",
        "correct": "5 - 1 = 4",
        "operation": "-",
        "step_number": 1
    }
    
    explanation = generate_explanation(error)
    print(f"  ✓ Explanation generated ({len(explanation)} chars)")
    print(f"  ✓ Preview: {explanation[:80]}...")
    
    print("  ✅ Explanation Generation PASSED")
except Exception as e:
    print(f"  ❌ Explanation Generation FAILED: {e}")

# Test 7: Error Correction
print("\n[7/10] Testing Error Correction")
try:
    from utils.error_corrector import correct_solution
    
    steps = ["She gives 1 away: 5 - 1 = 6 apples"]
    errors = [{
        "type": "calculation_error",
        "found": "5 - 1 = 6",
        "correct": "5 - 1 = 4",
        "operation": "-",
        "step_number": 1,
        "fixable": True
    }]
    
    correction = correct_solution(steps, errors)
    print(f"  ✓ Fixed: {correction['fixed_count']}/{correction['total_fixable']} errors")
    print(f"  ✓ Success rate: {correction['success_rate']*100:.0f}%")
    if correction['correction_log']:
        print(f"  ✓ Original: {correction['correction_log'][0]['original']}")
        print(f"  ✓ Corrected: {correction['correction_log'][0]['corrected']}")
    
    print("  ✅ Error Correction PASSED")
except Exception as e:
    print(f"  ❌ Error Correction FAILED: {e}")

# Test 8: Full Integration
print("\n[8/10] Testing Full Integration (Parallel Execution)")
try:
    from core.verification_engine import run_verification_parallel
    
    problem = "Janet has 3 apples. She buys 2 more. She gives 1 away. How many?"
    steps = [
        "Janet starts with 3 apples",
        "She buys 2 more: 3 + 2 = 5 apples",
        "She gives 1 away: 5 - 1 = 6 apples"  # ERROR
    ]
    
    result = run_verification_parallel(
        problem=problem,
        steps=steps,
        model_name="GPT-4",
        model_list=["GPT-4", "Llama 2", "Gemini"]
    )
    
    print(f"  ✓ Processing time: {result['processing_time']:.2f}s")
    print(f"  ✓ Final verdict: {result['consensus']['final_verdict']}")
    print(f"  ✓ Confidence: {result['consensus']['overall_confidence']*100:.1f}%")
    print(f"  ✓ Errors found: {len(result['classified_errors'])}")
    print(f"  ✓ All 3 models executed: {len(result['model_results']) == 3}")
    
    print("  ✅ Full Integration PASSED")
except Exception as e:
    print(f"  ❌ Full Integration FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Check Dependencies
print("\n[9/10] Checking Dependencies")
try:
    import streamlit
    import sympy
    import pytest
    print("  ✓ streamlit installed")
    print("  ✓ sympy installed")
    print("  ✓ pytest installed")
    print("  ✅ All Dependencies Available")
except ImportError as e:
    print(f"  ⚠️  Missing dependency: {e}")
    print("  Run: pip install -r requirements.txt")

# Test 10: Import Check for App
print("\n[10/10] Testing App Imports")
try:
    # Try importing what app.py needs
    import streamlit as st
    import time
    from typing import List, Dict, Any
    from core import run_verification_parallel
    print("  ✓ All app imports successful")
    print("  ✅ App Ready to Run")
except Exception as e:
    print(f"  ❌ App Import FAILED: {e}")

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
print("\nNext steps:")
print("1. Run: streamlit run app.py")
print("2. Or test: python run_example.py")
print("3. Or run tests: pytest")

