"""
Quick test script to verify all components are working.

Run this to test each module independently.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_input_processor():
    """Test InputProcessor."""
    print("Testing InputProcessor...")
    try:
        from src.input_module.processor import InputProcessor
        processor = InputProcessor()
        
        # Test normalization
        result = processor.normalize_expression("2x + 3")
        assert "*" in result, "Normalization failed"
        
        # Test validation
        assert processor.validate_input("2 + 2 = 4"), "Validation failed"
        
        print("  ✓ InputProcessor working")
        return True
    except Exception as e:
        print(f"  ✗ InputProcessor failed: {e}")
        return False

def test_symbolic_verifier():
    """Test SymbolicVerifier."""
    print("Testing SymbolicVerifier...")
    try:
        from src.verification_module.verifier import SymbolicVerifier
        verifier = SymbolicVerifier()
        
        # Test equation verification
        result = verifier.verify_equation("2 + 2 = 4")
        assert result["valid"] is True, "Verification failed"
        
        result = verifier.verify_equation("2 + 2 = 5")
        assert result["valid"] is False, "Should detect incorrect equation"
        
        print("  ✓ SymbolicVerifier working")
        return True
    except Exception as e:
        print(f"  ✗ SymbolicVerifier failed: {e}")
        return False

def test_error_taxonomy():
    """Test ErrorTaxonomy."""
    print("Testing ErrorTaxonomy...")
    try:
        from src.verification_module.error_taxonomy import ErrorTaxonomy
        taxonomy = ErrorTaxonomy()
        
        # Test error classification
        error_type = taxonomy.classify_error(
            "2 + 2 = 5",
            {"valid": False, "error": "incorrect calculation"}
        )
        assert error_type == "CALCULATION_ERROR", "Classification failed"
        
        # Test description
        desc = taxonomy.get_error_description("CALCULATION_ERROR")
        assert len(desc) > 0, "Description missing"
        
        print("  ✓ ErrorTaxonomy working")
        return True
    except Exception as e:
        print(f"  ✗ ErrorTaxonomy failed: {e}")
        return False

def test_reasoning_engine():
    """Test ReasoningEngine (may fail if model not available)."""
    print("Testing ReasoningEngine...")
    try:
        from src.reasoning_module.engine import ReasoningEngine
        # Use small model for testing
        engine = ReasoningEngine(model_name="gpt2", device="cpu")
        
        # Test structure (may not generate good results, but should not crash)
        result = engine.generate_solution("2 + 2 = ?")
        assert "steps" in result, "Missing steps"
        assert "final_answer" in result, "Missing answer"
        assert "confidence" in result, "Missing confidence"
        
        print("  ✓ ReasoningEngine working (structure OK)")
        return True
    except Exception as e:
        print(f"  ✗ ReasoningEngine failed: {e}")
        print("    (This is OK if model is not available)")
        return False

def test_pipeline():
    """Test MathVerifyPipeline."""
    print("Testing MathVerifyPipeline...")
    try:
        from src.pipeline import MathVerifyPipeline
        # Use small model for testing
        pipeline = MathVerifyPipeline(model_name="gpt2", device="cpu")
        
        # Test processing
        result = pipeline.process_problem("2 + 2 = ?")
        assert "problem" in result, "Missing problem"
        assert "solution" in result, "Missing solution"
        assert "errors" in result, "Missing errors"
        
        print("  ✓ MathVerifyPipeline working")
        return True
    except Exception as e:
        print(f"  ✗ MathVerifyPipeline failed: {e}")
        return False

def main():
    """Run all component tests."""
    print("=" * 60)
    print("MathVerify-Integrated: Component Tests")
    print("=" * 60)
    print()
    
    results = []
    results.append(("InputProcessor", test_input_processor()))
    results.append(("SymbolicVerifier", test_symbolic_verifier()))
    results.append(("ErrorTaxonomy", test_error_taxonomy()))
    results.append(("ReasoningEngine", test_reasoning_engine()))
    results.append(("MathVerifyPipeline", test_pipeline()))
    
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} components working")
    
    if passed == total:
        print("✅ All components working!")
    else:
        print("⚠️  Some components need attention")

if __name__ == "__main__":
    main()

