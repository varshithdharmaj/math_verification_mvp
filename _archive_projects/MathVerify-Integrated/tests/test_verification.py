"""
Unit tests for verification module.

Tests for SymbolicVerifier and ErrorTaxonomy classes.
"""

import pytest
from src.verification_module.verifier import SymbolicVerifier
from src.verification_module.error_taxonomy import ErrorTaxonomy


@pytest.fixture
def verifier():
    """Create a SymbolicVerifier instance for testing."""
    return SymbolicVerifier()


@pytest.fixture
def taxonomy():
    """Create an ErrorTaxonomy instance for testing."""
    return ErrorTaxonomy()


class TestSymbolicVerifier:
    """Test cases for SymbolicVerifier."""
    
    def test_verify_correct_equation(self, verifier):
        """Test verification of a correct equation."""
        result = verifier.verify_equation("2 + 2 = 4")
        assert result["valid"] is True
        assert result["error"] is None
    
    def test_verify_incorrect_equation(self, verifier):
        """Test verification of an incorrect equation."""
        result = verifier.verify_equation("2 + 2 = 5")
        assert result["valid"] is False
        assert result["error"] is not None
        assert "incorrect" in result["error"].lower() or "not equal" in result["error"].lower()
    
    def test_verify_algebraic_expression(self, verifier):
        """Test verification of algebraic expression."""
        result = verifier.verify_equation("x + 5 = 10, x = 5")
        # Should verify that x = 5 satisfies x + 5 = 10
        # This is a compound equation, so we check if it processes correctly
        assert isinstance(result, dict)
        assert "valid" in result
    
    def test_handle_invalid_expression(self, verifier):
        """Test graceful handling of invalid expressions."""
        result = verifier.verify_equation("abc xyz @#$")
        # Should handle gracefully without crashing
        assert isinstance(result, dict)
        assert "valid" in result or "error" in result
    
    def test_extract_expressions(self, verifier):
        """Test expression extraction from text."""
        text = "We have x = 5 and y = 10, so x + y = 15"
        expressions = verifier.extract_expressions(text)
        assert len(expressions) > 0
        assert any("x = 5" in expr or "x=5" in expr.replace(" ", "") for expr in expressions)
    
    def test_symbolic_simplify(self, verifier):
        """Test symbolic simplification."""
        result = verifier.symbolic_simplify("2*x + 3*x")
        # Should simplify to 5*x or similar
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_verify_step_with_context(self, verifier):
        """Test step verification with context."""
        step = "Therefore, x = 5"
        context = {"previous_steps": []}
        result = verifier.verify_step(step, context)
        assert isinstance(result, dict)
        assert "valid" in result


class TestErrorTaxonomy:
    """Test cases for ErrorTaxonomy."""
    
    def test_classify_calculation_error(self, taxonomy):
        """Test classification of calculation errors."""
        step = "2 + 2 = 5"
        verification_result = {"valid": False, "error": "incorrect calculation"}
        error_type = taxonomy.classify_error(step, verification_result)
        assert error_type == "CALCULATION_ERROR"
    
    def test_classify_notation_error(self, taxonomy):
        """Test classification of notation errors."""
        step = "2x+="
        verification_result = {"valid": False, "error": "invalid expression"}
        error_type = taxonomy.classify_error(step, verification_result)
        assert error_type == "NOTATION_ERROR"
    
    def test_classify_logical_error(self, taxonomy):
        """Test classification of logical errors."""
        step = "If A then B, therefore C"
        verification_result = {"valid": False, "error": "invalid inference"}
        error_type = taxonomy.classify_error(step, verification_result)
        assert error_type == "LOGICAL_ERROR"
    
    def test_classify_reasoning_gap(self, taxonomy):
        """Test classification of reasoning gaps."""
        step = "Therefore, x = 5"
        verification_result = {"valid": False, "error": "missing step"}
        error_type = taxonomy.classify_error(step, verification_result)
        assert error_type == "REASONING_GAP"
    
    def test_get_error_description(self, taxonomy):
        """Test getting error descriptions."""
        description = taxonomy.get_error_description("CALCULATION_ERROR")
        assert isinstance(description, str)
        assert len(description) > 0
        assert "calculation" in description.lower() or "arithmetic" in description.lower()
    
    def test_suggest_correction(self, taxonomy):
        """Test error correction suggestions."""
        suggestion = taxonomy.suggest_correction("CALCULATION_ERROR", "2 + 2 = 5")
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0
    
    def test_generate_report(self, taxonomy):
        """Test error report generation."""
        errors = [
            {"type": "CALCULATION_ERROR"},
            {"type": "CALCULATION_ERROR"},
            {"type": "LOGICAL_ERROR"},
        ]
        report = taxonomy.generate_report(errors)
        assert report["total_errors"] == 3
        assert report["by_type"]["CALCULATION_ERROR"] == 2
        assert report["by_type"]["LOGICAL_ERROR"] == 1
        assert "percentage" in report
    
    def test_generate_report_empty(self, taxonomy):
        """Test report generation with no errors."""
        report = taxonomy.generate_report([])
        assert report["total_errors"] == 0
        assert report["by_type"] == {}
    
    def test_get_all_error_types(self, taxonomy):
        """Test getting all error types."""
        error_types = taxonomy.get_all_error_types()
        assert isinstance(error_types, list)
        assert len(error_types) > 0
        assert "CALCULATION_ERROR" in error_types


class TestIntegration:
    """Integration tests for verification workflow."""
    
    def test_verify_and_classify_workflow(self, verifier, taxonomy):
        """Test complete verify and classify workflow."""
        step = "2 + 2 = 5"
        verification_result = verifier.verify_equation(step)
        error_type = taxonomy.classify_error(step, verification_result)
        
        assert verification_result["valid"] is False
        assert error_type == "CALCULATION_ERROR"
        
        description = taxonomy.get_error_description(error_type)
        correction = taxonomy.suggest_correction(error_type, step)
        
        assert len(description) > 0
        assert len(correction) > 0

