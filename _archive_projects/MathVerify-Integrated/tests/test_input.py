"""Unit tests for input processing module."""

import pytest
from src.input_module.processor import InputProcessor


@pytest.fixture
def processor():
    """Create an InputProcessor instance for testing."""
    return InputProcessor()


class TestInputProcessor:
    """Test cases for InputProcessor."""
    
    def test_normalize_expression_basic(self, processor):
        """Test basic expression normalization."""
        result = processor.normalize_expression("2x + 3")
        assert "2*x" in result or "2 * x" in result
    
    def test_normalize_expression_exponent(self, processor):
        """Test exponent normalization."""
        result = processor.normalize_expression("x^2")
        assert "x**2" in result
    
    def test_normalize_expression_fraction(self, processor):
        """Test fraction handling."""
        result = processor.normalize_expression("1/2")
        assert "1/2" in result
    
    def test_extract_problem_components(self, processor):
        """Test problem component extraction."""
        problem = "Solve for x: 2x + 3 = 7"
        components = processor.extract_problem_components(problem)
        assert "question" in components
        assert "unknowns" in components
        assert "x" in components["unknowns"]
    
    def test_validate_input_valid(self, processor):
        """Test validation of valid input."""
        assert processor.validate_input("Solve: 2 + 2 = ?") is True
        assert processor.validate_input("x + 5 = 10") is True
    
    def test_validate_input_invalid(self, processor):
        """Test validation of invalid input."""
        assert processor.validate_input("") is False
        assert processor.validate_input("   ") is False
        assert processor.validate_input("abc xyz") is False

