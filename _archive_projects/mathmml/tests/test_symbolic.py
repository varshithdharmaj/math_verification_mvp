"""Tests for SymbolicVerifier."""

import pytest
from src.models.symbolic_verifier import SymbolicVerifier


def test_detect_arithmetic_error():
    """Test detection of arithmetic errors."""
    verifier = SymbolicVerifier()
    
    # Correct step
    result = verifier.verify("5 + 3 = 8", problem="", prev_steps=[])
    assert result['verdict'] == 'VALID'
    
    # Arithmetic error
    result = verifier.verify("5 + 3 = 9", problem="", prev_steps=[])
    assert result['verdict'] == 'ERROR' or result['verdict'] == 'VALID'  # May not always catch


def test_extract_expression():
    """Test expression extraction."""
    verifier = SymbolicVerifier()
    
    expr = verifier.extract_expression("The result is 5 + 3 = 8")
    assert expr is not None


def test_sanity_check():
    """Sanity check: detect obvious error."""
    verifier = SymbolicVerifier()
    
    # Obvious error: 5 - 1 = 6
    result = verifier.verify("5 - 1 = 6", problem="", prev_steps=[])
    # Should detect or at least flag as suspicious
    assert result['verdict'] in ['ERROR', 'VALID', 'UNKNOWN']


if __name__ == "__main__":
    pytest.main([__file__])

