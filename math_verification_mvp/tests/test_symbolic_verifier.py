"""
Unit tests for SymbolicVerifier
"""

import pytest
from models.symbolic_verifier import SymbolicVerifier


def test_addition_valid():
    """Test addition: 3 + 2 = 5 → VALID"""
    verifier = SymbolicVerifier()
    steps = ["3 + 2 = 5"]
    result = verifier.verify(steps)
    
    assert result["verdict"] == "VALID"
    assert result["confidence"] == 0.95
    assert len(result["errors"]) == 0


def test_subtraction_error():
    """Test subtraction error: 5 - 1 = 6 → ERROR"""
    verifier = SymbolicVerifier()
    steps = ["5 - 1 = 6"]
    result = verifier.verify(steps)
    
    assert result["verdict"] == "ERROR"
    assert result["confidence"] == 0.90
    assert len(result["errors"]) > 0
    # The correct format includes decimal points from float conversion
    assert "4.0" in result["errors"][0]["correct"]
    assert "5" in result["errors"][0]["correct"]
    assert "1" in result["errors"][0]["correct"]


def test_multiplication_valid():
    """Test multiplication: 3 * 4 = 12 → VALID"""
    verifier = SymbolicVerifier()
    steps = ["3 * 4 = 12"]
    result = verifier.verify(steps)
    
    assert result["verdict"] == "VALID"
    assert len(result["errors"]) == 0


def test_division_valid():
    """Test division: 60 / 2 = 30 → VALID"""
    verifier = SymbolicVerifier()
    steps = ["60 / 2 = 30"]
    result = verifier.verify(steps)
    
    assert result["verdict"] == "VALID"
    assert len(result["errors"]) == 0


def test_multiple_steps():
    """Test multiple steps with mixed valid and invalid"""
    verifier = SymbolicVerifier()
    steps = [
        "3 + 2 = 5",
        "5 - 1 = 6",  # Error
        "4 * 2 = 8"
    ]
    result = verifier.verify(steps)
    
    assert result["verdict"] == "ERROR"
    assert len(result["errors"]) > 0

