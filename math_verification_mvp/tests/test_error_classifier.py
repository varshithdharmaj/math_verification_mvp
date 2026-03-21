"""
Unit tests for Error Classification
"""

import pytest
from utils.error_classifier import classify_error


def test_arithmetic_error_classification():
    """Test arithmetic error classification"""
    error = {
        "type": "calculation_error",
        "found": "5 - 1 = 6",
        "correct": "5 - 1 = 4",
        "operation": "-",
        "step_number": 1
    }
    
    classified = classify_error(error)
    
    assert classified["category"] == "Arithmetic Error"
    assert classified["severity"] == "HIGH"
    assert classified["fixable"] == True
    assert classified["fixability_score"] > 0.90


def test_logical_error_classification():
    """Test logical error classification"""
    error = {
        "type": "logical_error",
        "description": "Contradiction detected",
        "step_number": 1
    }
    
    classified = classify_error(error)
    
    assert classified["category"] == "Logical Error"
    assert classified["severity"] == "MEDIUM"
    assert classified["fixability_score"] == 0.60


def test_operation_mismatch_classification():
    """Test operation mismatch classification"""
    error = {
        "type": "operation_mismatch",
        "description": "Text mentions subtract but math uses +",
        "step_number": 1
    }
    
    classified = classify_error(error)
    
    assert classified["category"] == "Operation Mismatch"
    assert classified["severity"] == "HIGH"
    assert classified["fixable"] == True

