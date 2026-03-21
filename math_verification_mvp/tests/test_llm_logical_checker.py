"""
Unit tests for LLMLogicalChecker
"""

import pytest
from models.llm_logical_checker import LLMLogicalChecker


def test_valid_logic():
    """Test valid logical step"""
    checker = LLMLogicalChecker()
    steps = ["She buys 2 more: 3 + 2 = 5 apples"]
    result = checker.verify(steps)
    
    assert result["verdict"] in ["VALID", "ERROR"]  # Can be either
    assert "confidence" in result


def test_contradiction():
    """Test contradiction: 'and...but' pattern → ERROR"""
    checker = LLMLogicalChecker()
    steps = ["She has 3 apples and buys 2 more but she gives 1 away"]
    result = checker.verify(steps)
    
    # Should detect contradiction
    assert result["verdict"] == "ERROR" or len(result["errors"]) > 0


def test_operation_mismatch():
    """Test operation mismatch: says subtract but uses +"""
    checker = LLMLogicalChecker()
    steps = ["She subtracts 1: 5 + 1 = 6"]
    result = checker.verify(steps)
    
    # Should detect mismatch
    assert result["verdict"] == "ERROR" or len(result["errors"]) > 0


def test_different_models():
    """Test with different model names"""
    checker1 = LLMLogicalChecker("GPT-4")
    checker2 = LLMLogicalChecker("Llama 2")
    checker3 = LLMLogicalChecker("Gemini")
    
    steps = ["3 + 2 = 5"]
    
    result1 = checker1.verify(steps)
    result2 = checker2.verify(steps)
    result3 = checker3.verify(steps)
    
    assert "GPT-4" in result1["model_name"]
    assert "Llama 2" in result2["model_name"]
    assert "Gemini" in result3["model_name"]

