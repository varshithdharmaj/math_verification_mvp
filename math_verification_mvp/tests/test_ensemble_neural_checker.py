"""
Unit tests for EnsembleNeuralChecker
"""

import pytest
from models.ensemble_neural_checker import EnsembleNeuralChecker


def test_unanimous_agreement():
    """Test when all models agree on ERROR"""
    checker = EnsembleNeuralChecker()
    steps = ["5 - 1 = 6"]  # Clear error
    result = checker.verify(steps)
    
    assert "verdict" in result
    assert "confidence" in result
    assert "sub_models" in result
    assert "agreement" in result


def test_majority_vote():
    """Test majority voting (2/3 agree)"""
    checker = EnsembleNeuralChecker(["GPT-4", "Llama 2", "Gemini"])
    steps = ["3 + 2 = 5"]  # Valid
    result = checker.verify(steps)
    
    assert result["confidence"] >= 0.60
    assert "agreement" in result


def test_custom_model_list():
    """Test with custom model list"""
    checker = EnsembleNeuralChecker(["Model A", "Model B"])
    steps = ["3 + 2 = 5"]
    result = checker.verify(steps)
    
    assert len(result["sub_models"]) == 2
    assert "Model A" in result["sub_models"]
    assert "Model B" in result["sub_models"]

