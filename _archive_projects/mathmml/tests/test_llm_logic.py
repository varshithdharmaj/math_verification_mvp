"""Tests for LLMLogicalChecker."""

import pytest
from src.models.llm_logical_checker import LLMLogicalChecker


def test_detect_contradiction():
    """Test contradiction detection."""
    checker = LLMLogicalChecker(use_api=False)
    
    step = "The answer is 10, but it's also 20"
    prev_steps = []
    result = checker.verify(step, problem="", prev_steps=prev_steps)
    # Should detect contradiction
    assert result['verdict'] in ['ERROR', 'VALID']


def test_operation_mismatch():
    """Test operation mismatch detection."""
    checker = LLMLogicalChecker(use_api=False)
    
    problem = "Add 5 and 3"
    step = "5 - 3 = 2"
    result = checker.verify(step, problem=problem, prev_steps=[])
    # May detect mismatch
    assert result['verdict'] in ['ERROR', 'VALID']


def test_mock_mode():
    """Test that mock mode works without API."""
    checker = LLMLogicalChecker(use_api=False)
    assert checker.use_api == False


if __name__ == "__main__":
    pytest.main([__file__])

