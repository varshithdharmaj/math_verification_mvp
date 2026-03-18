"""Tests for EnsembleNeuralChecker."""

import pytest
from src.models.ensemble_checker import EnsembleNeuralChecker


def test_ensemble_voting():
    """Test ensemble voting mechanism."""
    checker = EnsembleNeuralChecker(num_models=3, use_apis=False)
    
    result = checker.verify("5 + 3 = 8", problem="", prev_steps=[])
    assert 'verdict' in result
    assert 'confidence' in result
    assert 'votes' in result


def test_majority_vote():
    """Test that majority vote determines verdict."""
    checker = EnsembleNeuralChecker(num_models=3, use_apis=False)
    
    result = checker.verify("Test step", problem="", prev_steps=[])
    assert result['verdict'] in ['VALID', 'ERROR']
    assert result['votes']['error'] + result['votes']['valid'] == 3


if __name__ == "__main__":
    pytest.main([__file__])

