"""Tests for MLStepClassifier."""

import pytest
import torch
from src.models.ml_step_classifier import MLStepClassifier, MLStepClassifierWrapper


def test_model_forward():
    """Test model forward pass."""
    model = MLStepClassifier("roberta-base", num_labels=10)
    
    # Mock input
    input_ids = torch.randint(0, 1000, (1, 128))
    attention_mask = torch.ones(1, 128)
    
    output = model(input_ids, attention_mask)
    assert 'logits' in output
    assert output['logits'].shape == (1, 10)


def test_wrapper_infer():
    """Test wrapper inference."""
    wrapper = MLStepClassifierWrapper(
        model_path=None,  # Use untrained model
        model_name="roberta-base",
        device="cpu"
    )
    
    result = wrapper.infer(
        problem="Test problem",
        prev_steps="",
        current_step="Test step"
    )
    
    assert 'label' in result
    assert 'confidence' in result
    assert 'prob_vector' in result


def test_label_mapping():
    """Test label mapping."""
    wrapper = MLStepClassifierWrapper(model_path=None, device="cpu")
    assert len(wrapper.id_to_label) > 0


if __name__ == "__main__":
    pytest.main([__file__])

