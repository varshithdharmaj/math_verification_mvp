"""Unit tests for reasoning module."""

import pytest
from src.reasoning_module.engine import ReasoningEngine


@pytest.fixture
def engine():
    """Create a ReasoningEngine instance for testing."""
    # Use a small model or mock for testing
    return ReasoningEngine(model_name="gpt2", device="cpu")  # Small model for testing


class TestReasoningEngine:
    """Test cases for ReasoningEngine."""
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert engine.max_steps > 0
    
    def test_generate_solution_structure(self, engine):
        """Test that generate_solution returns correct structure."""
        result = engine.generate_solution("2 + 2 = ?")
        assert isinstance(result, dict)
        assert "steps" in result
        assert "final_answer" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_generate_single_step(self, engine):
        """Test single step generation."""
        previous_steps = [{"content": "Step 1: Analyze problem"}]
        result = engine.generate_single_step("2 + 2 = ?", previous_steps)
        assert isinstance(result, dict)
        assert "content" in result
        assert "rationale" in result

