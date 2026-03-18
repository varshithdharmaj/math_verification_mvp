"""
Integration tests for MathVerify pipeline.
"""

import pytest
import json
from pathlib import Path
from src.pipeline import MathVerifyPipeline

@pytest.fixture
def pipeline():
    """Create a MathVerifyPipeline instance for testing."""
    # Use a dummy model or mock for faster testing if possible
    # For now, we assume the pipeline handles model loading
    return MathVerifyPipeline(model_name="gpt2")

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    def test_pipeline_on_sample_problems(self, pipeline):
        """Test pipeline on sample problems."""
        # Load sample problems
        sample_file = Path("data/test_samples/samples.json")
        if not sample_file.exists():
            pytest.skip("Sample data not found")
            
        with open(sample_file, 'r', encoding='utf-8') as f:
            problems = json.load(f)
            
        # Test on first 2 problems to save time
        for problem in problems[:2]:
            result = pipeline.process_problem(problem["question"])
            
            assert isinstance(result, dict)
            assert "solution" in result
            assert "verification" in result
            assert "errors" in result
            assert "final_answer" in result
            
            # Check structure of solution
            assert "steps" in result["solution"]
            assert len(result["solution"]["steps"]) > 0
            
            # Check verification
            assert isinstance(result["verification"], list)
            
    def test_error_correction_loop(self, pipeline):
        """Test that the pipeline attempts error correction."""
        # This is harder to test deterministically with a real model
        # We might need to mock the ReasoningEngine to produce an error
        pass

if __name__ == "__main__":
    # Manual run
    pipeline = MathVerifyPipeline(model_name="gpt2")
    print("Running integration test...")
    
    sample_file = Path("data/test_samples/samples.json")
    if sample_file.exists():
        with open(sample_file, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        
        problem = problems[0]
        print(f"Problem: {problem['question']}")
        result = pipeline.process_problem(problem["question"])
        print("Result:", json.dumps(result, indent=2))
    else:
        print("Sample file not found.")
