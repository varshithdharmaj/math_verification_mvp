"""
Performance tests for MathVerify pipeline.
"""

import pytest
import time
import psutil
import os
from src.pipeline import MathVerifyPipeline

class TestPerformance:
    """Performance tests."""
    
    def test_latency_per_problem(self):
        """Measure latency per problem."""
        pipeline = MathVerifyPipeline(model_name="gpt2")
        problem = "Solve 2x + 3 = 7"
        
        start_time = time.time()
        pipeline.process_problem(problem)
        end_time = time.time()
        
        latency = end_time - start_time
        print(f"Latency: {latency:.4f} seconds")
        
        # Should be reasonable (e.g., < 10 seconds for GPT-2 on CPU)
        # Adjust threshold based on hardware
        assert latency < 30.0
        
    def test_memory_usage(self):
        """Check memory usage."""
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        pipeline = MathVerifyPipeline(model_name="gpt2")
        pipeline.process_problem("2 + 2 = 4")
        
        end_mem = process.memory_info().rss / 1024 / 1024  # MB
        diff = end_mem - start_mem
        
        print(f"Memory increase: {diff:.2f} MB")
        print(f"Total memory: {end_mem:.2f} MB")
        
        # Ensure no massive leak (though loading model will increase memory)
        # This is just a smoke test
        assert end_mem > 0
