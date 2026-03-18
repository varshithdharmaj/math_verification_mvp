"""
MATH-V Evaluator Module
Wrapper for MATH-V benchmark evaluation
"""

import sys
import os
import json
from typing import List, Dict, Any

# Add MATH-V to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MATH-V'))

try:
    from evaluation.evaluate import evaluate
    from evaluation.utils import load_jsonl, save_jsonl
except ImportError:
    evaluate = None
    load_jsonl = None
    save_jsonl = None


class MathVEvaluator:
    """
    Evaluator for MATH-V benchmark.
    Handles multimodal mathematical reasoning evaluation.
    """
    
    def __init__(self, data_path: str = None, api_key: str = None):
        """
        Initialize MATH-V evaluator.
        
        Args:
            data_path: Path to MATH-V data directory
            api_key: OpenAI API key for model evaluation (optional)
        """
        self.data_path = data_path or os.path.join(
            os.path.dirname(__file__), '..', 'MATH-V', 'data'
        )
        self.api_key = api_key
        self.test_file = os.path.join(self.data_path, 'test.jsonl')
    
    def load_test_data(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Load test data from MATH-V dataset.
        
        Args:
            limit: Maximum number of examples to load (None for all)
            
        Returns:
            List of test examples
        """
        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"MATH-V test file not found: {self.test_file}")
        
        if load_jsonl is None:
            # Fallback implementation
            data = []
            with open(self.test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
                        if limit and len(data) >= limit:
                            break
            return data
        
        data = load_jsonl(self.test_file)
        return data[:limit] if limit else data
    
    def evaluate_model_outputs(self, output_file: str, regen_answer: bool = False) -> Dict[str, Any]:
        """
        Evaluate model outputs using MATH-V evaluation script.
        
        Args:
            output_file: Path to JSONL file with model outputs
            regen_answer: Whether to regenerate answers from responses
            
        Returns:
            Dictionary with evaluation metrics
        """
        if evaluate is None:
            raise ImportError("MATH-V evaluation module not available")
        
        # Run evaluation
        evaluate(output_file, regen_answer=regen_answer)
        
        # Load results
        results = load_jsonl(output_file)
        
        # Calculate metrics
        total = len(results)
        correct = sum(1 for r in results if r.get('correct', False))
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'results': results
        }
    
    def format_output(self, example_id: str, response: str, model_answer: str = None) -> Dict[str, Any]:
        """
        Format a single model output for evaluation.
        
        Args:
            example_id: ID of the example
            response: Model response text
            model_answer: Extracted model answer (optional)
            
        Returns:
            Formatted output dictionary
        """
        output = {
            'id': example_id,
            'response': response
        }
        if model_answer:
            output['model_answer'] = model_answer
        return output

