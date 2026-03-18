"""
MathVerse Evaluator Module
Wrapper for MathVerse benchmark evaluation
"""

import sys
import os
import json
from typing import List, Dict, Any, Optional

# Add MathVerse to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MathVerse'))

try:
    from evaluation.extract_answer_s1 import extract_answer
    from evaluation.score_answer_s2 import score_answer
    from evaluation.utils import load_json, save_json
except ImportError:
    extract_answer = None
    score_answer = None
    load_json = None
    save_json = None


class MathVerseEvaluator:
    """
    Evaluator for MathVerse benchmark.
    Handles visual math problem evaluation with CoT scoring.
    """
    
    def __init__(self, data_path: str = None, api_key: str = None):
        """
        Initialize MathVerse evaluator.
        
        Args:
            data_path: Path to MathVerse data directory
            api_key: OpenAI API key for answer extraction/scoring (optional)
        """
        self.data_path = data_path or os.path.join(
            os.path.dirname(__file__), '..', 'MathVerse', 'data'
        )
        self.api_key = api_key
        self.test_file = os.path.join(self.data_path, 'testmini.json')
        self.test_text_only_file = os.path.join(self.data_path, 'testmini_text_only.json')
    
    def load_test_data(self, text_only: bool = False, limit: int = None) -> List[Dict[str, Any]]:
        """
        Load test data from MathVerse dataset.
        
        Args:
            text_only: Whether to load text-only version
            limit: Maximum number of examples to load (None for all)
            
        Returns:
            List of test examples
        """
        test_file = self.test_text_only_file if text_only else self.test_file
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"MathVerse test file not found: {test_file}")
        
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data[:limit] if limit else data
    
    def extract_answers(self, 
                        model_output_file: str,
                        save_file: str,
                        cache: bool = True,
                        trunk_response: int = 30) -> List[Dict[str, Any]]:
        """
        Extract answers from model outputs (Step 1 of MathVerse evaluation).
        
        Args:
            model_output_file: Path to file with model outputs
            save_file: Path to save extracted answers
            cache: Whether to use caching
            trunk_response: Length to truncate responses
            
        Returns:
            List of extracted answers
        """
        if extract_answer is None:
            raise ImportError("MathVerse extraction module not available")
        
        if not self.api_key:
            raise ValueError("API key required for answer extraction")
        
        # This would call the actual extraction function
        # For now, return placeholder
        return []
    
    def score_answers(self,
                     extraction_file: str,
                     save_file: str,
                     cache: bool = True,
                     quick_match: bool = False) -> Dict[str, Any]:
        """
        Score extracted answers (Step 2 of MathVerse evaluation).
        
        Args:
            extraction_file: Path to file with extracted answers
            save_file: Path to save scores
            cache: Whether to use caching
            quick_match: Whether to use quick string matching
            
        Returns:
            Dictionary with scoring results
        """
        if score_answer is None:
            raise ImportError("MathVerse scoring module not available")
        
        if not self.api_key and not quick_match:
            raise ValueError("API key required for answer scoring (or use quick_match=True)")
        
        # This would call the actual scoring function
        # For now, return placeholder
        return {}
    
    def format_output(self, 
                     problem_index: int,
                     problem_version: str,
                     response: str) -> Dict[str, Any]:
        """
        Format a single model output for evaluation.
        
        Args:
            problem_index: Index of the problem
            problem_version: Version of the problem (e.g., 'text_dominant')
            response: Model response text
            
        Returns:
            Formatted output dictionary
        """
        return {
            'problem_index': problem_index,
            'problem_version': problem_version,
            'response': response
        }

