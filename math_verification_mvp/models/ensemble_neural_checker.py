"""
Model 3: Ensemble Neural Checker
Simulates multiple LLMs voting on solution validity
Uses voting logic to determine overall verdict
"""

from typing import List, Dict, Any
from models.llm_logical_checker import LLMLogicalChecker


class EnsembleNeuralChecker:
    """Simulates ensemble of multiple LLMs voting on solution validity."""
    
    def __init__(self, model_names: List[str] = None):
        if model_names is None:
            model_names = ["GPT-4", "Llama 2", "Gemini"]
        self.model_names = model_names
        self.model_name = f"🤖 Ensemble ({len(model_names)} models)"
        self.model_type = "ensemble"
    
    def verify(self, steps: List[str]) -> Dict[str, Any]:
        """
        Simulate multiple LLMs voting on solution validity.
        
        Args:
            steps: List of solution step strings
            
        Returns:
            Dictionary with verdict, confidence, and sub-model results
        """
        # Simulate each sub-model's verdict
        sub_model_results = {}
        
        for model_name in self.model_names:
            # Use LLMLogicalChecker to simulate each model's response
            checker = LLMLogicalChecker(model_name=model_name)
            result = checker.verify(steps)
            sub_model_results[model_name] = result["verdict"]
        
        # Count votes
        error_votes = sum(1 for verdict in sub_model_results.values() if verdict == "ERROR")
        valid_votes = sum(1 for verdict in sub_model_results.values() if verdict == "VALID")
        total_votes = len(sub_model_results)
        
        # Determine overall verdict using voting logic
        if error_votes > valid_votes:
            overall_verdict = "ERROR"
        else:
            overall_verdict = "VALID"
        
        # Calculate confidence based on agreement ratio
        agreement_ratio = max(error_votes, valid_votes) / total_votes
        
        if agreement_ratio == 1.0:  # All agree (3/3)
            confidence = 0.90
        elif agreement_ratio >= 2/3:  # Majority agree (2/3)
            confidence = 0.80
        else:  # Mixed (1/3)
            confidence = 0.60
        
        # Collect all errors from sub-models (simplified - in production would aggregate)
        all_errors = []
        for model_name, verdict in sub_model_results.items():
            if verdict == "ERROR":
                # Simulate error detection for each model
                checker = LLMLogicalChecker(model_name=model_name)
                result = checker.verify(steps)
                if result["errors"]:
                    all_errors.extend(result["errors"])
        
        # Format agreement string
        if agreement_ratio == 1.0:
            agreement = f"{total_votes}/{total_votes} agree"
        else:
            agreement = f"{max(error_votes, valid_votes)}/{total_votes} agree"
        
        return {
            "model": self.model_type,
            "model_name": self.model_name,
            "verdict": overall_verdict,
            "confidence": confidence,
            "sub_models": sub_model_results,
            "agreement": agreement,
            "errors": all_errors
        }

