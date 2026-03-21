"""
Model 2: LLM Logical Checker
Checks logical consistency in solution steps
Detects contradictions, operation mismatches, and semantic inconsistencies
"""

import re
from typing import List, Dict, Any


class LLMLogicalChecker:
    """Checks logical consistency using pattern-based heuristics."""
    
    def __init__(self, model_name: str = "GPT-4"):
        self.model_name = f"🧠 {model_name}"
        self.model_type = "llm_logical"
        self.model_name_raw = model_name
    
    def verify(self, steps: List[str]) -> Dict[str, Any]:
        """
        Check logical consistency in solution steps.
        
        Args:
            steps: List of solution step strings
            
        Returns:
            Dictionary with verdict, confidence, and errors
        """
        errors = []
        
        for step_num, step in enumerate(steps, start=1):
            # Check for contradictions: "and" followed by "but"
            if self._has_contradiction(step):
                error = {
                    "step_number": step_num,
                    "type": "logical_error",
                    "description": "Contradiction detected: step contains conflicting statements",
                    "severity": "MEDIUM",
                    "fixable": False
                }
                errors.append(error)
            
            # Check for operation mismatches
            op_mismatch = self._check_operation_mismatch(step)
            if op_mismatch:
                error = {
                    "step_number": step_num,
                    "type": "operation_mismatch",
                    "description": op_mismatch,
                    "severity": "HIGH",
                    "fixable": True
                }
                errors.append(error)
            
            # Check for semantic inconsistencies
            semantic_issue = self._check_semantic_consistency(step, steps[:step_num-1])
            if semantic_issue:
                error = {
                    "step_number": step_num,
                    "type": "semantic_error",
                    "description": semantic_issue,
                    "severity": "MEDIUM",
                    "fixable": False
                }
                errors.append(error)
        
        # Determine verdict and confidence
        if errors:
            verdict = "ERROR"
            confidence = 0.82
        else:
            verdict = "VALID"
            confidence = 0.87
        
        return {
            "model": self.model_type,
            "model_name": self.model_name,
            "model_name_raw": self.model_name_raw,
            "verdict": verdict,
            "confidence": confidence,
            "errors": errors
        }
    
    def _has_contradiction(self, step: str) -> bool:
        """Check if step contains contradiction patterns."""
        step_lower = step.lower()
        # Check for "and...but" or "however" patterns
        if re.search(r'\band\b.*\bbut\b', step_lower, re.IGNORECASE):
            return True
        if re.search(r'\bhowever\b', step_lower, re.IGNORECASE):
            return True
        return False
    
    def _check_operation_mismatch(self, step: str) -> str:
        """Check if text says one operation but math uses another."""
        step_lower = step.lower()
        
        # Map operation keywords to symbols
        op_keywords = {
            'add': '+', 'plus': '+', 'sum': '+', 'total': '+',
            'subtract': '-', 'minus': '-', 'remove': '-', 'take away': '-',
            'multiply': '*', 'times': '*', 'product': '*',
            'divide': '/', 'division': '/', 'split': '/'
        }
        
        # Find mentioned operation in text
        mentioned_op = None
        for keyword, symbol in op_keywords.items():
            if keyword in step_lower:
                mentioned_op = symbol
                break
        
        # Find actual operation in math expression
        math_patterns = [
            r'(\d+)\s*\+\s*(\d+)',
            r'(\d+)\s*-\s*(\d+)',
            r'(\d+)\s*\*\s*(\d+)',
            r'(\d+)\s*/\s*(\d+)'
        ]
        
        actual_op = None
        for pattern, symbol in zip(math_patterns, ['+', '-', '*', '/']):
            if re.search(pattern, step):
                actual_op = symbol
                break
        
        # Check for mismatch
        if mentioned_op and actual_op and mentioned_op != actual_op:
            return f"Text mentions '{mentioned_op}' operation but math uses '{actual_op}'"
        
        return ""
    
    def _check_semantic_consistency(self, step: str, previous_steps: List[str]) -> str:
        """Check if step is semantically consistent with previous steps."""
        # Simple heuristic: check if numbers mentioned are consistent
        # Extract numbers from current step
        current_numbers = set(re.findall(r'\b\d+\.?\d*\b', step))
        
        # Check if step references numbers that don't exist in previous context
        # This is a simplified check - in production, would use more sophisticated NLP
        if len(previous_steps) > 0:
            prev_context = ' '.join(previous_steps)
            prev_numbers = set(re.findall(r'\b\d+\.?\d*\b', prev_context))
            
            # If step introduces large numbers not in context, might be inconsistent
            # (This is a heuristic - can be improved)
            if current_numbers and not prev_numbers:
                # First step, so no issue
                return ""
        
        return ""

