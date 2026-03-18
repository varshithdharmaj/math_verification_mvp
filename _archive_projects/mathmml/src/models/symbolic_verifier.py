"""SymbolicVerifier using SymPy for arithmetic/algebra verification."""

import re
import sympy as sp
from typing import Dict, List, Optional, Tuple
from sympy import sympify, simplify, N


class SymbolicVerifier:
    """Verifies mathematical steps using symbolic computation (SymPy)."""
    
    def __init__(self, confidence_correct: float = 0.92, confidence_error: float = 0.95):
        """Initialize verifier.
        
        Args:
            confidence_correct: Confidence when no error found
            confidence_error: Confidence when error detected
        """
        self.confidence_correct = confidence_correct
        self.confidence_error = confidence_error
    
    def extract_expression(self, text: str) -> Optional[Tuple[str, Optional[float]]]:
        """Extract mathematical expression and claimed result from text.
        
        Args:
            text: Step text
            
        Returns:
            Tuple of (expression_string, claimed_result) or None
        """
        # Remove calculator annotations
        text = re.sub(r'<<[^>]+>>', '', text)
        
        # Try to find equation pattern: "x = y" or "calculation = result"
        eq_match = re.search(r'([^=]+)\s*=\s*([^=]+)', text)
        if eq_match:
            left = eq_match.group(1).strip()
            right = eq_match.group(2).strip()
            
            # Extract expression from left side (the calculation)
            # Keep numbers, operators, and parentheses
            expr = re.sub(r'[^\d+\-*/().\s]', '', left)
            expr = expr.strip()
            
            # Extract claimed result from right side
            claimed_result = None
            result_match = re.search(r'([\d.]+)', right)
            if result_match:
                try:
                    claimed_result = float(result_match.group(1))
                except:
                    pass
            
            if expr:
                return (expr, claimed_result)
        
        # Try to find standalone expression
        expr_match = re.search(r'([\d+\-*/().]+)', text)
        if expr_match:
            expr = expr_match.group(1).strip()
            return (expr, None)
        
        return None
    
    def evaluate_expression(self, expr: str) -> Optional[float]:
        """Safely evaluate expression.
        
        Args:
            expr: Expression string
            
        Returns:
            Numeric result or None if invalid
        """
        try:
            # Replace common patterns
            expr = expr.replace('×', '*').replace('÷', '/')
            # Parse and evaluate
            sympy_expr = sympify(expr)
            result = float(N(simplify(sympy_expr)))
            return result
        except:
            return None
    
    def check_arithmetic(self, step: str, prev_steps: List[str] = None) -> Dict:
        """Check arithmetic correctness of a step.
        
        Args:
            step: Current step text
            prev_steps: Previous steps for context
            
        Returns:
            Dict with 'verdict', 'confidence', 'error_type', 'details'
        """
        extracted = self.extract_expression(step)
        
        if not extracted:
            # Can't verify, return neutral
            return {
                'verdict': 'UNKNOWN',
                'confidence': 0.5,
                'error_type': None,
                'details': 'Could not extract expression'
            }
        
        expr, claimed_result = extracted
        
        # Try to evaluate
        calculated_result = self.evaluate_expression(expr)
        
        if calculated_result is None:
            # Invalid expression
            return {
                'verdict': 'ERROR',
                'confidence': self.confidence_error,
                'error_type': 'arithmetic_error',
                'details': f'Invalid expression: {expr}'
            }
        
        # If step claims a specific result, verify it matches
        if claimed_result is not None:
            if abs(calculated_result - claimed_result) > 1e-6:
                return {
                    'verdict': 'ERROR',
                    'confidence': self.confidence_error,
                    'error_type': 'arithmetic_error',
                    'details': f'Calculation error: {expr} = {calculated_result}, but step claims {claimed_result}'
                }
            else:
                return {
                    'verdict': 'VALID',
                    'confidence': self.confidence_correct,
                    'error_type': None,
                    'details': f'Expression {expr} = {calculated_result} is correct'
                }
        else:
            # No claimed result, just verify expression is valid
            return {
                'verdict': 'VALID',
                'confidence': self.confidence_correct * 0.9,  # Slightly lower confidence
                'error_type': None,
                'details': f'Expression {expr} = {calculated_result} evaluates correctly'
            }
    
    def verify(self, step: str, problem: str = "", prev_steps: List[str] = None) -> Dict:
        """Main verification method.
        
        Args:
            step: Step text to verify
            problem: Problem statement (for context)
            prev_steps: Previous steps
            
        Returns:
            Verification result dict
        """
        if prev_steps is None:
            prev_steps = []
        
        return self.check_arithmetic(step, prev_steps)

