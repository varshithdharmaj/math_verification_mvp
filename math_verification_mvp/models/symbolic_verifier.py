"""
Model 1: Symbolic Verifier
Uses SymPy for mathematical calculation verification
Detects arithmetic errors in solution steps
"""

import re
from sympy import sympify, simplify, N
from typing import List, Dict, Any


class SymbolicVerifier:
    """Verifies mathematical calculations using symbolic computation."""
    
    def __init__(self):
        self.model_name = "🔢 Symbolic (SymPy)"
        self.model_type = "symbolic"
    
    def verify(self, steps: List[str]) -> Dict[str, Any]:
        """
        Verify arithmetic operations in solution steps.
        
        Args:
            steps: List of solution step strings
            
        Returns:
            Dictionary with verdict, confidence, and errors
        """
        errors = []
        
        for step_num, step in enumerate(steps, start=1):
            # Detect arithmetic patterns: "(\d+)\s*[+\-*/]\s*(\d+)\s*=\s*(\d+)"
            patterns = [
                r'(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # Addition
                r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # Subtraction
                r'(\d+\.?\d*)\s*\*\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # Multiplication
                r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # Division
            ]
            
            operations = ['+', '-', '*', '/']
            
            for pattern, op in zip(patterns, operations):
                matches = re.finditer(pattern, step)
                for match in matches:
                    operand1 = float(match.group(1))
                    operand2 = float(match.group(2))
                    given_result = float(match.group(3))
                    
                    # Calculate correct result using SymPy
                    try:
                        expr = sympify(f"{operand1} {op} {operand2}")
                        correct_result = float(N(simplify(expr)))
                        
                        # Check if there's an error
                        if abs(given_result - correct_result) > 1e-9:  # Floating point tolerance
                            error = {
                                "step_number": step_num,
                                "type": "calculation_error",
                                "operation": op,
                                "found": f"{operand1} {op} {operand2} = {given_result}",
                                "correct": f"{operand1} {op} {operand2} = {correct_result}",
                                "severity": "HIGH",
                                "fixable": True
                            }
                            errors.append(error)
                    except Exception as e:
                        # If calculation fails, skip this match
                        continue
        
        # Determine verdict and confidence
        if errors:
            verdict = "ERROR"
            confidence = 0.90
        else:
            verdict = "VALID"
            confidence = 0.95
        
        return {
            "model": self.model_type,
            "model_name": self.model_name,
            "verdict": verdict,
            "confidence": confidence,
            "errors": errors
        }

