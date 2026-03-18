"""
Symbolic verification module for mathematical reasoning.

This module uses SymPy to verify mathematical expressions, equations, and reasoning steps.
"""

import re
from typing import Dict, List, Optional, Any
import sympy as sp
from sympy import sympify, simplify, Eq, solve, Symbol
from sympy.parsing.sympy_parser import parse_expr


class SymbolicVerifier:
    """
    Symbolic verifier for mathematical expressions using SymPy.
    
    Verifies equations, inequalities, and algebraic expressions
    to ensure mathematical correctness.
    
    Example:
        >>> verifier = SymbolicVerifier()
        >>> result = verifier.verify_equation("2 + 2 = 4")
        >>> print(result['valid'])
        True
        >>> result = verifier.verify_equation("x + 5 = 10, x = 5")
        >>> print(result['valid'])
        True
    """
    
    def __init__(self):
        """Initialize the SymbolicVerifier."""
        pass
    
    def verify_equation(self, equation: str) -> Dict[str, Any]:
        """
        Verify if an equation is mathematically correct.
        
        Parses equation (e.g., "2 + 2 = 4") and checks if both sides are equal.
        
        Args:
            equation: Equation string to verify
            
        Returns:
            Dictionary with keys:
                - valid: Boolean indicating if equation is correct
                - error: Error message if invalid, None otherwise
                - details: Additional verification details
        """
        try:
            # Clean and normalize the equation
            equation = equation.strip()
            
            # Handle multiple equations separated by commas or semicolons
            if ',' in equation or ';' in equation:
                # Split and verify each
                parts = re.split(r'[,;]', equation)
                results = []
                for part in parts:
                    result = self.verify_equation(part.strip())
                    results.append(result)
                
                # All must be valid
                all_valid = all(r['valid'] for r in results)
                return {
                    "valid": all_valid,
                    "error": None if all_valid else "One or more equations are invalid",
                    "details": {"sub_equations": results}
                }
            
            # Check if it's an equation (has =)
            if '=' not in equation:
                return {
                    "valid": False,
                    "error": "Not an equation (missing '=')",
                    "details": {}
                }
            
            # Split into left and right sides
            parts = equation.split('=', 1)
            if len(parts) != 2:
                return {
                    "valid": False,
                    "error": "Invalid equation format",
                    "details": {}
                }
            
            left_str, right_str = parts[0].strip(), parts[1].strip()
            
            # Try to parse and verify
            try:
                # Normalize expressions
                left_expr = self._parse_expression(left_str)
                right_expr = self._parse_expression(right_str)
                
                # Check if they're equal
                diff = simplify(left_expr - right_expr)
                
                # If difference is 0, they're equal
                if diff == 0:
                    return {
                        "valid": True,
                        "error": None,
                        "details": {
                            "left": str(left_expr),
                            "right": str(right_expr),
                            "difference": "0"
                        }
                    }
                else:
                    return {
                        "valid": False,
                        "error": f"Equation is incorrect: {left_str} ≠ {right_str}",
                        "details": {
                            "left": str(left_expr),
                            "right": str(right_expr),
                            "difference": str(diff)
                        }
                    }
                    
            except Exception as e:
                # Try numerical evaluation if symbolic fails
                try:
                    left_val = self._evaluate_numerical(left_str)
                    right_val = self._evaluate_numerical(right_str)
                    
                    if abs(left_val - right_val) < 1e-10:  # Floating point tolerance
                        return {
                            "valid": True,
                            "error": None,
                            "details": {
                                "left_value": left_val,
                                "right_value": right_val,
                                "method": "numerical"
                            }
                        }
                    else:
                        return {
                            "valid": False,
                            "error": f"Numerical values don't match: {left_val} ≠ {right_val}",
                            "details": {
                                "left_value": left_val,
                                "right_value": right_val,
                                "method": "numerical"
                            }
                        }
                except Exception as e2:
                    return {
                        "valid": False,
                        "error": f"Could not verify equation: {str(e2)}",
                        "details": {"parse_error": str(e)}
                    }
                    
        except Exception as e:
            return {
                "valid": False,
                "error": f"Verification failed: {str(e)}",
                "details": {}
            }
    
    def verify_step(
        self,
        step: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Verify a single reasoning step.
        
        Args:
            step: Reasoning step text to verify
            context: Optional context dictionary with previous steps
            
        Returns:
            Verification result dictionary
        """
        if context is None:
            context = {}
        
        # Extract equations from the step
        equations = self.extract_expressions(step)
        
        if not equations:
            # No mathematical expression found
            return {
                "valid": True,  # Non-mathematical step might be valid
                "error": None,
                "details": {"note": "No mathematical expression found"}
            }
        
        # Verify each equation
        results = []
        for eq in equations:
            result = self.verify_equation(eq)
            results.append(result)
        
        # Step is valid if all equations are valid
        all_valid = all(r['valid'] for r in results)
        
        return {
            "valid": all_valid,
            "error": None if all_valid else "One or more equations in step are invalid",
            "details": {
                "equations_checked": len(equations),
                "equation_results": results
            }
        }
    
    def extract_expressions(self, text: str) -> List[str]:
        """
        Extract all mathematical expressions from text.
        
        Args:
            text: Text to extract expressions from
            
        Returns:
            List of expression strings
        """
        expressions = []
        
        # Pattern 1: Equations with = (e.g., "x = 5", "2 + 2 = 4")
        equation_pattern = r'[^=]+=[^=]+'
        equations = re.findall(equation_pattern, text)
        expressions.extend(equations)
        
        # Pattern 2: Expressions in parentheses (e.g., "(2x + 3)")
        paren_pattern = r'\(([^)]+)\)'
        paren_exprs = re.findall(paren_pattern, text)
        expressions.extend(paren_exprs)
        
        # Pattern 3: Standalone expressions with operators
        expr_pattern = r'\b\d+\s*[+\-*/]\s*\d+\b'
        simple_exprs = re.findall(expr_pattern, text)
        expressions.extend(simple_exprs)
        
        # Clean and deduplicate
        expressions = [e.strip() for e in expressions if e.strip()]
        expressions = list(set(expressions))
        
        return expressions
    
    def symbolic_simplify(self, expr: str) -> str:
        """
        Simplify a mathematical expression using SymPy.
        
        Args:
            expr: Expression string to simplify
            
        Returns:
            Simplified expression string
        """
        try:
            parsed = self._parse_expression(expr)
            simplified = simplify(parsed)
            return str(simplified)
        except Exception as e:
            # Return original if simplification fails
            return expr
    
    def _parse_expression(self, expr_str: str) -> sp.Expr:
        """
        Parse a mathematical expression string into SymPy expression.
        
        Args:
            expr_str: Expression string
            
        Returns:
            SymPy expression object
        """
        # Clean the expression
        expr_str = expr_str.strip()
        
        # Replace common notations
        expr_str = expr_str.replace('^', '**')  # Exponent
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)  # Implicit multiplication
        expr_str = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr_str)  # Variable followed by number
        
        try:
            # Try parsing with sympify
            return sympify(expr_str, evaluate=False)
        except:
            # Fallback to parse_expr
            try:
                return parse_expr(expr_str, evaluate=False)
            except Exception as e:
                raise ValueError(f"Could not parse expression '{expr_str}': {str(e)}")
    
    def _evaluate_numerical(self, expr_str: str) -> float:
        """
        Evaluate a numerical expression.
        
        Args:
            expr_str: Expression string
            
        Returns:
            Numerical result
        """
        # Clean and prepare for evaluation
        expr_str = expr_str.strip()
        expr_str = expr_str.replace('^', '**')
        
        # Replace implicit multiplication
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
        
        # Evaluate safely
        try:
            # Only allow safe operations
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expr_str.replace('**', '')):
                return eval(expr_str)
            else:
                raise ValueError("Expression contains unsafe characters")
        except Exception as e:
            raise ValueError(f"Could not evaluate numerically: {str(e)}")

