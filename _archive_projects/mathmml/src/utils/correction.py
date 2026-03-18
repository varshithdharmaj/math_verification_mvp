"""Auto-correction engine for fixable errors."""

import re
from typing import Dict, Optional, Tuple
from src.utils.error_taxonomy import is_auto_fixable, get_error_info


class CorrectionEngine:
    """Engine for automatically correcting fixable errors."""
    
    def __init__(self):
        """Initialize correction engine."""
        pass
    
    def correct_arithmetic(self, step: str) -> Optional[str]:
        """Correct arithmetic errors by recalculating.
        
        PRD Req 4.3.2: Auto-correction for arithmetic (92% success rate target).
        
        Args:
            step: Step text with arithmetic error
            
        Returns:
            Corrected step or None
        """
        # Extract expression using improved pattern
        # Try to find "expression = result" pattern
        expr_match = re.search(r'([\d+\-*/().\s]+)\s*=\s*([\d.]+)', step)
        if not expr_match:
            return None
        
        expression = expr_match.group(1).strip()
        claimed_result = expr_match.group(2)
        
        try:
            # Clean expression
            expr_clean = expression.replace('×', '*').replace('÷', '/')
            # Remove extra spaces
            expr_clean = re.sub(r'\s+', '', expr_clean)
            
            # Evaluate expression safely
            result = eval(expr_clean)
            
            # Format result appropriately
            if isinstance(result, float):
                if result.is_integer():
                    result_str = str(int(result))
                else:
                    result_str = f"{result:.2f}".rstrip('0').rstrip('.')
            else:
                result_str = str(result)
            
            # Replace claimed result with correct result
            # Use regex to replace only the number after =, not other occurrences
            corrected = re.sub(
                r'=\s*' + re.escape(claimed_result) + r'(?:\s|$|\.|,|;|:)',
                f'= {result_str}',
                step,
                count=1
            )
            
            return corrected
        except Exception as e:
            # If eval fails, return None
            return None
    
    def correct_sign(self, step: str) -> Optional[str]:
        """Correct sign errors.
        
        Args:
            step: Step text with sign error
            
        Returns:
            Corrected step or None
        """
        # Simple heuristic: if step has + but should be -, or vice versa
        # This is context-dependent, so we'll just flip the first occurrence
        if '+' in step:
            corrected = step.replace('+', '-', 1)
            return corrected
        elif '-' in step and not step.startswith('-'):
            corrected = step.replace('-', '+', 1)
            return corrected
        
        return None
    
    def correct_operation(self, step: str, problem: str) -> Optional[str]:
        """Correct operation mismatch errors.
        
        Args:
            step: Step text
            problem: Problem statement
            
        Returns:
            Corrected step or None
        """
        # This requires understanding problem intent, which is complex
        # For now, return None (manual review)
        return None
    
    def correct(self, error_type: str, step: str, problem: str = "") -> Dict:
        """Attempt to correct an error.
        
        Args:
            error_type: Type of error
            step: Step text
            problem: Problem statement
            
        Returns:
            Dict with 'success', 'corrected_step', 'confidence', 'requires_review'
        """
        if not is_auto_fixable(error_type):
            return {
                'success': False,
                'corrected_step': None,
                'confidence': 0.0,
                'requires_review': True,
                'reason': f'{error_type} requires manual review'
            }
        
        corrected = None
        confidence = 0.5
        
        if error_type == "arithmetic_error":
            corrected = self.correct_arithmetic(step)
            confidence = 0.8 if corrected else 0.0
        elif error_type == "sign_error":
            corrected = self.correct_sign(step)
            confidence = 0.7 if corrected else 0.0
        elif error_type == "operation_mismatch":
            corrected = self.correct_operation(step, problem)
            confidence = 0.6 if corrected else 0.0
        elif error_type in ["notation_error", "unit_error", "order_ops_error"]:
            # These are harder to auto-correct
            return {
                'success': False,
                'corrected_step': None,
                'confidence': 0.0,
                'requires_review': True,
                'reason': f'{error_type} correction not yet implemented'
            }
        
        if corrected:
            return {
                'success': True,
                'corrected_step': corrected,
                'confidence': confidence,
                'requires_review': confidence < 0.8,
                'reason': 'Auto-correction applied'
            }
        else:
            return {
                'success': False,
                'corrected_step': None,
                'confidence': 0.0,
                'requires_review': True,
                'reason': 'Could not generate correction'
            }

