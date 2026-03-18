"""
Error Taxonomy for Mathematical Reasoning Verification.

This module provides comprehensive error classification for mathematical reasoning,
including a novel visual misinterpretation category for future extension.
"""

from typing import Dict, List, Optional
from enum import Enum


class ErrorType(Enum):
    """Enumeration of error types in mathematical reasoning."""
    CALCULATION_ERROR = "CALCULATION_ERROR"
    LOGICAL_ERROR = "LOGICAL_ERROR"
    NOTATION_ERROR = "NOTATION_ERROR"
    REASONING_GAP = "REASONING_GAP"
    VISUAL_MISINTERPRETATION = "VISUAL_MISINTERPRETATION"


class ErrorTaxonomy:
    """
    Comprehensive error taxonomy for mathematical reasoning verification.
    
    This is a key novel contribution that classifies errors into distinct categories,
    enabling targeted debugging and error correction.
    
    Error Categories:
        - CALCULATION_ERROR: Arithmetic mistakes (e.g., 2+2=5)
        - LOGICAL_ERROR: Invalid inference or reasoning (e.g., circular logic)
        - NOTATION_ERROR: Malformed expressions (e.g., "2x+=")
        - REASONING_GAP: Missing justification or skipped steps
        - VISUAL_MISINTERPRETATION: Diagram or visual element misreading (placeholder)
    
    Example:
        >>> taxonomy = ErrorTaxonomy()
        >>> result = {"valid": False, "error": "2 + 2 = 5"}
        >>> error_type = taxonomy.classify_error("2 + 2 = 5", result)
        >>> print(error_type)
        'CALCULATION_ERROR'
        >>> description = taxonomy.get_error_description(error_type)
        >>> print(description)
        'Arithmetic or computational mistake in numerical calculation'
    """
    
    def __init__(self):
        """Initialize the ErrorTaxonomy."""
        self.error_descriptions = {
            ErrorType.CALCULATION_ERROR.value: "Arithmetic or computational mistake in numerical calculation",
            ErrorType.LOGICAL_ERROR.value: "Invalid logical inference or reasoning step",
            ErrorType.NOTATION_ERROR.value: "Malformed mathematical expression or notation",
            ErrorType.REASONING_GAP.value: "Missing justification or skipped reasoning step",
            ErrorType.VISUAL_MISINTERPRETATION.value: "Misinterpretation of visual elements (diagrams, graphs, etc.)"
        }
        
        self.error_corrections = {
            ErrorType.CALCULATION_ERROR.value: "Recalculate the numerical operation carefully",
            ErrorType.LOGICAL_ERROR.value: "Review the logical flow and ensure each step follows from the previous",
            ErrorType.NOTATION_ERROR.value: "Check mathematical notation and ensure proper syntax",
            ErrorType.REASONING_GAP.value: "Add missing steps or justification to complete the reasoning chain",
            ErrorType.VISUAL_MISINTERPRETATION.value: "Re-examine visual elements and verify correct interpretation"
        }
    
    def classify_error(
        self,
        step: str,
        verification_result: Dict
    ) -> str:
        """
        Classify the type of error in a failed verification.
        
        Analyzes the step content and verification result to determine
        the most likely error category.
        
        Args:
            step: The reasoning step that failed verification
            verification_result: Dictionary from verification with 'valid' and 'error' keys
            
        Returns:
            Error type as string (one of ErrorType enum values)
        """
        if verification_result.get("valid", True):
            return "NO_ERROR"
        
        error_message = verification_result.get("error", "").lower()
        step_lower = step.lower()
        
        # Check for calculation errors (arithmetic mistakes)
        calculation_indicators = [
            "incorrect calculation",
            "wrong arithmetic",
            "math error",
            "computation error",
            "does not equal",
            "not equal",
        ]
        if any(indicator in error_message for indicator in calculation_indicators):
            # Also check if it's a simple arithmetic mistake
            if self._is_arithmetic_mistake(step):
                return ErrorType.CALCULATION_ERROR.value
        
        # Check for notation errors (malformed expressions)
        notation_indicators = [
            "invalid expression",
            "syntax error",
            "malformed",
            "parse error",
            "invalid notation",
        ]
        if any(indicator in error_message for indicator in notation_indicators):
            return ErrorType.NOTATION_ERROR.value
        
        # Check for logical errors (invalid inference)
        logical_indicators = [
            "invalid inference",
            "does not follow",
            "logical error",
            "circular reasoning",
            "contradiction",
        ]
        if any(indicator in error_message for indicator in logical_indicators):
            return ErrorType.LOGICAL_ERROR.value
        
        # Check for reasoning gaps (missing steps)
        gap_indicators = [
            "missing step",
            "unjustified",
            "no justification",
            "skipped",
            "gap in reasoning",
        ]
        if any(indicator in error_message for indicator in gap_indicators):
            return ErrorType.REASONING_GAP.value
        
        # Check for visual misinterpretation (if step mentions visual elements)
        visual_indicators = [
            "diagram",
            "graph",
            "figure",
            "chart",
            "visual",
            "image",
        ]
        if any(indicator in step_lower for indicator in visual_indicators):
            if "misinterpret" in error_message or "incorrect" in error_message:
                return ErrorType.VISUAL_MISINTERPRETATION.value
        
        # Default: if it's a calculation that's wrong, assume calculation error
        if self._is_arithmetic_mistake(step):
            return ErrorType.CALCULATION_ERROR.value
        
        # Default fallback
        return ErrorType.LOGICAL_ERROR.value
    
    def _is_arithmetic_mistake(self, step: str) -> bool:
        """
        Check if step contains an arithmetic operation that might be wrong.
        
        Args:
            step: Reasoning step text
            
        Returns:
            True if step appears to contain arithmetic
        """
        # Look for arithmetic patterns: numbers and operators
        import re
        has_arithmetic = bool(re.search(r'\d+\s*[+\-*/=]\s*\d+', step))
        has_equals = '=' in step
        return has_arithmetic and has_equals
    
    def get_error_description(self, error_type: str) -> str:
        """
        Get human-readable description of an error type.
        
        Args:
            error_type: Error type string (from ErrorType enum)
            
        Returns:
            Human-readable description
        """
        return self.error_descriptions.get(
            error_type,
            "Unknown error type"
        )
    
    def suggest_correction(
        self,
        error_type: str,
        step: str
    ) -> str:
        """
        Suggest how to fix an error based on its type.
        
        Args:
            error_type: Error type string
            step: The step that contains the error
            
        Returns:
            Suggestion string for fixing the error
        """
        base_suggestion = self.error_corrections.get(
            error_type,
            "Review the step and correct the identified issue"
        )
        
        # Add step-specific suggestions if possible
        if error_type == ErrorType.CALCULATION_ERROR.value:
            # Try to identify the operation
            if '+' in step:
                base_suggestion += ". Double-check addition."
            elif '-' in step:
                base_suggestion += ". Double-check subtraction."
            elif '*' in step or '×' in step:
                base_suggestion += ". Double-check multiplication."
            elif '/' in step or '÷' in step:
                base_suggestion += ". Double-check division."
        
        return base_suggestion
    
    def generate_report(self, errors: List[Dict]) -> Dict[str, any]:
        """
        Generate comprehensive error report from list of errors.
        
        Args:
            errors: List of error dictionaries, each with 'type' key
            
        Returns:
            Dictionary with:
                - total_errors: Total count
                - by_type: Count by error type
                - percentage: Percentage by error type
        """
        if not errors:
            return {
                "total_errors": 0,
                "by_type": {},
                "percentage": {}
            }
        
        total = len(errors)
        by_type = {}
        
        # Count errors by type
        for error in errors:
            error_type = error.get("type", "UNKNOWN")
            by_type[error_type] = by_type.get(error_type, 0) + 1
        
        # Calculate percentages
        percentage = {
            error_type: round((count / total) * 100, 2)
            for error_type, count in by_type.items()
        }
        
        return {
            "total_errors": total,
            "by_type": by_type,
            "percentage": percentage
        }
    
    def get_all_error_types(self) -> List[str]:
        """
        Get list of all error type strings.
        
        Returns:
            List of error type strings
        """
        return [error_type.value for error_type in ErrorType]

