"""
Automatic Error Correction
Applies corrections to fixable errors in solution steps
Tracks correction success rates
"""

import re
from typing import List, Dict, Any
from sympy import sympify, simplify, N


def correct_solution(steps: List[str], errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Automatically correct fixable errors in solution steps.
    
    Args:
        steps: Original solution steps list
        errors: List of error dictionaries
        
    Returns:
        Dictionary with corrected steps, correction log, and success rate
    """
    corrected_steps = steps.copy()
    correction_log = []
    manual_review_needed = []
    fixed_count = 0
    
    for error in errors:
        step_number = error.get("step_number", 0) - 1  # Convert to 0-based index
        if step_number < 0 or step_number >= len(corrected_steps):
            continue
        
        error_type = error.get("type", "")
        fixable = error.get("fixable", False)
        
        if not fixable:
            manual_review_needed.append(error)
            continue
        
        # Attempt correction based on error type
        if error_type == "calculation_error":
            success = _correct_arithmetic_error(corrected_steps, step_number, error)
            if success:
                fixed_count += 1
                correction_log.append({
                    "step": step_number + 1,
                    "type": "arithmetic",
                    "original": steps[step_number],
                    "corrected": corrected_steps[step_number],
                    "reason": "Arithmetic calculation corrected"
                })
            else:
                manual_review_needed.append(error)
        
        elif error_type == "operation_mismatch":
            success = _correct_operation_mismatch(corrected_steps, step_number, error)
            if success:
                fixed_count += 1
                correction_log.append({
                    "step": step_number + 1,
                    "type": "operation_mismatch",
                    "original": steps[step_number],
                    "corrected": corrected_steps[step_number],
                    "reason": "Operation mismatch corrected"
                })
            else:
                manual_review_needed.append(error)
        
        else:
            # Other error types need manual review
            manual_review_needed.append(error)
    
    # Calculate success rate
    total_fixable = len([e for e in errors if e.get("fixable", False)])
    if total_fixable > 0:
        success_rate = fixed_count / total_fixable
    else:
        success_rate = 0.0
    
    return {
        "corrected_steps": corrected_steps,
        "correction_log": correction_log,
        "success_rate": success_rate,
        "manual_review_needed": manual_review_needed,
        "fixed_count": fixed_count,
        "total_fixable": total_fixable
    }


def _correct_arithmetic_error(steps: List[str], step_index: int, error: Dict[str, Any]) -> bool:
    """Correct arithmetic calculation error."""
    try:
        found = error.get("found", "")
        correct = error.get("correct", "")
        
        # Extract the incorrect result from found
        found_nums = re.findall(r'\d+\.?\d*', found)
        correct_nums = re.findall(r'\d+\.?\d*', correct)
        
        if not found_nums or not correct_nums:
            return False
        
        incorrect_result = found_nums[-1]
        correct_result = correct_nums[-1]
        
        # Replace incorrect result with correct result in the step
        step = steps[step_index]
        # Replace the last occurrence of the incorrect result
        corrected_step = step.replace(incorrect_result, correct_result, 1)
        
        # If that didn't work, try more sophisticated replacement
        if corrected_step == step:
            # Try replacing the full expression
            corrected_step = step.replace(found, correct)
        
        steps[step_index] = corrected_step
        return True
    except Exception as e:
        return False


def _correct_operation_mismatch(steps: List[str], step_index: int, error: Dict[str, Any]) -> bool:
    """Correct operation mismatch error."""
    try:
        description = error.get("description", "")
        step = steps[step_index]
        
        # Extract operation from description
        # This is simplified - in production would use more sophisticated NLP
        if "subtract" in description.lower() and "+" in step:
            # Replace + with -
            corrected_step = step.replace("+", "-", 1)
            steps[step_index] = corrected_step
            return True
        elif "add" in description.lower() and "-" in step:
            # Replace - with +
            corrected_step = step.replace("-", "+", 1)
            steps[step_index] = corrected_step
            return True
        
        return False
    except Exception as e:
        return False

