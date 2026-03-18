"""Natural language generation for error explanations."""

from typing import Dict, Optional
from src.utils.error_taxonomy import get_error_info, ERROR_TAXONOMY


EXPLANATION_TEMPLATES = {
    "arithmetic_error": "The calculation contains an arithmetic error. {details}",
    "logical_error": "There is a logical inconsistency in the reasoning: {details}",
    "operation_mismatch": "The wrong operation was used. {details}",
    "conceptual_error": "This step shows a conceptual misunderstanding: {details}",
    "notation_error": "There is an error in mathematical notation: {details}",
    "sign_error": "The sign (positive/negative) is incorrect: {details}",
    "unit_error": "There is a unit mismatch or missing unit: {details}",
    "order_ops_error": "The order of operations is incorrect: {details}",
    "semantic_error": "This step is semantically inconsistent with the problem: {details}"
}


def generate_explanation(
    error_type: str,
    details: str = "",
    step: str = "",
    problem: str = "",
    found_value: str = "",
    correct_value: str = ""
) -> str:
    """Generate natural language explanation for an error.
    
    PRD Req 4.3.1: Template-based NLG, 2-3 sentences, student-friendly.
    
    Args:
        error_type: Type of error detected
        details: Additional details from verifier
        step: The step text
        problem: Problem statement
        found_value: Incorrect value found
        correct_value: Correct value
        
    Returns:
        Natural language explanation (2-3 sentences, student-friendly)
    """
    if error_type == "correct" or not error_type:
        return "No errors detected. The step appears to be correct."
    
    # PRD-specified format: What went wrong + why + how to fix
    if error_type == "arithmetic_error" and found_value and correct_value:
        explanation = (
            f"Error in this step: Arithmetic Calculation Error. "
            f"You wrote {found_value}, but the correct answer is {correct_value}. "
            f"Double-check your calculation to find where the mistake occurred."
        )
    elif error_type == "sign_error":
        explanation = (
            f"Error in this step: Sign Error. "
            f"The sign (positive/negative) is incorrect. "
            f"Check whether you should be adding or subtracting, and verify the signs of your numbers."
        )
    elif error_type == "operation_mismatch":
        explanation = (
            f"Error in this step: Operation Mismatch. "
            f"The operation you used doesn't match what the problem is asking for. "
            f"Re-read the problem to identify which operation (addition, subtraction, multiplication, or division) you should use."
        )
    elif error_type == "logical_error":
        explanation = (
            f"Error in this step: Logical Error. "
            f"There's a logical inconsistency in your reasoning. "
            f"Review your steps to find where the logic breaks down."
        )
    else:
        # Generic template
        template = EXPLANATION_TEMPLATES.get(error_type, "An error was detected: {details}")
        
        # Fill in details
        if not details:
            error_info = get_error_info(error_type)
            details = error_info.get("description", "")
        
        explanation = template.format(details=details)
        
        # Add how to fix
        hint = generate_correction_hint(error_type, step, problem)
        if hint:
            explanation += f" {hint}"
    
    # Ensure 2-3 sentences (PRD requirement)
    sentences = explanation.split('. ')
    if len(sentences) < 2:
        explanation += " Please review this step carefully."
    
    return explanation


def generate_correction_hint(
    error_type: str,
    step: str,
    problem: str
) -> Optional[str]:
    """Generate correction hint for auto-fixable errors.
    
    Args:
        error_type: Type of error
        step: Step text
        problem: Problem statement
        
    Returns:
        Correction hint or None
    """
    hints = {
        "arithmetic_error": "Recalculate the arithmetic operation carefully.",
        "sign_error": "Check the sign of the numbers and operations.",
        "operation_mismatch": "Verify that the correct operation is being used based on the problem requirements.",
        "notation_error": "Check the mathematical notation for correctness.",
        "unit_error": "Ensure units are consistent throughout the calculation.",
        "order_ops_error": "Apply the correct order of operations (PEMDAS/BODMAS)."
    }
    
    return hints.get(error_type)

