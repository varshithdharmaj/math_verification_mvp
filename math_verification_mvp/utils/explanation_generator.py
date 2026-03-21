"""
Explanation Generation
Generates natural language explanations for errors found
Provides educational context and learning suggestions
"""

from typing import Dict, Any


def generate_explanation(error: Dict[str, Any]) -> str:
    """
    Generate human-readable explanation for an error.
    
    Args:
        error: Error dictionary with type, found, correct, step_number, etc.
        
    Returns:
        Natural language explanation string (2-3 sentences)
    """
    error_type = error.get("type", "unknown")
    found = error.get("found", "")
    correct = error.get("correct", "")
    operation = error.get("operation", "")
    step_number = error.get("step_number", 0)
    
    # Extract numbers from found and correct
    import re
    found_nums = re.findall(r'\d+\.?\d*', found)
    correct_nums = re.findall(r'\d+\.?\d*', correct)
    
    # Generate explanation based on error type
    if error_type == "calculation_error":
        explanation = _explain_arithmetic_error(found, correct, operation, found_nums, correct_nums)
    elif error_type == "logical_error":
        explanation = _explain_logical_error(error, step_number)
    elif error_type == "operation_mismatch":
        explanation = _explain_operation_mismatch(error, found, correct)
    elif error_type == "semantic_error":
        explanation = _explain_semantic_error(error, found, correct)
    else:
        explanation = _explain_generic_error(found, correct)
    
    # Add learning context
    learning_tip = _get_learning_tip(error_type, operation)
    if learning_tip:
        explanation += f" {learning_tip}"
    
    return explanation


def _explain_arithmetic_error(found: str, correct: str, operation: str, found_nums: list, correct_nums: list) -> str:
    """Explain arithmetic calculation errors."""
    if not found_nums or not correct_nums:
        return f"You wrote '{found}', but the correct answer is '{correct}'."
    
    try:
        found_result = float(found_nums[-1]) if found_nums else 0
        correct_result = float(correct_nums[-1]) if correct_nums else 0
        
        if len(found_nums) >= 2:
            operand1 = float(found_nums[0])
            operand2 = float(found_nums[1])
            
            op_names = {
                '+': 'add',
                '-': 'subtract',
                '*': 'multiply',
                '/': 'divide'
            }
            op_name = op_names.get(operation, 'calculate')
            
            explanation = f"You wrote {found_result}, but {operand1} {operation} {operand2} actually equals {correct_result}. "
            
            if operation == '+':
                explanation += f"When you add {operand2} to {operand1}, you get {correct_result}, not {found_result}."
            elif operation == '-':
                explanation += f"When you subtract {operand2} from {operand1}, you count down: {operand1}, {correct_result}. So {operand1} - {operand2} = {correct_result}, not {found_result}."
            elif operation == '*':
                explanation += f"When you multiply {operand1} by {operand2}, you get {correct_result}, not {found_result}."
            elif operation == '/':
                explanation += f"When you divide {operand1} by {operand2}, you get {correct_result}, not {found_result}."
            else:
                explanation += f"The correct calculation gives {correct_result}, not {found_result}."
            
            return explanation
    except:
        pass
    
    return f"You wrote '{found}', but the correct answer is '{correct}'."


def _explain_logical_error(error: Dict[str, Any], step_number: int) -> str:
    """Explain logical consistency errors."""
    description = error.get("description", "There is a logical inconsistency in your reasoning.")
    
    explanation = f"In step {step_number}, {description.lower()} "
    explanation += "This contradicts your earlier statements or creates circular reasoning."
    
    return explanation


def _explain_operation_mismatch(error: Dict[str, Any], found: str, correct: str) -> str:
    """Explain operation mismatch errors."""
    description = error.get("description", "The operation doesn't match what you described.")
    
    explanation = f"{description} "
    explanation += "Make sure the mathematical operation matches what you're trying to do in the problem."
    
    return explanation


def _explain_semantic_error(error: Dict[str, Any], found: str, correct: str) -> str:
    """Explain semantic inconsistency errors."""
    description = error.get("description", "The meaning doesn't match the mathematical expression.")
    
    explanation = f"{description} "
    explanation += "Review the step to ensure it follows logically from the previous steps."
    
    return explanation


def _explain_generic_error(found: str, correct: str) -> str:
    """Generic error explanation."""
    return f"You wrote '{found}', but the correct answer should be '{correct}'. Please review your calculation."


def _get_learning_tip(error_type: str, operation: str) -> str:
    """Get learning tip based on error type."""
    tips = {
        "calculation_error": "Try double-checking your arithmetic by working through the problem step by step.",
        "logical_error": "Make sure each step follows logically from the previous one and doesn't contradict earlier statements.",
        "operation_mismatch": "Before writing the math, think about which operation matches what you're trying to do.",
        "semantic_error": "Review how this step connects to the overall problem and previous steps."
    }
    
    return tips.get(error_type, "Practice similar problems to improve your understanding.")

