"""
Error Classification & Taxonomy
Classifies errors into 10+ error types with severity and fixability assessment
"""

import re
from typing import Dict, Any, List


def classify_error(error: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify error into taxonomy with severity and fixability.
    
    Args:
        error: Error dictionary with type, found, correct, etc.
        
    Returns:
        Enhanced error dictionary with classification details
    """
    error_type = error.get("type", "unknown")
    found = error.get("found", "")
    correct = error.get("correct", "")
    operation = error.get("operation", "")
    
    # Classification mapping
    classification_map = {
        "calculation_error": {
            "category": "Arithmetic Error",
            "description": "Calculation mistakes in arithmetic operations",
            "severity": "HIGH",
            "fixability": 0.92,  # 92% fixable
            "fixable": True
        },
        "logical_error": {
            "category": "Logical Error",
            "description": "Contradictions, circular reasoning, or logical inconsistencies",
            "severity": "MEDIUM",
            "fixability": 0.60,  # 60% fixable
            "fixable": False
        },
        "operation_mismatch": {
            "category": "Operation Mismatch",
            "description": "Text describes one operation but math uses another",
            "severity": "HIGH",
            "fixability": 0.68,  # 68% fixable
            "fixable": True
        },
        "semantic_error": {
            "category": "Semantic Error",
            "description": "Meaning doesn't match the mathematical expression",
            "severity": "MEDIUM",
            "fixability": 0.45,  # 45% fixable
            "fixable": False
        }
    }
    
    # Determine classification based on error type
    if error_type in classification_map:
        classification = classification_map[error_type]
    else:
        # Try to infer from content
        classification = _infer_classification(error, found, correct, operation)
    
    # Additional error type detection from content
    additional_types = _detect_additional_error_types(found, correct, operation)
    
    # Enhance error dictionary
    enhanced_error = error.copy()
    enhanced_error.update({
        "category": classification["category"],
        "error_description": classification["description"],
        "severity": classification["severity"],
        "fixability_score": classification["fixability"],
        "fixable": classification["fixable"],
        "additional_types": additional_types
    })
    
    return enhanced_error


def _infer_classification(error: Dict[str, Any], found: str, correct: str, operation: str) -> Dict[str, Any]:
    """Infer error classification from content when type is unknown."""
    found_lower = found.lower()
    correct_lower = correct.lower()
    
    # Check for algebraic errors (variables)
    if re.search(r'[a-zA-Z]', found):
        return {
            "category": "Algebraic Error",
            "description": "Wrong operations on variables or algebraic expressions",
            "severity": "HIGH",
            "fixability": 0.75,
            "fixable": True
        }
    
    # Check for unit errors
    if re.search(r'\b(kg|g|m|cm|km|lb|oz|ft|in)\b', found_lower):
        return {
            "category": "Unit Error",
            "description": "Wrong units or unit conversions",
            "severity": "MEDIUM",
            "fixability": 0.70,
            "fixable": True
        }
    
    # Check for sign errors
    if operation in ['+', '-']:
        # Check if result has wrong sign
        found_nums = re.findall(r'-?\d+\.?\d*', found)
        correct_nums = re.findall(r'-?\d+\.?\d*', correct)
        if found_nums and correct_nums:
            try:
                found_result = float(found_nums[-1])
                correct_result = float(correct_nums[-1])
                if abs(found_result) == abs(correct_result) and found_result != correct_result:
                    return {
                        "category": "Sign Error",
                        "description": "Wrong positive/negative sign",
                        "severity": "HIGH",
                        "fixability": 0.90,
                        "fixable": True
                    }
            except:
                pass
    
    # Default to arithmetic error
    return {
        "category": "Arithmetic Error",
        "description": "Calculation mistakes",
        "severity": "HIGH",
        "fixability": 0.85,
        "fixable": True
    }


def _detect_additional_error_types(found: str, correct: str, operation: str) -> List[str]:
    """Detect additional error types from content."""
    additional_types = []
    found_lower = found.lower()
    
    # Check for notation errors
    if re.search(r'[≈~≈]', found) or re.search(r'\b(about|around|approximately)\b', found_lower):
        additional_types.append("Notation Error")
    
    # Check for order of operations issues
    if re.search(r'\d+\s*[+\-*/]\s*\d+\s*[+\-*/]\s*\d+', found):
        # Multiple operations without parentheses might indicate order issue
        if '(' not in found and '(' in correct:
            additional_types.append("Order of Operations")
    
    # Check for conceptual errors (complex patterns)
    if len(found.split()) > 10:  # Very long expressions might indicate conceptual issues
        additional_types.append("Conceptual Error")
    
    return additional_types

