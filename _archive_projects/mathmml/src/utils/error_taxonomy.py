"""Error taxonomy with 10+ error types, severity, and fixability."""

from typing import Dict, List, Optional
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ErrorFixability(Enum):
    """Error fixability levels."""
    AUTO_FIXABLE = "auto_fixable"
    MANUAL_REVIEW = "manual_review"


ERROR_TAXONOMY = {
    "correct": {
        "severity": None,
        "fixability": None,
        "description": "No error detected"
    },
    "arithmetic_error": {
        "severity": ErrorSeverity.MEDIUM,
        "fixability": ErrorFixability.AUTO_FIXABLE,
        "description": "Incorrect arithmetic calculation"
    },
    "logical_error": {
        "severity": ErrorSeverity.HIGH,
        "fixability": ErrorFixability.MANUAL_REVIEW,
        "description": "Logical inconsistency or flawed reasoning"
    },
    "operation_mismatch": {
        "severity": ErrorSeverity.MEDIUM,
        "fixability": ErrorFixability.AUTO_FIXABLE,
        "description": "Wrong operation used (e.g., addition instead of subtraction)"
    },
    "conceptual_error": {
        "severity": ErrorSeverity.HIGH,
        "fixability": ErrorFixability.MANUAL_REVIEW,
        "description": "Fundamental misunderstanding of concept"
    },
    "notation_error": {
        "severity": ErrorSeverity.LOW,
        "fixability": ErrorFixability.AUTO_FIXABLE,
        "description": "Incorrect mathematical notation"
    },
    "sign_error": {
        "severity": ErrorSeverity.MEDIUM,
        "fixability": ErrorFixability.AUTO_FIXABLE,
        "description": "Incorrect sign (positive/negative)"
    },
    "unit_error": {
        "severity": ErrorSeverity.MEDIUM,
        "fixability": ErrorFixability.AUTO_FIXABLE,
        "description": "Unit mismatch or missing units"
    },
    "order_ops_error": {
        "severity": ErrorSeverity.MEDIUM,
        "fixability": ErrorFixability.AUTO_FIXABLE,
        "description": "Incorrect order of operations"
    },
    "semantic_error": {
        "severity": ErrorSeverity.HIGH,
        "fixability": ErrorFixability.MANUAL_REVIEW,
        "description": "Semantic inconsistency with problem context"
    }
}


def get_error_info(error_type: str) -> Dict:
    """Get error taxonomy information.
    
    Args:
        error_type: Error type string
        
    Returns:
        Dict with severity, fixability, description
    """
    return ERROR_TAXONOMY.get(error_type, {
        "severity": ErrorSeverity.MEDIUM,
        "fixability": ErrorFixability.MANUAL_REVIEW,
        "description": "Unknown error type"
    })


def get_severity(error_type: str) -> Optional[ErrorSeverity]:
    """Get error severity.
    
    Args:
        error_type: Error type string
        
    Returns:
        ErrorSeverity enum or None
    """
    info = get_error_info(error_type)
    return info.get("severity")


def get_fixability(error_type: str) -> Optional[ErrorFixability]:
    """Get error fixability.
    
    Args:
        error_type: Error type string
        
    Returns:
        ErrorFixability enum or None
    """
    info = get_error_info(error_type)
    return info.get("fixability")


def is_auto_fixable(error_type: str) -> bool:
    """Check if error is auto-fixable.
    
    Args:
        error_type: Error type string
        
    Returns:
        True if auto-fixable
    """
    fixability = get_fixability(error_type)
    return fixability == ErrorFixability.AUTO_FIXABLE

