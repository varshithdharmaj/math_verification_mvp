"""
Utility helper functions for MathVerify-Integrated.
"""

import json
import os
from typing import Dict, List, Any, Optional


def load_json(file_path: str) -> Any:
    """
    Load JSON data from file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
    """
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_confidence(confidence: float) -> str:
    """
    Format confidence score as percentage string.
    
    Args:
        confidence: Confidence score (0.0-1.0)
        
    Returns:
        Formatted string (e.g., "85.5%")
    """
    return f"{confidence * 100:.1f}%"


def validate_result_structure(result: Dict) -> bool:
    """
    Validate that result dictionary has required structure.
    
    Args:
        result: Result dictionary to validate
        
    Returns:
        True if structure is valid
    """
    required_keys = ['problem', 'solution', 'verification', 'errors', 'final_answer', 'confidence']
    return all(key in result for key in required_keys)

