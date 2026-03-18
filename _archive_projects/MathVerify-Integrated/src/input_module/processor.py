"""
Input processing module for mathematical problems.

This module handles normalization, extraction, and validation of mathematical text input.
"""

import re
from typing import Dict, Optional


class InputProcessor:
    """
    Processes and normalizes mathematical problem input.
    
    Handles conversion of various mathematical notations to canonical form,
    extracts problem components, and validates input.
    
    Examples:
        >>> processor = InputProcessor()
        >>> processor.normalize_expression("2x + 3")
        '2*x + 3'
        >>> processor.normalize_expression("x^2 + 1/2")
        'x**2 + 1/2'
        >>> processor.extract_problem_components("Solve for x: 2x + 3 = 7")
        {'question': 'Solve for x: 2x + 3 = 7', 'constraints': [], 'unknowns': ['x']}
    """
    
    def __init__(self):
        """Initialize the InputProcessor."""
        pass
    
    def normalize_expression(self, text: str) -> str:
        """
        Converts mathematical notation to canonical form.
        
        Handles:
        - Fractions (1/2) -> kept as is
        - Exponents (x^2) -> x**2
        - Implicit multiplication (2x) -> 2*x
        - Multiple spaces -> single space
        
        Args:
            text: Input mathematical expression string
            
        Returns:
            Standardized mathematical expression string
            
        Raises:
            ValueError: If input is empty or None
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert exponent notation: x^2 -> x**2
        text = re.sub(r'(\w+)\^(\d+)', r'\1**\2', text)
        
        # Convert implicit multiplication: 2x -> 2*x, but preserve numbers like 2.5
        # Match number followed by variable: (number)(letter)
        text = re.sub(r'(\d+\.?\d*)([a-zA-Z])', r'\1*\2', text)
        
        # Match variable followed by number: (letter)(number) - less common but handle it
        text = re.sub(r'([a-zA-Z])(\d+)', r'\1*\2', text)
        
        # Match variable followed by variable: xy -> x*y
        text = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', text)
        
        # Normalize parentheses spacing: ( x ) -> (x)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        
        return text.strip()
    
    def extract_problem_components(self, text: str) -> Dict[str, any]:
        """
        Extracts structured components from a mathematical problem.
        
        Extracts:
        - question: The main problem statement
        - constraints: Any given constraints or conditions
        - unknowns: Variables to solve for
        
        Args:
            text: Input problem text
            
        Returns:
            Dictionary with keys: 'question', 'constraints', 'unknowns'
            
        Example:
            >>> processor = InputProcessor()
            >>> result = processor.extract_problem_components("Solve for x: 2x + 3 = 7")
            >>> result['unknowns']
            ['x']
        """
        if not text:
            return {
                "question": "",
                "constraints": [],
                "unknowns": []
            }
        
        # Extract unknowns (variables) - common single letters
        # Match patterns like "solve for x", "find x", "x =", etc.
        unknowns = []
        
        # Look for explicit mentions: "solve for x", "find x", "x ="
        solve_pattern = r'(?:solve|find|determine)\s+for?\s+([a-zA-Z])'
        matches = re.findall(solve_pattern, text, re.IGNORECASE)
        unknowns.extend(matches)
        
        # Look for variable assignments: "x =", "y =", etc.
        var_pattern = r'\b([a-zA-Z])\s*='
        var_matches = re.findall(var_pattern, text)
        unknowns.extend(var_matches)
        
        # Look for common variable names in equations
        equation_pattern = r'\b([a-zA-Z])\s*[+\-*/=]'
        eq_matches = re.findall(equation_pattern, text)
        unknowns.extend(eq_matches)
        
        # Remove duplicates and sort
        unknowns = sorted(list(set(unknowns)))
        
        # Extract constraints (lines with keywords like "given", "if", "when")
        constraints = []
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in ['given', 'if', 'when', 'provided', 'assume']):
                constraints.append(line.strip())
        
        return {
            "question": text.strip(),
            "constraints": constraints,
            "unknowns": unknowns
        }
    
    def validate_input(self, text: str) -> bool:
        """
        Checks if input is valid mathematical text.
        
        Validates:
        - Non-empty string
        - Contains mathematical content (numbers, operators, variables)
        - Not just whitespace or special characters
        
        Args:
            text: Input text to validate
            
        Returns:
            True if input appears to be valid mathematical text, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
        
        # Remove whitespace
        text_clean = text.strip()
        if not text_clean:
            return False
        
        # Check for mathematical content
        # Must contain at least one: number, operator, or common math keyword
        has_number = bool(re.search(r'\d', text_clean))
        has_operator = bool(re.search(r'[+\-*/=<>≤≥]', text_clean))
        has_variable = bool(re.search(r'\b[a-zA-Z]\b', text_clean))
        has_math_keyword = bool(re.search(
            r'\b(solve|find|calculate|compute|evaluate|equation|expression|formula)\b',
            text_clean,
            re.IGNORECASE
        ))
        
        # Valid if has at least one mathematical element
        return has_number or has_operator or (has_variable and has_math_keyword)

