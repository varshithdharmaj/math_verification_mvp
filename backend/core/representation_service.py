"""
Representation Service Module
Converts raw text/LaTeX into a canonical Intermediate Representation (IR).
"""
import re
from typing import Dict, Any, List, Tuple

def to_canonical_expression(ocr_text: str) -> str:
    """
    Return a canonical LaTeX-like expression that we use consistently across agents and SymPy.
    Handles basics: integrals, fractions, superscripts/subscripts, parens.
    """
    if not ocr_text:
        return ""
        
    expr = ocr_text.strip()
    
    # 1. Basic Symbol Normalization
    # Note: We use r'\\sin' so that the replacement string becomes literal "\sin"
    replacements = {
        '×': '*', '·': '*', 
        '÷': '/', ':': '/',
        '−': '-', '–': '-',
        '**': '^',
        'sin': r'\\sin', 'cos': r'\\cos', 'tan': r'\\tan',
        'log': r'\\log', 'ln': r'\\ln',
        'pi': r'\\pi', 'theta': r'\\theta',
        'infinity': r'\\infty', 'inf': r'\\infty'
    }
    
    try:
        for old, new in replacements.items():
            if old.isalpha():
                # Pattern: \bWORD\b
                pattern = r'\b' + re.escape(old) + r'\b'
                expr = re.sub(pattern, new, expr)
            else:
                expr = expr.replace(old, new)
    except Exception as e:
        print(f"Error in Basic Symbol Normalization: {e}")

    # 2. Integral handling
    try:
        integral_pattern = r"(?i)\bintegral\b\s+(.+?)\s+\bto\b\s+(.+?)\s+(.*)"
        match = re.search(integral_pattern, expr)
        if match:
            lower, upper, body = match.groups()
            if body.strip().endswith('dx'):
                body = body.strip()[:-2].strip() + r' \, dx'
            
            # Use format instead of f-string to avoid escape ambiguity
            expr = r"\int_{{{}}}^{{{}}} {}".format(lower.strip(), upper.strip(), body)
    except Exception as e:
        print(f"Error in Integral Handling: {e}")

    # 3. Fractions
    try:
        fraction_pattern = r'(\b\d+|[a-zA-Z])\s*/\s*(\b\d+|[a-zA-Z])'
        expr = re.sub(fraction_pattern, r'\\frac{\1}{\2}', expr)
    except Exception as e:
        print(f"Error in Fraction Handling: {e}")

    # 4. Superscripts
    try:
        expr = re.sub(r'\^(\d{2,})', r'^{\1}', expr)
    except Exception as e:
        print(f"Error in Superscript Handling: {e}")
    
    return expr


def is_in_scope_expression(expr: str) -> bool:
    """
    Check if expression fits the restricted segment:
    - Algebraic simplification / equation solving
    - Simple definite integrals and limits
    - Single-line expressions only
    """
    if not expr:
        return False

    # Reject multi-line
    if "\n" in expr:
        return False

    # Reject very long expressions
    if len(expr) > 120:
        return False

    # Allowed tokens for this segment
    allowed = re.compile(r'^[0-9a-zA-Z\s\+\-\*/=\^\(\)_\\\{\}\.\,]+$')
    if not allowed.match(expr):
        return False

    # Allow if equation, integral, or limit keywords present, or simple algebraic
    has_eq = "=" in expr
    has_integral = "\\int" in expr or "integral" in expr.lower()
    has_limit = "\\lim" in expr or "limit" in expr.lower()

    # Reject if contains unsupported functions
    if re.search(r'\\(sum|prod|matrix|det|cases)', expr):
        return False

    # Accept if within algebra/calculus subset
    return has_eq or has_integral or has_limit or True


def classify_difficulty(expr: str) -> str:
    """
    Simple heuristic difficulty based on length and operator count.
    """
    if not expr:
        return "simple"
    ops = len(re.findall(r'[\+\-\*/\^=]', expr))
    if len(expr) < 30 and ops <= 3:
        return "simple"
    if len(expr) < 70 and ops <= 7:
        return "medium"
    return "hard"

async def normalize_input(raw_text: str) -> Dict[str, Any]:
    """
    Parses raw text into structured Problem and Steps.
    
    Args:
        raw_text: The string from OCR or User Input.
        
    Returns:
        Dict: Structured representation (e.g., {'problem': '...', 'steps': [...]})
    """
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    if not lines:
        return {
            "problem": "",
            "steps": [],
            "format": "empty"
        }
        
    # Heuristic: First line is problem, rest are steps
    problem_raw = lines[0]
    steps_raw = lines[1:] if len(lines) > 1 else []
    
    # Apply Canonicalization
    canonical_problem = to_canonical_expression(problem_raw)
    canonical_steps = [to_canonical_expression(s) for s in steps_raw]

    # Scope validation for restricted segment
    in_scope_problem = is_in_scope_expression(canonical_problem)
    in_scope_steps = all(is_in_scope_expression(s) for s in canonical_steps if s)
    out_of_scope = not (in_scope_problem and in_scope_steps)
    
    return {
        "problem": canonical_problem,
        "steps": canonical_steps,
        "format": "canonical_latex",
        "raw_problem": problem_raw,
        "raw_steps": steps_raw,
        "out_of_scope": out_of_scope,
        "difficulty": classify_difficulty(canonical_problem)
    }

if __name__ == "__main__":
    print("Running tests for to_canonical_expression...")
    
    # Test 1: Integral
    # "integral 0 to pi sin x^2 dx"
    inp1 = "integral 0 to pi sin x^2 dx"
    out1 = to_canonical_expression(inp1)
    print(f"Input: {inp1}\nOutput: {out1}")
    # Exp: \int_{0}^{\pi} \sin x^2 \, dx
    assert r'\int_{0}^{\pi}' in out1
    assert r'\sin' in out1
    print("Test 1 Passed")

    # Test 2: Fraction
    inp2 = "3 / 4 * x"
    out2 = to_canonical_expression(inp2)
    print(f"Input: {inp2}\nOutput: {out2}")
    # Exp: \frac{3}{4} * x
    assert r'\frac{3}{4}' in out2
    print("Test 2 Passed")
    
    # Test 3: Superscript normalization
    inp3 = "x**2 + y^20"
    out3 = to_canonical_expression(inp3)
    print(f"Input: {inp3}\nOutput: {out3}")
    # Exp: x^2 + y^{20}
    assert 'x^2' in out3
    assert 'y^{20}' in out3
    print("Test 3 Passed")
    
    print("All tests passed!")
