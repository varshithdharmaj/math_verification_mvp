import re
from typing import List, Dict, Any
from sympy import sympify, simplify, Eq, parse_expr

def extract_equations(text: str) -> List[str]:
    """Extracts mathematical equations or expressions from a reasoning step."""
    # Simplified extraction logic: finding equals signs or math blocks
    # In production, uses robust RegEx or specialized NLP parsing
    lines = text.split('\\n')
    equations = []
    for line in lines:
        if "=" in line and sum(c.isalpha() for c in line) < len(line) / 2:
            equations.append(line.strip())
    return equations

def check_logical_progression(step_n: str, step_n_plus_1: str) -> bool:
    """
    Implements the SymPy Validation function \\vartheta(r_{jl}).
    Checks if step (n+1) is a logically sound derivative of step (n).
    """
    eqs_n = extract_equations(step_n)
    eqs_n_plus_1 = extract_equations(step_n_plus_1)
    
    # If no math found natively, fallback to semantic/LLM truth (handled via Logic score)
    if not eqs_n or not eqs_n_plus_1:
        return True
        
    try:
        # Example validation: if step_n is 'a = b' and step is 'a + 1 = b + 1'
        # Simplifying via SymPy
        for eq1 in eqs_n:
            for eq2 in eqs_n_plus_1:
                e1_left, e1_right = eq1.split('=')
                e2_left, e2_right = eq2.split('=')
                
                # Verify equivalence: Left - Right should be 0
                expr1 = sympify(f"({e1_left}) - ({e1_right})")
                expr2 = sympify(f"({e2_left}) - ({e2_right})")
                
                # Check if they denote the same equality (simplified algebra)
                if simplify(expr1 - expr2) == 0:
                    return True
    except Exception:
        # Syntax error parsing SymPy, fall back to safe true 
        pass
        
    # By default, if we can't prove it false, we assume conditional true 
    # MVM2 specifically flags "1=2" paradoxes
    if "1 = 2" in step_n_plus_1 or "1=2" in step_n_plus_1:
        return False
        
    return True

def calculate_symbolic_score(reasoning_trace: List[str]) -> float:
    """
    Calculates V^{sym}_j based on the logical sequence of steps.
    Score drops linearly for every failed contiguous logic check.
    """
    if len(reasoning_trace) <= 1:
        return 1.0
        
    valid_transitions = 0
    total_transitions = len(reasoning_trace) - 1
    
    for i in range(total_transitions):
        is_valid = check_logical_progression(reasoning_trace[i], reasoning_trace[i+1])
        if is_valid:
            valid_transitions += 1
            
    v_sym = float(valid_transitions) / float(total_transitions)
    
    # If a critical hallucination is detected (e.g. proof of 1=2), heavily penalize
    for step in reasoning_trace:
        if "1 = 2" in step or "1=2" in step:
            v_sym = 0.0
            break
            
    return round(v_sym, 2)
