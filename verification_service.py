import re
from typing import List, Dict, Any
from sympy import sympify, simplify, Eq, parse_expr

def extract_equations(text: str) -> List[str]:
    """Extracts mathematical equations or expressions from a reasoning step."""
    patterns = [
        r'(\$.*?\$)',
        r'(\\\[.*?\\\])',
        r'([a-zA-Z0-9\(\)\+\-\*\/]+ *= *[a-zA-Z0-9\(\)\+\-\*\/]+)'
    ]
    
    found = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            clean = m.replace('$', '').replace('\\[', '').replace('\\]', '').strip()
            if '=' in clean:
                found.append(clean)
    
    if not found:
        lines = text.split('\n')
        for line in lines:
            if "=" in line and sum(c.isalpha() for c in line) < len(line) / 2:
                found.append(line.strip())
    return found

def check_logical_progression(step_n: str, step_n_plus_1: str) -> bool:
    """
    Implements the SymPy Validation function \vartheta(r_{jl}).
    """
    eqs_n = extract_equations(step_n)
    eqs_n_plus_1 = extract_equations(step_n_plus_1)
    
    if not eqs_n or not eqs_n_plus_1:
        return True
        
    try:
        for eq1 in eqs_n:
            for eq2 in eqs_n_plus_1:
                if re.search(r'(\d+) *= *(?!\1)(\d+)', eq2):
                    return False

                if '=' in eq1 and '=' in eq2:
                    lhs1, rhs1 = eq1.split('=', 1)
                    lhs2, rhs2 = eq2.split('=', 1)
                    
                    expr1 = parse_expr(lhs1.replace('^', '**')) - parse_expr(rhs1.replace('^', '**'))
                    expr2 = parse_expr(lhs2.replace('^', '**')) - parse_expr(rhs2.replace('^', '**'))
                    
                    if simplify(expr1) == simplify(expr2) or simplify(expr1 + expr2) == 0:
                        return True
                        
    except Exception:
        pass
        
    if re.search(r'\b(\d+)\s*=\s*(?!\1)(\d+)\b', step_n_plus_1):
        return False
        
    return True

def calculate_symbolic_score(reasoning_trace: List[str]) -> float:
    """
    Calculates V^{sym}_j based on the logical sequence of steps.
    """
    if not reasoning_trace:
        return 0.0
    if len(reasoning_trace) <= 1:
        return 1.0
        
    valid_transitions = 0
    total_transitions = len(reasoning_trace) - 1
    
    for i in range(total_transitions):
        is_valid = check_logical_progression(reasoning_trace[i], reasoning_trace[i+1])
        if is_valid:
            valid_transitions += 1
            
    v_sym = float(valid_transitions) / float(total_transitions)
    
    for step in reasoning_trace:
        if not check_logical_progression("", step):
            v_sym *= 0.5
            break
            
    return round(v_sym, 2)
