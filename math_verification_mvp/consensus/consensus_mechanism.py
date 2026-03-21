import logging
import sys
import os

# Add MathVerifyProject to path for math_verify library
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HANDWRITING_REASONING_PATH = os.path.join(BASE_PATH, "h -math llm reasoning")
MATH_VERIFY_PATH = os.path.join(HANDWRITING_REASONING_PATH, "MathVerifyProject")

if os.path.exists(MATH_VERIFY_PATH):
    sys.path.insert(0, MATH_VERIFY_PATH)

try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

logger = logging.getLogger(__name__)

def evaluate_symbolic_math(steps: List[str]) -> float:
    """
    Symbolically check step-by-step transformations.
    Uses math_verify library if available, falls back to basic SymPy.
    Returns a score 0.0 to 1.0 based on how many mathematical equalities hold.
    """
    if not steps:
        return 0.0
        
    valid_count = 0
    math_statements = 0
    
    for step in steps:
        if "=" in step:
            math_statements += 1
            clean_step = step.replace("Let x", "").replace("let", "").strip()
            
            try:
                left, right = clean_step.split('=', 1)
                
                if MATH_VERIFY_AVAILABLE:
                    # Use math_verify for robust LaTeX-aware comparison
                    res_left = parse(left)
                    res_right = parse(right)
                    if verify(res_left, res_right):
                        valid_count += 1
                else:
                    # Fallback to simple SymPy
                    import sympy
                    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_equals_signs
                    transformations = standard_transformations + (implicit_multiplication_application, convert_equals_signs)
                    left_expr = parse_expr(left, transformations=transformations)
                    right_expr = parse_expr(right, transformations=transformations)
                    if sympy.simplify(left_expr - right_expr) == 0:
                        valid_count += 1
            except Exception:
                pass
                
    if math_statements == 0:
        return 0.5 # Neutral if no explicit math equations found
    
    return valid_count / math_statements

def compute_neurosymbolic_consensus(agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Implements the MVM² Hybrid Verification System: SymPy + Divergence Matrix + Weighted Scoring.
        40% Symbolic check validity
        35% Logical Consistency (Simulated via Agent Confidence Trace)
        25% Classifier Signal (Divergence Matrix step alignment)
    """
    
    # 1. Divergence Matrix (Step-Level Alignment)
    divergence_scores = {}
    hallucination_alerts = []
    
    # Simple proxy for divergence: Compare the variance in number of reasoning steps
    agent_names = list(agent_results.keys())
    trace_lengths = {name: len(agent_results[name].get("reasoning_trace", [])) for name in agent_names}
    avg_length = sum(trace_lengths.values()) / max(1, len(trace_lengths))
    
    for name, length in trace_lengths.items():
        # High deviation from average length implies divergent reasoning path
        if avg_length == 0:
            divergence = 0.5
        else:
            deviation = min(1.0, abs(length - avg_length) / avg_length)
            divergence = 1.0 - deviation # 1.0 = perfect alignment
            
        divergence_scores[name] = divergence
        
        # Hallucination Alert Threshold (<0.7)
        if divergence < 0.7:
            hallucination_alerts.append(f"Alert: {name} fell below 0.7 step agreement (Score: {divergence:.2f}). Possible hallucination detected.")
            
    # 2. Extract Answers and Score Individual Agents
    final_agent_scores = {}
    
    for name, data in agent_results.items():
        steps = data.get("reasoning_trace", [])
        
        # A. Symbolic Check (40%)
        symbolic_score = evaluate_symbolic_math(steps)
        
        # B. Logical Consistency (35%)
        # Map agent's internal confidence evaluation
        conf_text = str(data.get("confidence_explanation", "")).lower()
        if "hallucination" in conf_text or "error" in conf_text or "guess" in conf_text:
            logical_score = 0.3
        else:
            logical_score = 0.95 
            
        # C. Classifier Signal (25%)
        # Use divergence matrix step-alignment score
        clf_score = divergence_scores.get(name, 0.5)
        
        # Calculate Domain-Informed Weighted Scoring
        weighted_score = (0.40 * symbolic_score) + (0.35 * logical_score) + (0.25 * clf_score)
        
        final_agent_scores[name] = {
            "symbolic": round(symbolic_score, 3),
            "logical": round(logical_score, 3),
            "classifier": round(clf_score, 3),
            "weighted_score": round(weighted_score, 3),
            "final_answer": data.get("final_answer", "ERROR")
        }
        
    # 3. Overall System Decision
    best_agent = max(final_agent_scores.items(), key=lambda x: x[1]["weighted_score"])
    best_name = best_agent[0]
    best_score = best_agent[1]["weighted_score"]
    
    if best_score > 0.65:
        final_verdict = "VALID"
        overall_confidence = min(0.99, best_score * 1.1)
    else:
        final_verdict = "ERROR"
        overall_confidence = max(0.1, best_score)
        
    return {
        "final_verdict": final_verdict,
        "overall_confidence": round(overall_confidence, 3),
        "chosen_answer": best_agent[1]["final_answer"],
        "chosen_agent": best_name,
        "hallucination_alerts": hallucination_alerts,
        "divergence_scores": {k: round(v, 3) for k, v in divergence_scores.items()},
        "agent_scoring_breakdown": final_agent_scores,
        "all_errors": [], # Kept for API compatibility with legacy Dashboard
        "individual_verdicts": {k: "VALID" if v["weighted_score"] > 0.65 else "ERROR" for k,v in final_agent_scores.items()}
    }
