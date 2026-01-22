"""
Classifier Service Module
Aggregates verification results and assigns a final score/category.
"""
from typing import Dict, Any, List
# Import verification service helpers (ensure this circular import is handled or logic is moved)
# Since classifier depends on verification outputs, simple import should be fine if main calls them sequentially.
try:
    from backend.core.verification_service import (
        compute_symbolic_score, 
        compute_logical_score,
        compute_step_consensus
    )
except ImportError:
    # Fallback for direct execution testing
    pass

def compute_final_confidence(score: float, ocr_conf: float) -> float:
    """
    Computes final confidence with OCR calibration (MVM² Eq similar).
    FinalConf = Score * (0.9 + 0.1 * OCRconf)
    This ensures that even a perfect logic score is dampened if OCR was garbage.
    """
    calibration = 0.9 + (0.1 * ocr_conf)
    return score * calibration

def compute_clf_score(agent_consensus_scores: List[float], steps: List[str]) -> float:
    """
    Computes classifier score (Rule-Based for now).
    - Penalize low consensus steps (potential hallucinations).
    - Penalize 'short' answers if others are long? (optional)
    """
    if not agent_consensus_scores:
        return 0.5 # Neutral
        
    avg_consensus = sum(agent_consensus_scores) / len(agent_consensus_scores)
    
    # Penalize if any step is very low consensus (< 0.4 implies hallucination/contradiction)
    min_score = min(agent_consensus_scores)
    penalty = 0.0
    if min_score < 0.4:
        penalty = 0.2
        
    # Base score is average consensus
    return max(0.0, avg_consensus - penalty)

async def classify_and_score(verification_results: Dict[str, Any], ocr_confidence: float = 1.0, use_ocr_calibration: bool = True) -> Dict[str, Any]:
    """
    Computes final verdict using weighted consensus of Agents.
    """
    # 1. Unpack Agent Results
    llm_output = verification_results.get("llm", {})
    agent_results = llm_output.get("details", [])
    
    # If no agents (or fallback "Offline"), use Global SymPy score if available
    if not agent_results:
        # Fallback to simple logic
        sym_global = 1.0 if verification_results.get("sympy", {}).get("valid") else 0.0
        return {
            "final_verdict": "VALID" if sym_global > 0.5 else "ERROR",
            "confidence_score": sym_global,
            "error_category": "Offline/No Agents",
            "best_agent": "None"
        }
    
    # 2. Compute Consensus Map
    # Prepare map: { "AgentName": ["step1", "step2"] }
    # Only use agents that provided steps
    agent_steps_map = {
        res["agent_name"]: res.get("steps", []) 
        for res in agent_results 
        if res.get("steps")
    }
    
    consensus_map = compute_step_consensus(agent_steps_map)
    
    scored_agents = []
    
    # 3. Score Each Agent
    for res in agent_results:
        name = res["agent_name"]
        
        # A. Symbolic Score
        sym = compute_symbolic_score(res)
        
        # B. Logical Score
        logic = compute_logical_score(res)
        
        # C. Classifier Score (Consensus)
        cons_scores = consensus_map.get(name, [])
        clf = compute_clf_score(cons_scores, res.get("steps", []))
        
        # D. Weighted Sum
        # Score_j = 0.4*sym + 0.35*logic + 0.25*clf
        raw_score = (0.4 * sym) + (0.35 * logic) + (0.25 * clf)
        
        # E. Final Confidence
        if use_ocr_calibration:
            final_conf = compute_final_confidence(raw_score, ocr_confidence)
        else:
            final_conf = raw_score
        
        # Calculate Consensus Stats
        avg_cons = sum(cons_scores)/len(cons_scores) if cons_scores else 0.0
        low_cons_steps = sum(1 for s in cons_scores if s < 0.4)
        hallucination_rate = low_cons_steps / len(cons_scores) if cons_scores else 0.0
        
        scored_agents.append({
            "agent": name,
            "raw_score": raw_score,
            "final_conf": final_conf,
            "components": {"sym": sym, "logic": logic, "clf": clf},
            "consensus_stats": {
                "avg_consensus": avg_cons,
                "hallucination_rate": hallucination_rate,
                "total_steps": len(cons_scores)
            },
            "data": res
        })
        
    # 4. Select Best Agent
    if not scored_agents:
        return {"final_verdict": "ERROR", "confidence_score": 0.0, "error_category": "No Scorable Agents"}
        
    best_agent = max(scored_agents, key=lambda x: x["final_conf"])
    
    # 5. Determine Final Verdict based on Best Agent
    # If Best Agent says "valid" (implied by high score usually, but usually we check the content too)
    # Actually, high score means it's a "Good Solution". 
    # Whether the solution says "Problem is Correct" or "Here is the Correct Answer" depends on prompt.
    # Our prompt asked "Solve... Return valid/invalid".
    
    # Check best agent's internal validity flag
    # If logic score is low, it might be invalid.
    
    is_valid = best_agent["final_conf"] > 0.6
    
    return {
        "final_verdict": "VALID" if is_valid else "ERROR",
        "confidence_score": round(best_agent["final_conf"], 3),
        "error_category": "None" if is_valid else f"Low Confidence ({best_agent['agent']})",
        "best_agent": best_agent["agent"],
        "final_answer": best_agent.get("data", {}).get("final_answer", ""),
        "consensus_stats": best_agent.get("consensus_stats", {}),
        "all_scores": [
            {
                "name": a["agent"], 
                "score": round(a["final_conf"], 3),
                "breakdown": a["components"]
            } 
            for a in scored_agents
        ],
        "winning_reasoning": best_agent["data"].get("reasoning", "")
    }
