"""
Reporting Service Module
Generates the final comprehensive report for the user.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

def build_verification_report(
    problem_id: str,
    ocr_output: Dict[str, Any],
    canonical_expr: Dict[str, Any],
    agent_results: List[Dict[str, Any]],
    step_consensus: Dict[str, List[float]],
    scores: Dict[str, Any],
    final_choice: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Constructs a detailed verification report containing all pipeline artifacts.
    """
    
    # 1. Input Section
    input_report = {
        "problem_id": problem_id,
        "ocr_text": ocr_output.get("raw_text", ""),
        "ocr_confidence": ocr_output.get("confidence", 0.0),
        "input_type": "image" if ocr_output.get("raw_text") else "text",
    }
    
    # 2. Canonical Section
    canonical_report = {
        "problem_latex": canonical_expr.get("problem", ""),
        "steps_latex": canonical_expr.get("steps", [])
    }
    
    # 3. Agent Analysis
    agents_report = []
    
    # We need to map scores back to agents. 'scores' usually comes from classifier output
    # `scores` might be the dict returned by classify_and_score which contains "all_scores" list
    all_scores_map = {
        s["name"]: s 
        for s in scores.get("all_scores", [])
    }
    
    for agent_res in agent_results:
        name = agent_res.get("agent_name", "Unknown")
        score_data = all_scores_map.get(name, {})
        
        # Breakdown steps with consensus
        steps = agent_res.get("steps", [])
        consensus_vals = step_consensus.get(name, [0.0]*len(steps))
        
        steps_with_flags = []
        for i, step in enumerate(steps):
            cons = consensus_vals[i] if i < len(consensus_vals) else 0.0
            steps_with_flags.append({
                "step_content": step,
                "consensus_score": round(cons, 2),
                "is_hallucination_risk": cons < 0.4
            })
            
        agents_report.append({
            "agent_name": name,
            "final_answer": agent_res.get("final_answer"),
            "steps_analysis": steps_with_flags,
            "metrics": {
                "symbolic_score": score_data.get("breakdown", {}).get("sym", 0.0),
                "logical_score": score_data.get("breakdown", {}).get("logic", 0.0),
                "clf_score": score_data.get("breakdown", {}).get("clf", 0.0),
                "total_score": score_data.get("score", 0.0) # This is final_conf usually
            }
        })

    # 4. Teacher Explanation
    best_agent_name = final_choice.get("best_agent", "None")
    verdict = final_choice.get("final_verdict", "UNKNOWN")
    reasoning = final_choice.get("winning_reasoning", "No detailed reasoning provided.")
    
    explanation_parts = [
        f"The system has analyzed the solution using multiple agents and determined the result is {verdict}.",
        f"The most reliable analysis came from {best_agent_name}.",
        f"Reasoning: {reasoning}"
    ]
    
    # Add note about low consensus if relevant
    hallucination_risks = [
        s for a in agents_report for s in a["steps_analysis"] if s["is_hallucination_risk"]
    ]
    if hallucination_risks:
        explanation_parts.append(
            f"Note: {len(hallucination_risks)} steps were flagged as potential logical jumps or hallucinations (low consensus)."
        )
        
    teacher_explanation = "\n\n".join(explanation_parts)

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0.0"
        },
        "input": input_report,
        "canonical_representation": canonical_report,
        "multi_agent_analysis": agents_report,
        "final_decision": {
            "verdict": verdict,
            "confidence": final_choice.get("confidence_score", 0.0),
            "chosen_agent": best_agent_name,
            "teacher_explanation": teacher_explanation
        }
    }

async def generate_full_report(
    input_type: str,
    raw_input: Any,
    scoring_result: Dict[str, Any],
    details: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Adapter to call build_verification_report from valid `details`.
    """
    # Extract pieces from "details" blob passed by main.py
    ocr_data = {
        "raw_text": details.get("ocr_text", raw_input if input_type=="text" else ""),
        "confidence": details.get("ocr_confidence", 1.0 if input_type=="text" else 0.0)
    }
    
    canonical = details.get("structure", {})
    
    # verification->llm->details is the list of agent results
    agent_results = details.get("verification", {}).get("llm", {}).get("details", [])
    
    # We might need to re-compute consensus if it wasn't passed, 
    # but efficiently main.py should pass it. 
    # For now, we stub or re-use what we have. 
    # Ideally, main.py should pass `step_consensus` in details if available.
    # If not available, we send empty dict.
    step_consensus = details.get("step_consensus", {})
    
    return build_verification_report(
        problem_id="prob_" + datetime.now().strftime("%H%M%S"),
        ocr_output=ocr_data,
        canonical_expr=canonical,
        agent_results=agent_results,
        step_consensus=step_consensus,
        scores=scoring_result, # This contains "all_scores"
        final_choice=scoring_result
    )
