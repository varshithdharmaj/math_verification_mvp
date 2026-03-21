import time
import logging
from typing import List, Dict, Any, Optional
from .agent_orchestrator import AGENT_PROFILES, simulate_agent_execution
from .consensus_module import evaluate_consensus

logger = logging.getLogger(__name__)

def run_neurosymbolic_pipeline_stream(
    problem: str,
    steps: Optional[List[str]] = None,
    model_name: str = "Ensemble",
    model_list: Optional[List[str]] = None
):
    """
    Adapter that executes the Phase 9 Core Neuro-Symbolic 4-Agent pipeline
    but yields asynchronous dictionary chunks formatted exactly as the
    Streamlit UI expects from the legacy system.
    """
    start_time = time.time()
    logger.info(f"Dispatching problem to 4 Phase 9 Agents in parallel...")
    
    agent_results = {}
    
    # We execute sequentially to avoid the WinError 6 thread crashing bug on Windows
    for agent_profile in AGENT_PROFILES:
        agent_name = agent_profile["name"]
        try:
            res = simulate_agent_execution(agent_profile, problem, steps=steps)
            raw_response = res["response"]
            agent_results[agent_name] = {
                "final_answer": raw_response.get("Answer", "ERROR"),
                "reasoning_trace": raw_response.get("Reasoning Trace", []),
                "confidence_explanation": raw_response.get("Confidence Explanation", "")
            }
        except Exception as exc:
            logger.error(f"Agent {agent_name} failed: {exc}")
            agent_results[agent_name] = {
                "final_answer": "ERROR",
                "reasoning_trace": [],
                "confidence_explanation": str(exc)
            }
            res = {
                "agent": agent_name,
                "response": {
                    "Answer": "ERROR",
                    "Reasoning Trace": [],
                    "Confidence Explanation": str(exc)
                }
            }
            
        # Yield partial result to stream to UI exactly as it expects
        yield {
            "type": "partial",
            "agent_name": agent_name,
            "agent_result": agent_results[agent_name]
        }
    
    # After all agents execute, compute true Phase 9 consensus (Math-Verify + QWED)
    # Reconstruct the raw responses format for evaluate_consensus
    raw_responses_for_consensus = []
    for a_name, a_result in agent_results.items():
        raw_responses_for_consensus.append({
            "agent": a_name,
            "response": {
                "Answer": a_result["final_answer"],
                "Reasoning Trace": a_result["reasoning_trace"],
                "Confidence Explanation": a_result["confidence_explanation"]
            }
        })
        
    phase9_consensus = evaluate_consensus(raw_responses_for_consensus)
    
    # Map the new phase 9 consensus output to the legacy UI schema
    # The UI expects decision.get("final_verdict") and decision.get("overall_confidence")
    decision = {
        "final_verdict": "VALID" if phase9_consensus["winning_score"] > 0.6 else "ERRONEOUS",
        "overall_confidence": phase9_consensus["winning_score"], # 0.0 to 1.0 mapped to UI
        "verified_answer": phase9_consensus["final_verified_answer"],
        "divergence_groups": phase9_consensus["divergence_groups"],
        "detail_scores": phase9_consensus["detail_scores"]
    }
    
    # Map any flagged errors based on low symbolic validation
    classified_errors = []
    for ds in phase9_consensus["detail_scores"]:
        if ds["V_sym"] < 0.5:
            classified_errors.append({
                "step_number": 0,
                "category": f"Symbolic Validation Failure ({ds['agent']})",
                "found": ds['raw_answer'],
                "correct": phase9_consensus["final_verified_answer"]
            })
            
    processing_time = time.time() - start_time
    
    # Yield final result
    yield {
        "type": "final",
        "problem": problem,
        "base_steps": steps, 
        "model_results": agent_results,
        "consensus": decision,
        "classified_errors": classified_errors,
        "processing_time": processing_time
    }
