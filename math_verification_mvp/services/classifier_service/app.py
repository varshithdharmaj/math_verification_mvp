from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import httpx
import os

app = FastAPI(title="Classifier Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOWNSTREAM_REPORTING_URL = "http://reporting-service:8006/report"

# We define 10 rigorous error classification types:
ERROR_TYPES = [
    "Arithmetic Error",      # Caught by SymPy checks evaluating basic ops
    "Sign Error",            # Subset of Arithmetic/Algebraic where only the sign is flipped
    "Copying / OCR Error",   # High OCR uncertainty leading to garbage states
    "Logical Jump",          # High divergence across agents, missing intermediate steps
    "Syntax Error",          # SymPy failed to parse entirely
    "Formula Error",         # Applied wrong formula (e.g. Area = 2*pi*r)
    "Substitution Error",    # Plugged in wrong values into a correct formula
    "Unsimplified Form",     # Correct algebraically but not final (e.g. 4/2 instead of 2)
    "Out of Scope",          # Non-math query
    "Final Answer Mismatch", # Agents diverged at the very last step
    "Unknown / Unscorable"   # Blanket fallback
]

class ClassificationRequest(BaseModel):
    out_of_scope: bool
    sympy_valid: bool
    sympy_errors: List[Dict[str, Any]]
    llm_details: List[Dict[str, Any]]
    divergence_matrix: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = {}

def compute_symbolic_score(agent_result: Dict) -> float:
    valid_steps = agent_result.get("valid", False)
    return 1.0 if valid_steps else 0.0

def compute_logical_score(agent_result: Dict) -> float:
    score = 1.0
    if not agent_result.get("final_answer"): score -= 0.5
    reasoning = agent_result.get("reasoning", "").lower()
    if any(k in reasoning for k in ["unknown", "error", "cannot solve", "invalid"]): score -= 0.3
    if not agent_result.get("steps") and agent_result.get("final_answer"): score -= 0.2
    return max(0.0, score)

def determine_error_category(agent: Dict, request: ClassificationRequest, avg_consensus: float) -> str:
    """Explicit multi-class error routing based on heuristics and SymPy matrices."""
    
    if request.out_of_scope:
        return "Out of Scope"
        
    ocr_conf = request.metadata.get("ocr_confidence", 1.0)
    if ocr_conf < 0.6:
        return "Copying / OCR Error"
        
    # Check SymPy formal verification errors
    if request.sympy_errors:
        err_msg = str(request.sympy_errors).lower()
        if "syntax" in err_msg or "parse" in err_msg:
            return "Syntax Error"
        if "sign" in err_msg or "-" in err_msg:
            return "Sign Error"
        return "Arithmetic Error"
        
    # High divergence (hallucination) across agents -> Logical Jump
    if avg_consensus < 0.4:
        return "Logical Jump"
        
    reasoning = agent.get("reasoning", "").lower()
    
    # NLP keyword heuristics based on Critic Agent feedbacks
    if "formula" in reasoning or "theorem" in reasoning:
        return "Formula Error"
    if "substituted" in reasoning or "plugged" in reasoning:
        return "Substitution Error"
    if "simplify" in reasoning or "reduce" in reasoning:
        return "Unsimplified Form"
        
    if not agent.get("final_answer"):
        return "Final Answer Mismatch"
        
    return "Unknown / Unscorable"

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "classifier"}

@app.post("/classify")
async def classify_endpoint(request: ClassificationRequest):
    """
    Combines SymPy scores and LLM reasoning matrices into a final verdict, 
    and formally categorizes the errors into 10+ logical types.
    """
    if request.out_of_scope:
        payload = {
            "final_verdict": "OUT_OF_SCOPE",
            "confidence_score": 0.0,
            "error_category": "Out of Scope",
            "best_agent": "None",
            "metadata": request.metadata
        }
    else:
        ocr_conf = request.metadata.get("ocr_confidence", 1.0)
        scored_agents = []
        
        for agent_res in request.llm_details:
            name = agent_res.get("agent_name", "Unknown")
            
            sym = compute_symbolic_score(agent_res)
            logic = compute_logical_score(agent_res)
            
            # Consensus from matrix
            divergences = request.divergence_matrix.get(name, {}).values()
            consensuses = [1.0 - d for d in divergences]
            avg_cons = sum(consensuses) / len(consensuses) if consensuses else 0.0
            
            # Weighted Scoring Engine
            raw_score = (0.4 * sym) + (0.35 * logic) + (0.25 * avg_cons)
            final_conf = raw_score * (0.9 + 0.1 * ocr_conf)
            
            error_category = "None"
            if final_conf <= 0.6:
                error_category = determine_error_category(agent_res, request, avg_cons)
            
            scored_agents.append({
                "agent": name,
                "final_conf": final_conf,
                "error_category": error_category,
                "components": {"sym": sym, "logic": logic, "consensus": avg_cons},
                "data": agent_res
            })
            
        if not scored_agents:
            payload = {"final_verdict": "ERROR", "confidence_score": 0.0, "error_category": "Unknown / Unscorable", "metadata": request.metadata}
        else:
            best_agent = max(scored_agents, key=lambda x: x["final_conf"])
            is_valid = best_agent["final_conf"] > 0.6
            
            payload = {
                "final_verdict": "VALID" if is_valid else "ERROR",
                "confidence_score": round(best_agent["final_conf"], 3),
                "error_category": best_agent["error_category"],
                "best_agent": best_agent["agent"],
                "final_answer": best_agent.get("data", {}).get("final_answer", ""),
                "all_scores": [{"name": a["agent"], "score": round(a["final_conf"], 3), "breakdown": a["components"], "error": a["error_category"]} for a in scored_agents],
                "winning_reasoning": best_agent["data"].get("reasoning", ""),
                "divergence_matrix": request.divergence_matrix,
                "metadata": request.metadata
            }

    # Forward to Reporting Service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(DOWNSTREAM_REPORTING_URL, json=payload, timeout=60.0)
            response.raise_for_status()
            return response.json()
            
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Downstream Reporting service unavailable: {exc}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail="Downstream Reporting service error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
