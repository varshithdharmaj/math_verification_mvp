from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
from datetime import datetime

app = FastAPI(title="Reporting Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReportingRequest(BaseModel):
    final_verdict: str
    confidence_score: float
    error_category: str
    best_agent: str
    final_answer: str = ""
    all_scores: List[Dict[str, Any]] = []
    winning_reasoning: str = ""
    divergence_matrix: Dict[str, Any] = {}
    metadata: Optional[Dict[str, Any]] = {}

def generate_teacher_explanation(request: ReportingRequest, agents_report: List[Dict]) -> str:
    """Generates a comprehensive pedagogical explanation based on consensus and error mapping."""
    
    parts = []
    
    if request.final_verdict == "VALID":
        parts.append("✅ **Solution Accepted.** The mathematical steps adhere to formal logic without significant divergence.")
        parts.append(f"The most reliable reasoning track was provided by **{request.best_agent}** with a confidence score of {request.confidence_score * 100:.1f}%.")
    else:
        parts.append("❌ **Solution Rejected.** The system detected flaws in the reasoning or mathematical execution.")
        
        # Categorical mapping explanation
        cat = request.error_category
        if cat == "Arithmetic Error":
            parts.append("There was a basic arithmetic calculation mistake detected during the Symbolic (SymPy) evaluation step.")
        elif cat == "Syntax Error":
            parts.append("The mathematical string could not be parsed. There may be malformed LaTeX or invalid syntax.")
        elif cat == "Logical Jump":
            parts.append("The AI agents diverged significantly at an intermediate step, indicating a missing logical connection or hallucination.")
        elif cat == "Formula Error":
            parts.append("An incorrect formula or theorem was applied to the problem statement.")
        elif cat == "Substitution Error":
            parts.append("Values were incorrectly substituted into the formulas.")
        elif cat == "Copying / OCR Error":
            parts.append("The OCR engine struggled to confidently read the input. A visual noise or handwriting ambiguity likely disrupted the parsing.")
        elif cat == "Sign Error":
            parts.append("A negative/positive sign error occurred during algebraic manipulation.")
        elif cat == "Out of Scope":
            parts.append("The provided text does not contain a recognizable mathematical expression.")
        elif cat == "Final Answer Mismatch":
            parts.append("The steps were mostly consistent, but the agents arrived at completely different final answers.")
        elif cat == "Unsimplified Form":
            parts.append("The result is mathematically correct but not in its simplest canonical form.")
        else:
            parts.append(f"The solution was flagged as Error Type: **{cat}**.")
            
    # Include Hallucination Risks
    hallucinations = [s for a in agents_report for s in a.get("steps_analysis", []) if s.get("is_hallucination_risk")]
    if hallucinations:
        parts.append(f"\n*Pedagogical Note:* {len(hallucinations)} intermediate steps showed extremely low consensus across models. These steps are considered high-risk for AI hallucinations or missing human work.")
        
    parts.append(f"\n*Winning Agent's Chain of Thought:*\n{request.winning_reasoning}")
    
    return "\n\n".join(parts)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "reporting"}

@app.post("/report")
async def report_endpoint(request: ReportingRequest):
    """
    Receives final payload from Classifier Service, invokes the Explanation Generator, 
    and structures it for the frontend Dashboard.
    """
    metadata = request.metadata or {}
    
    agents_report = []
    
    for agent_data in request.all_scores:
        name = agent_data.get("name")
        score = agent_data.get("score")
        breakdown = agent_data.get("breakdown", {})
        error_cat = agent_data.get("error", "None")
        
        divergences = request.divergence_matrix.get(name, {})
        
        steps_analysis = []
        if divergences:
            for step_idx, div in enumerate(divergences.values()):
                cons = 1.0 - div
                steps_analysis.append({
                    "step_idx": step_idx + 1,
                    "consensus_score": round(cons, 2),
                    "is_hallucination_risk": cons < 0.4
                })
                
        agents_report.append({
            "agent_name": name,
            "error_category_flagged": error_cat,
            "steps_analysis": steps_analysis,
            "metrics": {
                "symbolic_score": breakdown.get("sym", 0.0),
                "logical_score": breakdown.get("logic", 0.0),
                "consensus_score": breakdown.get("consensus", 0.0),
                "total_score": score
            }
        })
        
    final_report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "2.0.0-microservices"
        },
        "input": {
            "ocr_confidence": metadata.get("ocr_confidence", 1.0),
            "ocr_method": metadata.get("ocr_method", "Unknown"),
            "original_metadata": metadata.get("raw_metadata")
        },
        "multi_agent_analysis": agents_report,
        "inter_agent_divergence_matrix": request.divergence_matrix,
        "final_decision": {
            "verdict": request.final_verdict,
            "error_category": request.error_category,
            "confidence": request.confidence_score,
            "chosen_agent": request.best_agent,
            "final_answer": request.final_answer,
            "explanation": generate_teacher_explanation(request, agents_report)
        }
    }
    
    return final_report

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
