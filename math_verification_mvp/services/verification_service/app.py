from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import httpx
import sympy as sp
import re
import os
import json
import google.generativeai as genai

app = FastAPI(title="Verification Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOWNSTREAM_CLASSIFIER_URL = "http://classifier-service:8005/classify"

class VerificationRequest(BaseModel):
    problem: str
    steps: List[str]
    format: str
    raw_problem: str
    raw_steps: List[str]
    out_of_scope: bool
    difficulty: str
    metadata: Optional[Dict[str, Any]] = {}

class MultiAgentReasoner:
    def __init__(self, configs: List[Dict]):
        self.configs = configs
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        
    async def verify(self, problem: str, steps: List[str]) -> List[Dict[str, Any]]:
        results = []
        for config in self.configs:
            prompt = self._build_prompt(config, problem, steps)
            
            # Route to appropriate LLM
            model_provider = config.get("provider", "gemini")
            try:
                if model_provider == "gemini":
                    response_text = await self._call_gemini(prompt)
                elif model_provider == "openai":
                    response_text = await self._call_openai(prompt)
                elif model_provider == "anthropic":
                    response_text = await self._call_anthropic(prompt)
                else:
                    response_text = "Unsupported provider."
            except Exception as e:
                response_text = f'{{"final_answer": null, "reasoning": "Error: {str(e)}", "steps": []}}'
                
            parsed = self._parse_json_response(response_text)
            parsed["agent_name"] = config["name"]
            parsed["valid"] = parsed.get("final_answer") is not None
            parsed["confidence"] = 0.9 if parsed["valid"] else 0.5
            results.append(parsed)
            
        return results

    def _build_prompt(self, config: Dict, problem: str, steps: List[str]) -> str:
        role = config.get("type", "solver")
        steps_text = chr(10).join(f"{i+1}. {s}" for i, s in enumerate(steps))
        
        if role == "critic":
            return f"""
            You are a rigorous Math Critic. Review the following solution for errors.
            Problem: {problem}
            Proposed Steps:
            {steps_text}
            
            Return ONLY a JSON object:
            {{
                "final_answer": "valid" or "invalid",
                "reasoning": "your critique",
                "steps": ["step1 status", "step2 status"]
            }}
            """
        else: # solver or verifier
            return f"""
            Solve the problem step-by-step and verify the provided steps.
            Problem: {problem}
            Reference Steps:
            {steps_text}
            
            Return ONLY a JSON object:
            {{
                "final_answer": "the final result",
                "steps": ["corrected step 1", "corrected step 2"],
                "reasoning": "brief explanation"
            }}
            """

    async def _call_gemini(self, prompt: str) -> str:
        if not self.gemini_key: return '{"error": "Offline (No API Key)"}'
        genai.configure(api_key=self.gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
        resp = model.generate_content(prompt)
        return resp.text

    async def _call_openai(self, prompt: str) -> str:
        if not self.openai_key: return '{"error": "Offline (No API Key)"}'
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.openai_key}"}
            data = {
                "model": "gpt-4-turbo",
                "messages": [{"role": "user", "content": prompt}]
            }
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _call_anthropic(self, prompt: str) -> str:
        if not self.anthropic_key: return '{"error": "Offline (No API Key)"}'
        async with httpx.AsyncClient() as client:
            headers = {"x-api-key": self.anthropic_key, "anthropic-version": "2023-06-01"}
            data = {
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            }
            resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

    def _parse_json_response(self, text: str) -> Dict:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        try:
            return json.loads(clean_text)
        except:
            return {"final_answer": None, "raw_response": text, "reasoning": "Could not parse JSON"}

async def verify_steps_with_sympy(steps: List[str]) -> List[Dict]:
    errors = []
    for i, step in enumerate(steps):
        pattern = r'(\d+\.?\d*)\s*([+\-*/×÷^])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)'
        for match in re.findall(pattern, step):
            a, op, b, stated_result = match
            try:
                norm_op = op.replace('×', '*').replace('÷', '/')
                correct = float(a) ** float(b) if norm_op == '^' else eval(f"{a}{norm_op}{b}")
                if abs(float(stated_result) - correct) > 0.001:
                    errors.append({"step": i+1, "type": "arithmetic", "msg": f"{a}{op}{b} should be {correct}"})
            except Exception:
                pass
    return errors

def calculate_step_similarity(step_a: str, step_b: str) -> float:
    if step_a == step_b: return 1.0
    s_a, s_b = step_a.replace(" ", ""), step_b.replace(" ", "")
    if s_a == s_b: return 1.0
    try:
        if "=" in step_a and "=" in step_b:
            lhs_a, rhs_a = step_a.split("=", 1)
            lhs_b, rhs_b = step_b.split("=", 1)
            expr_a = sp.sympify(f"({lhs_a}) - ({rhs_a})")
            expr_b = sp.sympify(f"({lhs_b}) - ({rhs_b})")
            if sp.simplify(expr_a - expr_b) == 0: return 1.0
            if sp.simplify(expr_a + expr_b) == 0: return 0.7
        return 0.3
    except:
        return 0.3

def generate_divergence_matrix(all_agent_steps: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Formally computes the inter-agent divergence matrix as requested by the user and paper.
    """
    agents = list(all_agent_steps.keys())
    matrix = {}
    
    for agent_x in agents:
        matrix[agent_x] = {}
        steps_x = all_agent_steps[agent_x]
        for agent_y in agents:
            steps_y = all_agent_steps[agent_y]
            diff_scores = []
            
            # Compare each step in X to best match in Y
            for step_x in steps_x:
                best_sim = 0.0
                for step_y in steps_y:
                    sim = calculate_step_similarity(step_x, step_y)
                    if sim > best_sim: best_sim = sim
                # Divergence is 1 - similarity
                diff_scores.append(1.0 - best_sim)
                
            avg_divergence = sum(diff_scores) / len(diff_scores) if diff_scores else 1.0
            matrix[agent_x][agent_y] = round(avg_divergence, 3)
            
    return matrix

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "verification"}

@app.post("/verify")
async def verify_endpoint(request: VerificationRequest):
    """
    Accepts canonicalized representation, calls multiple LLM agents, 
    calculates SymPy errors and Divergence Matrices, and forwards to Classifier Service.
    """
    if request.out_of_scope:
        payload = {
            "out_of_scope": True,
            "reason": "Expression outside restricted segment",
            "sympy_valid": False,
            "llm_details": [],
            "divergence_matrix": {},
            "metadata": request.metadata
        }
    else:
        # SymPy Check
        sympy_errors = await verify_steps_with_sympy(request.steps)
        sympy_valid = len(sympy_errors) == 0
        
        # Multi-Agent Configuration (Integrating paper requirements: GPT-4, Claude, Gemini)
        agents_config = [
            {"name": "GPT-4 Solver", "type": "solver", "provider": "openai"},
            {"name": "Claude Critic", "type": "critic", "provider": "anthropic"},
            {"name": "Gemini Verifier", "type": "solver", "provider": "gemini"}
        ]
        
        reasoner = MultiAgentReasoner(agents_config)
        agent_results = await reasoner.verify(request.problem, request.steps)
        
        # Generate formal divergence matrix
        agent_steps_map = {res["agent_name"]: res.get("steps", []) for res in agent_results if res.get("steps")}
        divergence_matrix = generate_divergence_matrix(agent_steps_map)
        
        payload = {
            "out_of_scope": False,
            "sympy_valid": sympy_valid,
            "sympy_errors": sympy_errors,
            "llm_details": agent_results,
            "divergence_matrix": divergence_matrix,
            "metadata": request.metadata
        }
        
    # Forward to Classifier Service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(DOWNSTREAM_CLASSIFIER_URL, json=payload, timeout=90.0)
            response.raise_for_status()
            return response.json()
            
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Downstream Classifier service unavailable: {exc}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=f"Downstream Classifier service error: {exc.response.text}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
