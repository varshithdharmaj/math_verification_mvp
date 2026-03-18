import concurrent.futures
import time
import json
from typing import Dict, List, Any, Optional

import llm_agent

# Active agents for the MVM2 Parallel reasoning layer.
# For Hugging Face Spaces we default all agents to simulated mode so that
# the demo does not depend on external API keys or outbound network access.
AGENT_PROFILES = [
    {"name": "GPT-4", "use_real_api": False},
    {"name": "Llama 3", "use_real_api": False},
    {"name": "Gemini 2.0 Pro", "use_real_api": False},
    {"name": "Qwen-2.5-Math-7B", "use_real_api": False},
]

def run_agent_orchestrator(problem: str) -> List[Dict[str, Any]]:
    """
    Dispatches the problem to heterogeneous LLM agents.
    """
    print(f"[Orchestrator] Dispatching to {len(AGENT_PROFILES)} Parallel Models...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_agent = {
            executor.submit(llm_agent.LLMAgent(agent["name"], use_real_api=agent["use_real_api"]).generate_solution, problem): agent 
            for agent in AGENT_PROFILES
        }
        
        for future in concurrent.futures.as_completed(future_to_agent):
            agent_info = future_to_agent[future]
            try:
                raw_res = future.result()
                
                normalized_res = {
                    "Answer": str(raw_res.get("final_answer", raw_res.get("Answer", "ERROR"))),
                    "Reasoning Trace": raw_res.get("reasoning_trace", raw_res.get("Reasoning Trace", [])),
                    "Confidence Explanation": raw_res.get("confidence_explanation", raw_res.get("Confidence Explanation", ""))
                }
                
                results.append({
                    "agent": agent_info["name"],
                    "response": normalized_res
                })
                print(f"[OK] {agent_info['name']} completed reasoning.")
            except Exception as exc:
                print(f"[ERROR] {agent_info['name']} generated an exception: {exc}")
                
    return results

if __name__ == "__main__":
    test_out = run_agent_orchestrator("\\int_{0}^{\\pi} \\sin(x^{2}) \\, dx")
    print(json.dumps(test_out, indent=2))
