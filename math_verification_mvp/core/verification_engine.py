import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import logging

from models.llm_agent import LLMAgent
from consensus.consensus_mechanism import compute_neurosymbolic_consensus

logger = logging.getLogger(__name__)

def run_verification_parallel(
    problem: str,
    steps: Optional[List[str]] = None,
    model_name: str = "Ensemble",
    model_list: Optional[List[str]] = None
):
    """
    Run verification with Multi-Agent LLMs in parallel.
    Yields intermediate results for UI streaming, and computes empirical neuro-symbolic consensus.
    """
    start_time = time.time()
    
    agent_names = model_list if model_list else ["Gemini 1.5 Pro", "GPT-4", "Claude 3.5 Sonnet", "Llama 3"]
    agents = []
    
    for name in agent_names:
        # Route ALL Multi-Agent logic through the genuine LLM backend defined in LLMAgent
        agents.append(LLMAgent(model_name=name, use_real_api=True))
        
    logger.info(f"Dispatching problem to {len(agents)} agents in parallel...")
    
    agent_results = {}
    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as executor:
        future_to_agent = {executor.submit(agent.generate_solution, problem): agent for agent in agents}
        
        for future in as_completed(future_to_agent):
            agent = future_to_agent[future]
            try:
                res = future.result()
                # Ensure the strict triplet format
                agent_results[agent.model_name] = {
                    "final_answer": res.get("final_answer", "ERROR"),
                    "reasoning_trace": res.get("reasoning_trace", []),
                    "confidence_explanation": res.get("confidence_explanation", "")
                }
            except Exception as exc:
                logger.error(f"Agent {agent.model_name} failed: {exc}")
                agent_results[agent.model_name] = {
                    "final_answer": "ERROR",
                    "reasoning_trace": [],
                    "confidence_explanation": str(exc)
                }
            
            # Yield partial result to stream to UI
            yield {
                "type": "partial",
                "agent_name": agent.model_name,
                "agent_result": agent_results[agent.model_name]
            }
    
    # Compute true Hybrid Consensus (SymPy + Divergence Matrix + Domain Weights)
    consensus_result = compute_neurosymbolic_consensus(agent_results)
    
    processing_time = time.time() - start_time
    
    # Yield final result
    yield {
        "type": "final",
        "problem": problem,
        "base_steps": steps, 
        "model_results": agent_results,
        "consensus": consensus_result,
        "processing_time": processing_time
    }
