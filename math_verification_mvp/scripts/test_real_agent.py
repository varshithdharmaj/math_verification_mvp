import asyncio
import os
import sys
import json
import httpx
import google.generativeai as genai

sys.path.append(os.path.abspath("."))
import json
import httpx
import google.generativeai as genai

# We bypass the full local microservice setup to directly test the LLM verification logic
from services.verification_service.app import MultiAgentReasoner, generate_divergence_matrix

async def test_real_agent():
    # User provided Gemini key earlier in the conversation
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyDFSVZa9rSCb_2XcftaKsaRT5n3Qcj8Z_w")
    os.environ["GEMINI_API_KEY"] = api_key
    
    # Configure so that all agents use Gemini for this test since we only have the Gemini key
    agents_config = [
        {"name": "Gemini Solver 1", "type": "solver", "provider": "gemini"},
        {"name": "Gemini Critic", "type": "critic", "provider": "gemini"},
        {"name": "Gemini Solver 2", "type": "solver", "provider": "gemini"}
    ]
    
    reasoner = MultiAgentReasoner(agents_config)
    # The MultiAgentReasoner uses os.getenv internally, let's just make sure it picks it up
    reasoner.gemini_key = api_key
    
    problem = "John has 5 apples. He buys 3 more and gives 2 away. How many does he have?"
    steps = ["John starts with 5.", "He buys 3, so 5 + 3 = 8.", "He gives 2 away, so 8 - 2 = 6."]
    
    print(f"Testing problem: {problem}")
    print("Calling Gemini API...")
    
    results = await reasoner.verify(problem, steps)
    
    print("\n--- AGENT RESULTS ---")
    print(json.dumps(results, indent=2))
    
    agent_steps_map = {res["agent_name"]: res.get("steps", []) for res in results if res.get("steps")}
    matrix = generate_divergence_matrix(agent_steps_map)
    
    print("\n--- DIVERGENCE MATRIX ---")
    print(json.dumps(matrix, indent=2))

if __name__ == "__main__":
    asyncio.run(test_real_agent())
