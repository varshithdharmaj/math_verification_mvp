import sys
import os

sys.path.insert(0, r'c:\Users\Varshith Dharmaj\Downloads\major')

print("--- Testing Bug 2: Agent Diversity ---")
from llm_agent import LLMAgent
problem = "2x + 4 = 10"
answers = {}
for name in ["GPT-4", "Llama 3", "Gemini 2.0 Pro", "Qwen-2.5-Math-7B"]:
    agent = LLMAgent(name, use_real_api=False)
    result = agent.generate_solution(problem)
    answers[name] = result["final_answer"]
    print(f"{name}: {result['final_answer']} | Steps: {result['reasoning_trace'][:1]}")

print()
unique = set(answers.values())
print(f"Unique answers: {unique}  (should be diverse, not just '42' or '5')")

print("\n--- Testing Bug 3: Consensus Logic ---")
from consensus_fusion import evaluate_consensus
responses = [{'agent': n, 'response': {'Answer': a, 'Reasoning Trace': ['step1','step2','step3'], 'Confidence Explanation': f'Answered by {n}'}} for n, a in answers.items()]
result = evaluate_consensus(responses)
print()
print('Verdict:', result.get('verdict'))
print('Has divergence:', result.get('has_divergence'))
print('Hallucination alerts:', len(result.get('hallucination_alerts', [])))
print('Final answer:', result.get('final_verified_answer'))
