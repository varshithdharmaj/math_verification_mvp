import os
import sys
import json

# Ensure we can import from services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.core_engine.agent_orchestrator import run_agent_orchestrator
from services.core_engine.consensus_module import evaluate_consensus
from services.reporting_service.report_generator import generate_mvm2_report

def run_numerical_noise_test():
    print("="*60)
    print("[TEST] MVM2 NUMERICAL NOISE DISTRACTOR TEST")
    print("="*60)
    
    # Problem with a blatant numeric distractor
    problem = "Solve 5 * 6. Ignore factor k=4.2 and noise attribute = 100."
    print(f"Problem: {problem}")
    
    # 1. Dispatch to Agents
    agent_responses = run_agent_orchestrator(problem)
    
    # 2. Run Consensus
    consensus_result = evaluate_consensus(agent_responses, ocr_confidence=0.95)
    
    # 3. Verify Result
    final_ans = consensus_result["final_verified_answer"]
    score = consensus_result["winning_score"]
    
    print("\n[VERDICT]")
    print(f"Verified Answer:   {final_ans}")
    print(f"Consensus Score:   {score:.3f}")
    
    if final_ans == "30":
        print("\n✅ SUCCESS: Consensus engine ignored distractors (4.2, 100) and found correct solution (30).")
    else:
        print(f"\n❌ FAILURE: Expected 30, but got {final_ans}.")
        
    # Generate report
    report = generate_mvm2_report(consensus_result, problem, 0.95)
    with open("noise_test_report.md", "w", encoding="utf-8") as f:
        f.write(report["markdown"])
    print("Report saved to noise_test_report.md")

if __name__ == "__main__":
    run_numerical_noise_test()
