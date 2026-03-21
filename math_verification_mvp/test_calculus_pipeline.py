import os
import sys
import time

# Ensure root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.core_engine.pipeline_streamer import run_neurosymbolic_pipeline_stream

def test_hard_problem():
    problem = "Evaluate the definite integral of x * e^x from x=0 to x=1."
    steps = [
        "Use integration by parts: integral u dv = uv - integral v du",
        "Let u = x, so du = dx. Let dv = e^x dx, so v = e^x.",
        "integral_0^1 x e^x dx = [x e^x]_0^1 - integral_0^1 e^x dx",
        "= (1 * e^1 - 0 * e^0) - [e^x]_0^1",
        "= e - (e^1 - e^0)",
        "= e - e + 1 = 1"
    ]
    
    print(f"Testing Problem: {problem}")
    print("Dispatching to MVM2 Ensemble...")
    
    results = None
    for partial_res in run_neurosymbolic_pipeline_stream(
        problem=problem,
        steps=steps,
        model_name="MVM2 Ensemble"
    ):
        if partial_res["type"] == "partial":
            agent = partial_res["agent_name"]
            ans = partial_res["agent_result"].get("final_answer")
            print(f" [OK] Agent {agent} produced answer: {ans}")
        elif partial_res["type"] == "final":
            results = partial_res

    if results:
        consensus = results.get("consensus", {})
        print("\n" + "="*40)
        print("FINAL VERIFICATION RESULT")
        print("="*40)
        print(f"Verdict:    {consensus.get('final_verdict')}")
        print(f"Answer:     {consensus.get('verified_answer')}")
        print(f"Score:      {consensus.get('overall_confidence'):.3f}")
        print(f"Confidence: {consensus.get('overall_confidence', 0)*100:.1f}%")
        print("="*40)
    else:
        print("Failed to get results from pipeline.")

if __name__ == "__main__":
    test_hard_problem()
