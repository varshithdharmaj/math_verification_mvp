import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)

from core import run_verification_parallel

EDGE_CASES = [
    {
        "name": "Deliberate Self-Contradiction Paradox",
        "problem": "Prove that 1 = 2 using standard arithmetic.",
        "steps": [
            "Let a = b",
            "a^2 = ab",
            "a^2 - b^2 = ab - b^2",
            "(a-b)(a+b) = b(a-b)",
            "a+b = b",
            "Since a=b, 2b = b",
            "2 = 1"
        ]
    },
    {
        "name": "Calculus Ambiguity (Division by Zero limit)",
        "problem": "Evaluate the limit of 1/x as x approaches 0.",
        "steps": [
            "We want to find the limit of 1/x as x goes to 0.",
            "Plug in 0 for x.",
            "1 / 0 is infinity.",
            "Therefore the limit is infinity."
        ]
    }
]

def run_tests():
    for case in EDGE_CASES:
        print(f"\n======================================")
        print(f"🧪 Running Edge Case: {case['name']}")
        print(f"Problem: {case['problem']}")
        print(f"======================================")
        
        start = time.time()
        for partial_res in run_verification_parallel(case['problem'], case['steps']):
            # Stream the results logic to terminal
            if partial_res.get("type") == "partial":
                agent = partial_res["agent_name"]
                ans = partial_res["agent_result"]["final_answer"]
                print(f" [STREAM] {agent} finished analyzing. Conclusion: {ans}")
            elif partial_res.get("type") == "final":
                print("\n--- CONSENSUS RESULT ---")
                print(f"Verdict: {partial_res['consensus'].get('final_verdict')}")
                print(f"Confidence: {partial_res['consensus'].get('overall_confidence', 0)*100:.1f}%")
                errors = partial_res.get('consensus', {}).get('classified_errors', [])
                if errors:
                    print("Errors Caught:")
                    for err in errors:
                        print(f" - Step {err.get('step_number')}: {err.get('category')} (Found: {err.get('found')})")
                else:
                    print("No explicit errors caught.")
        
        print(f"   => Edge case resolved in {time.time() - start:.2f}s\n")

if __name__ == "__main__":
    run_tests()
