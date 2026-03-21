"""
Quick example script to test the verification system
"""

from core import run_verification_parallel

# Example problem
problem = "Janet has 3 apples. She buys 2 more. She gives 1 away. How many?"
steps = [
    "Janet starts with 3 apples",
    "She buys 2 more: 3 + 2 = 5 apples",
    "She gives 1 away: 5 - 1 = 6 apples"  # ERROR: should be 4
]

print("Running verification...")
print(f"Problem: {problem}")
print(f"Steps: {steps}")
print("\n" + "="*50 + "\n")

result = run_verification_parallel(
    problem=problem,
    steps=steps,
    model_name="GPT-4",
    model_list=["GPT-4", "Llama 2", "Gemini"]
)

# Display results
consensus = result["consensus"]
print(f"Final Verdict: {consensus['final_verdict']}")
print(f"Confidence: {consensus['overall_confidence'] * 100:.1f}%")
print(f"Agreement: {consensus['agreement_type']}")
print(f"Processing Time: {result['processing_time']:.2f}s")
print("\n" + "="*50 + "\n")

# Display errors
classified_errors = result.get("classified_errors", [])
if classified_errors:
    print(f"Found {len(classified_errors)} error(s):\n")
    for error in classified_errors:
        print(f"Step {error['step_number']}: {error['category']}")
        print(f"  Found: {error.get('found', 'N/A')}")
        print(f"  Correct: {error.get('correct', 'N/A')}")
        
        explanations = result.get("explanations", {})
        step_num = error.get("step_number", 0)
        if step_num in explanations:
            print(f"  Explanation: {explanations[step_num]}")
        print()
else:
    print("No errors found! Solution is valid.")

