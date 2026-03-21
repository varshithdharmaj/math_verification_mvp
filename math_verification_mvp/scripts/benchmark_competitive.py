import time
import asyncio
import sys
import os

# Ensure the root of the project is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.verification_engine import run_verification_parallel

COMPETITIVE_EXAM_MOCK = [
    {
        "exam": "GATE (CS) - Linear Algebra",
        "question": "Let M be a 2x2 matrix such that M = [[4, 1], [2, 3]]. Find the sum of the eigenvalues of M.",
        "steps": [
            "The sum of the eigenvalues of a matrix is equal to its trace.",
            "Trace(M) = 4 + 3",
            "Trace(M) = 7",
            "Therefore, the sum of the eigenvalues is 7."
        ],
        "answer": "7"
    },
    {
        "exam": "JEE Advanced - Calculus",
        "question": "Evaluate the definite integral of x * e^x from x=0 to x=1.",
        "steps": [
            r"Use integration by parts: \int u dv = uv - \int v du",
            r"Let u = x, so du = dx. Let dv = e^x dx, so v = e^x.",
            r"\int_0^1 x e^x dx = [x e^x]_0^1 - \int_0^1 e^x dx",
            "= (1 * e^1 - 0 * e^0) - [e^x]_0^1",
            "= e - (e^1 - e^0)",
            "= e - e + 1 = 1",
            "The final evaluated definite integral is 1."
        ],
        "answer": "1"
    },
    {
        "exam": "GATE (EC) - Probability",
        "question": "A box contains 4 red balls and 6 black balls. Three balls are drawn at random without replacement. What is the probability that exactly two are red?",
        "steps": [
            "Total ways to draw 3 balls from 10 is C(10,3).",
            "C(10,3) = (10*9*8)/(3*2*1) = 120",
            "Ways to draw exactly 2 red balls from 4 is C(4,2) = 6.",
            "Ways to draw 1 black ball from 6 is C(6,1) = 6.",
            "Total favorable ways = 6 * 6 = 36.",
            "Probability = 36 / 120 = 3 / 10 = 0.3."
        ],
        "answer": "0.3"
    },
    {
        "exam": "JEE Mains - Kinematics Paradox",
        "question": "A particle moves such that its velocity v is given by v = t^2 - 4t + 3. Find the acceleration when velocity is zero.",
        "steps": [
            "Velocity v = t^2 - 4t + 3 = 0",
            "(t - 1)(t - 3) = 0",
            "So t = 1 or t = 3.",
            "Acceleration a = dv/dt = 2t - 4.",
            "At t = 1, a = 2(1) - 4 = -2.",
            "At t = 3, a = 2(3) - 4 = 2.",
            "The accelerations are -2 and 2."
        ],
        "answer": "2" # or -2, testing logic handling branching paths
    },
    {
        "exam": "GATE (ME) - Differential Equations",
        "question": "Solve the initial value problem dy/dx = 2xy, y(0) = 1. Find y at x = 1.",
        "steps": [
            "dy/y = 2x dx",
            r"\int dy/y = \int 2x dx",
            "ln(y) = x^2 + C",
            "Use y(0) = 1: ln(1) = 0 + C => C = 0",
            "ln(y) = x^2 => y = e^(x^2)",
            "At x = 1, y = e^(1^2) = e."
        ],
        "answer": "e"
    }
]

def evaluate_competitive_problem(item: dict):
    problem = item["question"]
    steps = item["steps"]
    expected = item["answer"]
    exam = item["exam"]
    
    print(f"\n[EVALUATING] {exam}")
    print(f"Problem: {problem[:80]}...")
    
    # Consume generator to get final consensus
    result = None
    # We use all 4 models to simulate max rigorous verification for competitive exams
    active_models = ["GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3"]
    
    for partial_res in run_verification_parallel(problem, steps, model_name="Ensemble", model_list=active_models):
        if partial_res["type"] == "final":
            result = partial_res
            
    consensus = result.get("consensus", {})
    verdict = consensus.get("final_verdict", "ERROR")
    latency = result.get("processing_time", 0.0)
    confidence = consensus.get("overall_confidence", 0.0)
    
    is_correct = verdict == "VALID"
    
    if not is_correct:
        print(f"    [ATTENTION] System flagged errors in logic: {result.get('classified_errors', [])}")
        
    return is_correct, latency, confidence

def run_competitive_benchmark():
    num_samples = len(COMPETITIVE_EXAM_MOCK)
    correct_count = 0
    latencies = []
    confidences = []
    
    print("="*60)
    print("🎓 MVM² ADVANCED COMPETITIVE EXAM BENCHMARK (GATE / JEE)")
    print("="*60)
    print(f"Total problems queued: {num_samples}")
    
    for item in COMPETITIVE_EXAM_MOCK:
        is_correct, lat, conf = evaluate_competitive_problem(item)
        if is_correct: 
            correct_count += 1
        latencies.append(lat)
        confidences.append(conf)
        print(f"  -> Result: {'✅ VERIFIED' if is_correct else '❌ FLAGGED'} | Confidence: {conf*100:.1f}% | Latency: {lat:.3f}s")
        
    accuracy = (correct_count / num_samples) * 100
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print("\n" + "="*60)
    print("🏆 FINAL COMPETITIVE BENCHMARK METRICS")
    print("="*60)
    print(f"Advanced Exam Accuracy: {accuracy:.1f}% (Expected > 85%)")
    print(f"Average Confidence:     {avg_conf*100:.1f}%")
    print(f"Average Latency:        {avg_latency:.3f}s")
    print("="*60)

if __name__ == "__main__":
    # Ensure UTF-8 output
    sys.stdout.reconfigure(encoding='utf-8')
    run_competitive_benchmark()
