import time
import asyncio
import sys
import os

# Ensure the root of the project is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import run_verification_parallel

GSM8K_MOCK = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "steps": ["Natalia sold 48 clips in April", "She sold 48 / 2 = 24 clips in May", "Total = 48 + 24 = 72 clips."],
        "answer": "72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "steps": ["Weng earns 12 per 60 minutes", "For 50 minutes, she earns 12 * (50/60) = 10"],
        "answer": "10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "steps": ["Betty needs 100", "She has 100 / 2 = 50", "Parents give 15", "Grandparents give 15 * 2 = 30", "Total she has is 50 + 15 + 30 = 95", "She still needs 100 - 95 = 5"],
        "answer": "5"
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "steps": ["Total is 120", "Read 12 yesterday", "Read 12 * 2 = 24 today", "Remaining is 120 - 12 - 24 = 84", "Half of remaining is 84 / 2 = 42 pages"],
        "answer": "42"
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "steps": ["3 pages * 2 friends = 6 pages per time", "6 pages * twice a week = 12 pages per week", "12 pages * 52 weeks = 624 pages"],
        "answer": "624"
    }
]

async def evaluate_problem_in_memory(problem: str, steps: list, expected_answer: str):
    # Consume the parallel verification engine generator
    result = None
    for partial_res in run_verification_parallel(problem, steps, model_name="Ensemble", model_list=["GPT-4"]):
        if partial_res["type"] == "final":
            result = partial_res
    
    # We define correctly verified if the verdict is VALID
    consensus = result.get("consensus", {})
    verdict = consensus.get("final_verdict", "ERROR")
    latency = result.get("processing_time", 0.0)
    
    if verdict != "VALID":
        print(f"    [DEBUG] Verdict: {verdict}")
        print(f"    [DEBUG] Consensus data: {consensus}")
        print(f"    [DEBUG] Errors: {result.get('classified_errors', [])}")
        
    is_correct = verdict == "VALID"
    return is_correct, latency

async def run_benchmark():
    num_samples = len(GSM8K_MOCK)
    correct_count = 0
    latencies = []
    
    print(f"Running evaluation on {num_samples} hardcoded GSM8K samples...\n")
    for i, sample in enumerate(GSM8K_MOCK):
        print(f"[{i+1}/{num_samples}] Evaluating: {sample['question'][:40]}...")
        is_correct, lat = await evaluate_problem_in_memory(sample["question"], sample["steps"], sample["answer"])
        if is_correct: correct_count += 1
        latencies.append(lat)
        print(f"  -> Result: {'✅ Correct' if is_correct else '❌ Failed'} | Latency: {lat:.4f}s")
        
    accuracy = (correct_count / num_samples) * 100
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    print("\n" + "="*40)
    print("🏆 SYSTEM PERFORMANCE METRICS")
    print("="*40)
    print(f"Total Samples Tested: {num_samples}")
    print(f"Accuracy: {accuracy:.1f}%  (Target: >= 71%)")
    print(f"Average Latency: {avg_latency:.4f}s per problem (Target: < 5s)")
    print(f"Runtime Constraints: {'✅ PASS' if avg_latency < 5 else '❌ FAIL'}")
    print(f"Accuracy Threshold: {'✅ PASS' if accuracy >= 71 else '❌ FAIL'}")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(run_benchmark())

