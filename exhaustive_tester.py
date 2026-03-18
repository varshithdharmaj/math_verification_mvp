import os
import json
import time
import sys
import re
from typing import List, Dict, Any

# Ensure we can import local modules
sys.path.append(os.path.abspath("math_verification_mvp"))

from services.local_ocr.mvm2_ocr_engine import MVM2OCREngine
from models.llm_agent import LLMAgent
from services.core_engine.consensus_module import evaluate_consensus
from utils.math_utils import normalize_math_string

class ExhaustiveTester:
    def __init__(self):
        self.ocr = MVM2OCREngine()
        self.agents = [
            LLMAgent("GPT-4"),
            LLMAgent("Llama 3"),
            LLMAgent("Gemini 2.0 Pro"),
            LLMAgent("Qwen-2.5-Math-7B")
        ]
        self.results = []

    def define_test_cases(self):
        """Define a variety of mathematical edge cases for deep validation."""
        return [
            {
                "id": "TC_01_CJK_CLEAN",
                "latex": "2x + 5 = 15 沪州批发发大米",
                "ground_truth": "5",
                "challenge": "CJK Leakage & Algebra"
            },
            {
                "id": "TC_02_COMPLEX_INT",
                "latex": "\\int_{0}^{\\pi} \\sin(x^{2}) \\, dx",
                "ground_truth": "0.7799",
                "challenge": "Symbolic Integration"
            },
            {
                "id": "TC_03_QUADRATIC",
                "latex": "x^2 - 5x + 6 = 0",
                "ground_truth": "2, 3",
                "challenge": "Multi-value derivation"
            },
            {
                "id": "TC_04_CONTRADICTION",
                "latex": "Prove 1 + 1 = 3",
                "ground_truth": "False",
                "challenge": "Consensus Hallucination Logic"
            },
            {
                "id": "TC_05_NOISY_OCR",
                "latex": "100 * 20 / 5",
                "ground_truth": "400",
                "challenge": "Arithmetic Precision"
            }
        ]

    def run_benchmark(self):
        cases = self.define_test_cases()
        print(f"--- Starting Exhaustive ML Benchmark ({len(cases)} cases) ---\n")
        
        for case in cases:
            print(f"[{case['id']}] Challenge: {case['challenge']}")
            
            input_latex = case["latex"]
            
            agent_responses = []
            for agent in self.agents:
                try:
                    sol = agent.generate_solution(input_latex)
                    agent_responses.append({
                        "agent": agent.model_name,
                        "response": {
                            "Answer": sol.get("final_answer"),
                            "Reasoning Trace": sol.get("reasoning_trace"),
                            "Confidence Explanation": sol.get("confidence_explanation")
                        }
                    })
                except Exception as e:
                    print(f"  [ERROR] Agent {agent.model_name} failed: {e}")

            # Consensus Fusion
            consensus = evaluate_consensus(agent_responses, ocr_confidence=0.95)
            
            # Robust Comparison
            actual = normalize_math_string(consensus["final_verified_answer"])
            expected = normalize_math_string(case["ground_truth"])
            is_correct = actual == expected
            
            res = {
                "id": case["id"],
                "input": input_latex,
                "verified_answer": consensus["final_verified_answer"],
                "verdict": consensus["verdict"],
                "score": consensus["winning_score"],
                "ground_truth": case["ground_truth"],
                "is_accurate": is_correct,
                "hallucinations": len(consensus.get("hallucination_alerts", [])),
                "has_divergence": consensus.get("has_divergence", False)
            }
            self.results.append(res)
            print(f"  Result: {res['verdict']} | Score: {res['score']} | Accurate: {res['is_accurate']}")
            print("-" * 40)

    def generate_report(self):
        total = len(self.results)
        accurate = sum(1 for r in self.results if r["is_accurate"])
        hallucination_detected = sum(1 for r in self.results if r["hallucinations"] > 0)
        
        report = f"""# MVM² Deep Test ML Report
        
## Benchmark Summary
- **Total Cases:** {total}
- **Accuracy Rate:** {(accurate/total)*100:.1f}%
- **Hallucination Detection Rate:** {(hallucination_detected/total)*100:.1f}%
- **Mean Consensus Score:** {sum(r['score'] for r in self.results)/total:.3f}

## Detailed Findings
"""
        for r in self.results:
            status = "✅" if r["is_accurate"] else "❌"
            report += f"- **[{r['id']}]** {status} Verdict: `{r['verdict']}` | Score: {r['score']} | Div: {r['has_divergence']}\n"
            if not r["is_accurate"]:
                report += f"  - Expected: `{r['ground_truth']}` | Got: `{r['verified_answer']}`\n"
            
        with open("Deep_Test_Report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n[SUCCESS] Final ML Deep Test Report generated: Deep_Test_Report.md")

if __name__ == "__main__":
    tester = ExhaustiveTester()
    tester.run_benchmark()
    tester.generate_report()
