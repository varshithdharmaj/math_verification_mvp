"""
Evaluation script for MathVerify-Integrated.

Runs baseline and full pipeline evaluation on GSM8K dataset.
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from src.pipeline import MathVerifyPipeline
from src.reasoning_module.engine import ReasoningEngine
from analysis.visualize import (
    plot_error_distribution,
    plot_verification_pipeline,
    create_comparison_table,
    generate_report
)

def load_test_data(num_problems: int = 50) -> List[Dict]:
    """Load test problems from GSM8K."""
    data_path = Path("data/gsm8k_test/test.json")
    if not data_path.exists():
        print("GSM8K data not found. Using sample data.")
        data_path = Path("data/test_samples/samples.json")
        
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Shuffle and select
    # random.shuffle(data) # Keep deterministic for now
    return data[:num_problems]

def run_baseline(problems: List[Dict]) -> Dict[str, Any]:
    """Run baseline evaluation (reasoning only)."""
    print("Running baseline evaluation...")
    engine = ReasoningEngine(model_name="gpt2")
    
    results = []
    total_errors = 0
    errors_by_type = {}
    correct_count = 0
    
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        print(f"Processing baseline {i+1}/{len(problems)}...", end='\r')
        
        # Generate solution without verification
        solution = engine.generate_solution(problem["question"])
        
        # Simple check against ground truth (exact match or inclusion)
        gold_answer = problem.get("answer", "")
        pred_answer = solution.get("final_answer", "")
        
        # Very basic accuracy check for baseline
        # In real scenario, we'd use a more robust grader
        is_correct = pred_answer in gold_answer or gold_answer in pred_answer
        
        if is_correct:
            correct_count += 1
        else:
            total_errors += 1
            # Baseline errors are mostly calculation or logic
            # We can't classify them well without the verifier!
            # So we just mark them as "INCORRECT"
            errors_by_type["INCORRECT"] = errors_by_type.get("INCORRECT", 0) + 1
            
        results.append({
            "problem": problem["question"],
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "is_correct": is_correct
        })
        
    print(f"\nBaseline complete in {time.time() - start_time:.2f}s")
    
    return {
        "accuracy": (correct_count / len(problems)) * 100,
        "total_errors": total_errors,
        "errors_by_type": errors_by_type,
        "results": results
    }

def run_pipeline(problems: List[Dict]) -> Dict[str, Any]:
    """Run full pipeline evaluation (with verification)."""
    print("Running pipeline evaluation...")
    pipeline = MathVerifyPipeline(model_name="gpt2")
    
    results = []
    total_errors = 0
    errors_by_type = {}
    correct_count = 0
    
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        print(f"Processing pipeline {i+1}/{len(problems)}...", end='\r')
        
        # Run full pipeline
        result = pipeline.process_problem(problem["question"])
        
        # Check against ground truth
        gold_answer = problem.get("answer", "")
        pred_answer = result.get("final_answer", "")
        
        is_correct = pred_answer in gold_answer or gold_answer in pred_answer
        
        if is_correct:
            correct_count += 1
        
        # Collect errors found by the pipeline
        pipeline_errors = result.get("errors", {})
        p_total = pipeline_errors.get("total_errors", 0)
        total_errors += p_total
        
        p_by_type = pipeline_errors.get("by_type", {})
        for k, v in p_by_type.items():
            errors_by_type[k] = errors_by_type.get(k, 0) + v
            
        results.append(result)
        
    print(f"\nPipeline complete in {time.time() - start_time:.2f}s")
    
    return {
        "accuracy": (correct_count / len(problems)) * 100,
        "total_errors": total_errors,
        "errors_by_type": errors_by_type,
        "results": results
    }

def main():
    print("=" * 60)
    print("MathVerify-Integrated: Evaluation Suite")
    print("=" * 60)
    
    # Load data
    problems = load_test_data(50)
    print(f"Loaded {len(problems)} test problems")
    
    # Run evaluations
    baseline_results = run_baseline(problems)
    pipeline_results = run_pipeline(problems)
    
    # Save raw results
    eval_results = {
        "baseline": baseline_results,
        "with_verification": pipeline_results
    }
    
    with open("evaluation_results.json", "w", encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2)
    print("Saved evaluation_results.json")
    
    # Generate artifacts
    print("\nGenerating analysis artifacts...")
    
    # 1. Comparison Table
    create_comparison_table(
        baseline_results, 
        pipeline_results, 
        save_path="comparison_table.md"
    )
    
    # 2. Error Analysis Chart
    # We use pipeline errors for detailed analysis
    errors_for_chart = {
        "by_type": pipeline_results["errors_by_type"],
        "percentage": {
            k: (v / pipeline_results["total_errors"] * 100) if pipeline_results["total_errors"] > 0 else 0
            for k, v in pipeline_results["errors_by_type"].items()
        }
    }
    plot_error_distribution(errors_for_chart, save_path="error_analysis.png")
    
    # 3. Pipeline Performance Chart
    plot_verification_pipeline(pipeline_results["results"], save_path="pipeline_performance.png")
    
    # 4. Comprehensive Report
    generate_report(pipeline_results["results"], save_path="results_report.md")
    
    print("\nEvaluation complete! Artifacts generated.")

if __name__ == "__main__":
    main()
