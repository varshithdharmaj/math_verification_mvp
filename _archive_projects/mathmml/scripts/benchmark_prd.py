"""Benchmark script matching PRD evaluation requirements."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import numpy as np
from scipy import stats
from collections import defaultdict
from src.data.loaders import load_gsm8k, create_step_dataset
from scripts.evaluate_system import evaluate_on_dataset, print_metrics


def run_prd_benchmarks():
    """Run all PRD-specified benchmarks."""
    print("=" * 60)
    print("PRD BENCHMARK EVALUATION")
    print("=" * 60)
    
    # Benchmark 1: GSM8K Test Set
    print("\n📊 Benchmark 1: GSM8K Test Set")
    print("-" * 60)
    
    try:
        # Load GSM8K test data
        gsm8k_test = load_gsm8k(".", "test", "main")
        print(f"Loaded {len(gsm8k_test)} GSM8K test examples")
        
        # Create step dataset with errors
        step_data = create_step_dataset(gsm8k_test[:100], inject_errors=True, error_ratio=0.5)
        
        # Save to temp file for evaluation
        temp_file = Path("data/processed/temp_benchmark.json")
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_file, 'w') as f:
            json.dump(step_data[:200], f)  # Sample 200 steps
        
        # Evaluate
        metrics_gsm8k = evaluate_on_dataset(str(temp_file), num_samples=200)
        print_metrics(metrics_gsm8k)
        
    except Exception as e:
        print(f"⚠️ GSM8K benchmark failed: {e}")
        metrics_gsm8k = None
    
    # Benchmark 2: Math500 (if available)
    print("\n📊 Benchmark 2: Math500")
    print("-" * 60)
    
    try:
        from src.data.loaders import load_math500
        math500 = load_math500("math_500_test.csv")
        if math500:
            math500_steps = create_step_dataset(math500[:50], inject_errors=True, error_ratio=0.5)
            temp_file2 = Path("data/processed/temp_math500.json")
            with open(temp_file2, 'w') as f:
                json.dump(math500_steps[:100], f)
            
            metrics_math500 = evaluate_on_dataset(str(temp_file2), num_samples=100)
            print_metrics(metrics_math500)
        else:
            print("⚠️ Math500 not available")
            metrics_math500 = None
    except Exception as e:
        print(f"⚠️ Math500 benchmark failed: {e}")
        metrics_math500 = None
    
    # Statistical Significance Test (PRD requirement)
    print("\n📈 Statistical Significance Analysis")
    print("-" * 60)
    
    if metrics_gsm8k:
        # Compare against baseline (64.7% from PRD)
        baseline_accuracy = 0.647
        system_accuracy = metrics_gsm8k['accuracy']
        n = metrics_gsm8k['total_steps']
        
        # Simple z-test approximation
        se = np.sqrt(baseline_accuracy * (1 - baseline_accuracy) / n)
        z_score = (system_accuracy - baseline_accuracy) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print(f"Baseline Accuracy: {baseline_accuracy:.3f}")
        print(f"System Accuracy: {system_accuracy:.3f}")
        print(f"Improvement: {(system_accuracy - baseline_accuracy)*100:.1f}%")
        print(f"Z-score: {z_score:.3f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("✅ Statistically significant improvement (p < 0.05)")
        else:
            print("⚠️ Not statistically significant (p >= 0.05)")
    
    # PRD Target Achievement Summary
    print("\n🎯 PRD Target Achievement Summary")
    print("-" * 60)
    
    if metrics_gsm8k:
        targets = {
            'Overall Accuracy': (metrics_gsm8k['accuracy'], 0.715),
            'Error Detection Rate': (metrics_gsm8k['error_detection_rate'], 0.783),
            'False Positive Rate': (metrics_gsm8k['false_positive_rate'], 0.021, True),  # Lower is better
            'Avg Processing Time (ms)': (metrics_gsm8k['avg_processing_time'] * 1000, 500, True)  # Lower is better
        }
        
        for metric_name, (actual, target, *reverse) in targets.items():
            reverse_flag = reverse[0] if reverse else False
            if reverse_flag:
                achieved = actual <= target
                symbol = "✅" if achieved else "⚠️"
                print(f"{symbol} {metric_name}: {actual:.3f} (target: ≤{target:.3f})")
            else:
                achieved = actual >= target
                symbol = "✅" if achieved else "⚠️"
                print(f"{symbol} {metric_name}: {actual:.3f} (target: ≥{target:.3f})")
    
    print("\n" + "=" * 60)
    print("✅ Benchmark evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_prd_benchmarks()

