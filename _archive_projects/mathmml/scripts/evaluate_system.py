"""Comprehensive evaluation script matching PRD requirements."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.symbolic_verifier import SymbolicVerifier
from src.models.llm_logical_checker import LLMLogicalChecker
from src.models.ensemble_checker import EnsembleNeuralChecker
from src.models.ml_step_classifier import MLStepClassifierWrapper
from src.pipeline.consensus import ConsensusEngine
from src.data.loaders import parse_gsm8k_answer, create_step_dataset
from src.utils.error_location import ErrorLocationTracker


def evaluate_on_dataset(
    dataset_path: str,
    num_samples: int = 100,
    model_path: str = None
) -> Dict:
    """Evaluate system on a dataset.
    
    Args:
        dataset_path: Path to test dataset JSON
        num_samples: Number of samples to evaluate
        model_path: Path to trained ML classifier
        
    Returns:
        Evaluation metrics dict
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Sample if needed
    if num_samples and num_samples < len(dataset):
        import random
        random.seed(42)
        dataset = random.sample(dataset, num_samples)
    
    # Initialize verifiers
    verifiers = {
        'symbolic': SymbolicVerifier(),
        'llm_logical': LLMLogicalChecker(use_api=False),
        'ensemble': EnsembleNeuralChecker(use_apis=False),
        'ml_classifier': MLStepClassifierWrapper(model_path=model_path, device="cpu")
    }
    
    consensus_engine = ConsensusEngine()
    
    # Metrics
    metrics = {
        'total_problems': len(dataset),
        'total_steps': 0,
        'correct_verdicts': 0,
        'error_detections': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'true_positives': 0,
        'true_negatives': 0,
        'processing_times': [],
        'error_type_counts': defaultdict(int),
        'verifier_agreements': defaultdict(int)
    }
    
    # Evaluate each problem
    for problem_data in dataset:
        problem = problem_data.get('problem', '')
        answer = problem_data.get('answer', '')
        label = problem_data.get('label', 'correct')  # Ground truth
        
        # Parse steps
        steps = parse_gsm8k_answer(answer)
        if not steps:
            continue
        
        metrics['total_steps'] += len(steps)
        
        # Verify each step
        for step in steps:
            start_time = time.time()
            
            # Run verifiers
            verifier_results = {}
            for name, verifier in verifiers.items():
                result = verifier.verify(step, problem, [])
                verifier_results[name] = result
            
            # Compute consensus
            consensus = consensus_engine.compute_consensus(verifier_results)
            
            processing_time = time.time() - start_time
            metrics['processing_times'].append(processing_time)
            
            # Check if correct
            final_verdict = consensus['final_verdict']
            is_error = label != 'correct'
            
            if is_error and final_verdict == 'ERROR':
                metrics['true_positives'] += 1
                metrics['error_detections'] += 1
            elif not is_error and final_verdict == 'VALID':
                metrics['true_negatives'] += 1
                metrics['correct_verdicts'] += 1
            elif is_error and final_verdict == 'VALID':
                metrics['false_negatives'] += 1
            elif not is_error and final_verdict == 'ERROR':
                metrics['false_positives'] += 1
            
            # Track error types
            if final_verdict == 'ERROR':
                error_type = consensus.get('primary_error_type', 'unknown')
                metrics['error_type_counts'][error_type] += 1
            
            # Track agreement
            agreement = consensus.get('agreement_type', 'MIXED')
            metrics['verifier_agreements'][agreement] += 1
    
    # Calculate final metrics
    total = metrics['total_steps']
    tp = metrics['true_positives']
    tn = metrics['true_negatives']
    fp = metrics['false_positives']
    fn = metrics['false_negatives']
    
    metrics['accuracy'] = (tp + tn) / total if total > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    metrics['error_detection_rate'] = metrics['error_detections'] / total if total > 0 else 0
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['avg_processing_time'] = np.mean(metrics['processing_times']) if metrics['processing_times'] else 0
    
    return metrics


def print_metrics(metrics: Dict):
    """Print evaluation metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    
    print(f"\n🔍 Error Detection:")
    print(f"  Error Detection Rate: {metrics['error_detection_rate']:.3f} ({metrics['error_detection_rate']*100:.1f}%)")
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.3f} ({metrics['false_positive_rate']*100:.1f}%)")
    print(f"  True Positives: {metrics['true_positives']}")
    print(f"  True Negatives: {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    print(f"\n⚡ Performance:")
    print(f"  Average Processing Time: {metrics['avg_processing_time']*1000:.1f}ms per step")
    print(f"  Total Steps Processed: {metrics['total_steps']}")
    
    print(f"\n📈 Agreement Distribution:")
    for agreement, count in metrics['verifier_agreements'].items():
        print(f"  {agreement}: {count} ({count/metrics['total_steps']*100:.1f}%)")
    
    print(f"\n🔢 Error Type Distribution:")
    for error_type, count in sorted(metrics['error_type_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type}: {count}")
    
    print("\n" + "=" * 60)
    
    # PRD Target Comparison
    print("\n🎯 PRD Target Comparison:")
    targets = {
        'accuracy': 0.715,
        'error_detection_rate': 0.783,
        'false_positive_rate': 0.021,
        'avg_processing_time': 0.5  # seconds
    }
    
    for metric, target in targets.items():
        actual = metrics.get(metric, 0)
        if metric == 'avg_processing_time':
            status = "✅" if actual <= target else "⚠️"
            print(f"  {status} {metric}: {actual:.3f} (target: ≤{target:.3f})")
        else:
            status = "✅" if actual >= target else "⚠️"
            print(f"  {status} {metric}: {actual:.3f} (target: ≥{target:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate system on benchmark dataset")
    parser.add_argument("--dataset", type=str, default="data/processed/test.json",
                       help="Path to test dataset")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained ML classifier")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    print("🚀 Starting System Evaluation")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    
    metrics = evaluate_on_dataset(
        args.dataset,
        num_samples=args.num_samples,
        model_path=args.model_path
    )
    
    # Print results
    print_metrics(metrics)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()

