"""CLI script for inference with ML classifier."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ml_step_classifier import MLStepClassifierWrapper


def main():
    parser = argparse.ArgumentParser(description="Run inference with ML classifier")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--problem", type=str, required=True,
                       help="Problem statement")
    parser.add_argument("--step", type=str, required=True,
                       help="Step text")
    parser.add_argument("--prev_steps", type=str, default="",
                       help="Previous steps context")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Load model
    classifier = MLStepClassifierWrapper(
        model_path=args.model_path,
        device=args.device
    )
    
    # Run inference
    result = classifier.infer(args.problem, args.prev_steps, args.step)
    
    # Print results
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"\nAll probabilities:")
    for label, prob in result['all_probs'].items():
        print(f"  {label}: {prob:.3f}")


if __name__ == "__main__":
    main()

