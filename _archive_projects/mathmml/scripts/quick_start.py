"""Quick start script to demonstrate the verification system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.symbolic_verifier import SymbolicVerifier
from src.models.llm_logical_checker import LLMLogicalChecker
from src.models.ensemble_checker import EnsembleNeuralChecker
from src.models.ml_step_classifier import MLStepClassifierWrapper
from src.pipeline.consensus import ConsensusEngine


def main():
    """Run a quick demo."""
    print("🔢 Math Verification System - Quick Start\n")
    
    # Initialize verifiers
    print("Initializing verifiers...")
    verifiers = {
        'symbolic': SymbolicVerifier(),
        'llm_logical': LLMLogicalChecker(use_api=False),
        'ensemble': EnsembleNeuralChecker(use_apis=False),
        'ml_classifier': MLStepClassifierWrapper(model_path=None, device="cpu")
    }
    
    # Initialize consensus
    consensus_engine = ConsensusEngine()
    
    # Example problem
    problem = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    steps = [
        "Natalia sold 48/2 = 24 clips in May.",
        "Natalia sold 48+24 = 72 clips altogether in April and May."
    ]
    
    print(f"\nProblem: {problem}\n")
    
    # Verify each step
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step}")
        prev_steps = steps[:i-1]
        
        # Run verification
        consensus = consensus_engine.verify_step(step, problem, prev_steps, verifiers)
        
        verdict = consensus['final_verdict']
        confidence = consensus['overall_confidence']
        agreement = consensus['agreement_type']
        
        print(f"  Verdict: {verdict}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Agreement: {agreement}")
        
        if consensus.get('primary_error_type'):
            print(f"  Error Type: {consensus['primary_error_type']}")
        
        print()
    
    print("✅ Verification complete!")


if __name__ == "__main__":
    main()

