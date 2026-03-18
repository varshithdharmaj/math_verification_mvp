"""Example: Using multiple LLM models (GPT, Gemini, Llama) for verification."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.llm_logical_checker import LLMLogicalChecker
from src.models.ensemble_checker import EnsembleNeuralChecker
from src.pipeline.consensus import ConsensusEngine
from src.models.symbolic_verifier import SymbolicVerifier
from src.models.ml_step_classifier import MLStepClassifierWrapper


def example_single_models():
    """Example: Using individual models."""
    print("=" * 60)
    print("Example 1: Single Model Verification")
    print("=" * 60)
    
    problem = "Natalia sold clips to 48 of her friends in April."
    step = "Natalia sold 48/2 = 25 clips in May."  # Error: should be 24
    
    # Test with GPT-4
    print("\n1. Testing with GPT-4:")
    try:
        checker_gpt = LLMLogicalChecker(
            use_api=True,
            api_provider="openai",
            model="gpt-4"
        )
        result = checker_gpt.verify(step, problem, [])
        print(f"   Verdict: {result['verdict']}")
        print(f"   Confidence: {result['confidence']:.3f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test with Gemini
    print("\n2. Testing with Gemini:")
    try:
        checker_gemini = LLMLogicalChecker(
            use_api=True,
            api_provider="gemini",
            model="gemini-pro"
        )
        result = checker_gemini.verify(step, problem, [])
        print(f"   Verdict: {result['verdict']}")
        print(f"   Confidence: {result['confidence']:.3f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test with Llama (Ollama)
    print("\n3. Testing with Llama (Ollama):")
    try:
        checker_llama = LLMLogicalChecker(
            use_api=True,
            api_provider="llama",
            model="llama2"
        )
        result = checker_llama.verify(step, problem, [])
        print(f"   Verdict: {result['verdict']}")
        print(f"   Confidence: {result['confidence']:.3f}")
    except Exception as e:
        print(f"   Error: {e} (Make sure Ollama is running)")


def example_ensemble():
    """Example: Using ensemble with multiple models."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Model Ensemble")
    print("=" * 60)
    
    problem = "A store has 15 apples. They sell 8 apples."
    step = "15 - 8 = 6 apples left."  # Error: should be 7
    
    # Create ensemble with GPT, Gemini, and Llama
    ensemble = EnsembleNeuralChecker(
        use_apis=True,
        num_models=3,
        model_configs=[
            {"provider": "openai", "model": "gpt-3.5-turbo"},
            {"provider": "gemini", "model": "gemini-pro"},
            {"provider": "llama", "model": "llama2"}
        ]
    )
    
    print(f"\nProblem: {problem}")
    print(f"Step: {step}")
    print("\nRunning ensemble with 3 models...")
    
    result = ensemble.verify(step, problem, [])
    
    print(f"\nEnsemble Result:")
    print(f"  Verdict: {result['verdict']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Votes: {result.get('votes', {})}")
    print(f"  Details: {result['details']}")


def example_full_pipeline():
    """Example: Full 4-model pipeline with different LLMs."""
    print("\n" + "=" * 60)
    print("Example 3: Full Pipeline with Multi-Model Support")
    print("=" * 60)
    
    problem = "Calculate 5 + 3"
    step = "5 + 3 = 9"  # Error!
    
    # Initialize all verifiers
    verifiers = {
        'symbolic': SymbolicVerifier(),
        'llm_logical': LLMLogicalChecker(
            use_api=True,
            api_provider="openai",
            model="gpt-3.5-turbo"
        ),
        'ensemble': EnsembleNeuralChecker(
            use_apis=True,
            num_models=2,
            model_configs=[
                {"provider": "openai", "model": "gpt-3.5-turbo"},
                {"provider": "gemini", "model": "gemini-pro"}
            ]
        ),
        'ml_classifier': MLStepClassifierWrapper(model_path=None, device="cpu")
    }
    
    # Consensus engine
    consensus_engine = ConsensusEngine()
    
    print(f"\nProblem: {problem}")
    print(f"Step: {step}")
    print("\nRunning full pipeline...")
    
    consensus = consensus_engine.verify_step(step, problem, [], verifiers)
    
    print(f"\nFinal Consensus:")
    print(f"  Verdict: {consensus['final_verdict']}")
    print(f"  Confidence: {consensus['overall_confidence']:.3f}")
    print(f"  Agreement: {consensus['agreement_type']}")
    print(f"  Error Score: {consensus['error_score']:.3f}")
    
    print("\nPer-Verifier Results:")
    for name, result in consensus['per_verifier_results'].items():
        print(f"  {name}: {result['verdict']} (conf: {result['confidence']:.3f})")


if __name__ == "__main__":
    print("\n🔢 Multi-Model Verification Examples\n")
    print("Note: Requires API keys or Ollama for LLM models")
    print("System works in mock mode if APIs are not available\n")
    
    # Run examples
    example_single_models()
    
    try:
        example_ensemble()
    except Exception as e:
        print(f"\nEnsemble example failed: {e}")
        print("(This is normal if API keys are not set)")
    
    try:
        example_full_pipeline()
    except Exception as e:
        print(f"\nPipeline example failed: {e}")
        print("(This is normal if API keys are not set)")
    
    print("\n" + "=" * 60)
    print("✅ Examples complete!")
    print("=" * 60)

