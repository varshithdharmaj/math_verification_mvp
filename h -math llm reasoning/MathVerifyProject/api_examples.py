"""
Python API Examples for MathVerifyProject
Perfect for Jupyter notebooks, Colab, and script integration

Usage:
    import api_examples
    # Or copy code snippets into your notebook/script
"""

from core_verification import MathVerifier
# Optional imports - only if needed
try:
    from benchmark_evaluation import MathVEvaluator, MathVerseEvaluator
except ImportError:
    pass
try:
    from ocr_input import HandwritingTranscriber
except ImportError:
    pass
try:
    from main import MathVerifyPipeline
except ImportError:
    MathVerifyPipeline = None


# ============================================================================
# Example 1: Basic Verification (Simple Boolean Result)
# ============================================================================

def example_basic_verification():
    """Basic verification - returns True/False."""
    verifier = MathVerifier()
    
    # Simple verification
    result = verifier.verify_answer(gold="1/2", prediction="0.5")
    print(f"Result: {result}")  # True or False
    
    return result


# ============================================================================
# Example 2: Detailed Verification (Returns Dict with Full Details)
# ============================================================================

def example_detailed_verification():
    """Detailed verification - returns dict with all information."""
    verifier = MathVerifier()
    
    # Detailed verification with full information
    result = verifier.verify_answer(
        gold="1/2", 
        prediction="0.5",
        return_details=True
    )
    
    print("Verification Result:")
    print(f"  Valid: {result['valid']}")
    print(f"  Gold: {result['gold']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Gold Parsed: {result['gold_parsed']}")
    print(f"  Prediction Parsed: {result['pred_parsed']}")
    if result.get('error_type'):
        print(f"  Error Type: {result['error_type']}")
    print(f"  Details: {result['details']}")
    
    return result


# ============================================================================
# Example 3: Batch Verification
# ============================================================================

def example_batch_verification():
    """Verify multiple answers at once."""
    verifier = MathVerifier()
    
    # Prepare data
    gold_answers = ["1/2", "2+2", "sqrt(4)", "42"]
    predictions = ["0.5", "4", "2", "43"]
    
    # Batch verification (simple boolean results)
    results = verifier.verify_batch(gold_answers, predictions)
    print(f"Results: {results}")  # [True, True, True, False]
    
    # Batch verification with details
    detailed_results = verifier.verify_batch(
        gold_answers, 
        predictions,
        return_details=True
    )
    
    print("\nDetailed Results:")
    for i, result in enumerate(detailed_results, 1):
        status = "✓" if result['valid'] else "✗"
        print(f"{i}. {status} Gold: {result['gold']} | Pred: {result['prediction']}")
        if result.get('error_type'):
            print(f"   Error: {result['error_type']}")
    
    return detailed_results


# ============================================================================
# Example 4: Using the Full Pipeline
# ============================================================================

def example_full_pipeline():
    """Use the complete integrated pipeline."""
    if MathVerifyPipeline is None:
        print("Pipeline not available (optional dependencies missing)")
        return None
    
    # Initialize pipeline
    pipeline = MathVerifyPipeline()
    
    # Process a math problem
    result = pipeline.process_math_problem(
        problem_text="What is 1/2?",
        model_answer="0.5",
        gold_answer="1/2",
        use_ocr=False
    )
    
    print("Pipeline Result:")
    print(f"  Verification: {result['verification']}")
    print(f"  Problem: {result['problem']}")
    print(f"  Model Answer: {result['model_answer']}")
    print(f"  Gold Answer: {result['gold_answer']}")
    
    return result


# ============================================================================
# Example 5: Custom Configuration
# ============================================================================

def example_custom_config():
    """Use custom verification configuration."""
    from core_verification import MathVerifier
    from math_verify import ExprExtractionConfig, LatexExtractionConfig
    
    # Create verifier with custom config
    verifier = MathVerifier(
        gold_extraction_config=[ExprExtractionConfig()],
        pred_extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        float_rounding=8,  # More precision
        numeric_precision=20,
        strict=False  # Less strict comparison
    )
    
    result = verifier.verify_answer(
        gold="1/2",
        prediction="0.5",
        return_details=True
    )
    
    print(f"Custom Config Result: {result['valid']}")
    return result


# ============================================================================
# Example 6: Jupyter/Colab Notebook Style
# ============================================================================

def notebook_example():
    """
    Example formatted for Jupyter/Colab notebooks.
    Copy this into a notebook cell.
    """
    from core_verification import MathVerifier
    
    # Initialize
    verifier = MathVerifier()
    
    # Test cases
    test_cases = [
        {"gold": "1/2", "pred": "0.5"},
        {"gold": "2+2", "pred": "4"},
        {"gold": "sqrt(4)", "pred": "2"},
        {"gold": "42", "pred": "43"},
    ]
    
    # Verify all
    results = []
    for case in test_cases:
        result = verifier.verify_answer(
            gold=case["gold"],
            prediction=case["pred"],
            return_details=True
        )
        results.append(result)
    
    # Display results
    print("Verification Results:")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        status = "✓ CORRECT" if result['valid'] else "✗ INCORRECT"
        print(f"{i}. {status}")
        print(f"   Gold: {result['gold']} | Pred: {result['prediction']}")
        if result.get('error_type'):
            print(f"   Error: {result['error_type']}")
        print()
    
    return results


# ============================================================================
# Example 7: Integration with Other Libraries
# ============================================================================

def example_with_pandas():
    """Example using pandas for data analysis."""
    try:
        import pandas as pd
        
        verifier = MathVerifier()
        
        # Create DataFrame
        data = {
            'gold': ["1/2", "2+2", "sqrt(4)", "42"],
            'prediction': ["0.5", "4", "2", "43"]
        }
        df = pd.DataFrame(data)
        
        # Verify all rows
        df['correct'] = df.apply(
            lambda row: verifier.verify_answer(
                gold=row['gold'],
                prediction=row['prediction']
            ),
            axis=1
        )
        
        # Calculate accuracy
        accuracy = df['correct'].mean() * 100
        print(f"Accuracy: {accuracy:.2f}%")
        print(df)
        
        return df
    except ImportError:
        print("pandas not installed. Install with: pip install pandas")
        return None


# ============================================================================
# Example 8: Error Analysis
# ============================================================================

def example_error_analysis():
    """Analyze errors in predictions."""
    verifier = MathVerifier()
    
    gold_answers = ["1/2", "2+2", "sqrt(4)", "42", "1/3"]
    predictions = ["0.5", "4", "2", "43", "0.333"]
    
    # Get detailed results
    results = verifier.verify_batch(
        gold_answers,
        predictions,
        return_details=True
    )
    
    # Analyze errors
    errors = [r for r in results if not r['valid']]
    error_types = {}
    
    for error in errors:
        error_type = error.get('error_type', 'Unknown')
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    print("Error Analysis:")
    print("=" * 60)
    print(f"Total errors: {len(errors)}")
    print("\nError Types:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}")
    
    return error_types


# ============================================================================
# Main Examples Runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MathVerifyProject - Python API Examples")
    print("=" * 60)
    print()
    
    print("Example 1: Basic Verification")
    print("-" * 60)
    example_basic_verification()
    print()
    
    print("Example 2: Detailed Verification")
    print("-" * 60)
    example_detailed_verification()
    print()
    
    print("Example 3: Batch Verification")
    print("-" * 60)
    example_batch_verification()
    print()
    
    print("Example 4: Full Pipeline")
    print("-" * 60)
    example_full_pipeline()
    print()
    
    print("Example 5: Custom Configuration")
    print("-" * 60)
    example_custom_config()
    print()
    
    print("Example 6: Notebook Style")
    print("-" * 60)
    notebook_example()
    print()
    
    print("Example 7: With Pandas")
    print("-" * 60)
    example_with_pandas()
    print()
    
    print("Example 8: Error Analysis")
    print("-" * 60)
    example_error_analysis()
    print()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)

