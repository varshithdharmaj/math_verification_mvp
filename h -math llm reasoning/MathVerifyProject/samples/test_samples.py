"""
Test all interfaces with sample problems from MATH-V
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core_verification import MathVerifier
from main import MathVerifyPipeline
from samples.load_mathv_samples import load_mathv_samples


def test_pipeline_mode(samples):
    """Test pipeline mode with samples."""
    print("\n" + "="*60)
    print("Testing Pipeline Mode")
    print("="*60)
    
    pipeline = MathVerifyPipeline()
    results = []
    
    for i, sample in enumerate(samples[:5], 1):
        gold = sample['answer']
        # Simulate a prediction (for demo, use correct answer)
        prediction = gold
        
        result = pipeline.process_math_problem(
            problem_text=sample['question'],
            model_answer=prediction,
            gold_answer=gold,
            use_ocr=False
        )
        
        results.append({
            'sample_id': sample['id'],
            'gold': gold,
            'prediction': prediction,
            'verification': result['verification'],
            'question': sample['question'][:50] + "..."
        })
        
        print(f"\nSample {i} (ID: {sample['id']}):")
        print(f"  Question: {sample['question'][:60]}...")
        print(f"  Gold: {gold} | Prediction: {prediction}")
        print(f"  Verification: {'✓ CORRECT' if result['verification'] else '✗ INCORRECT'}")
    
    return results


def test_api_mode(samples):
    """Test API mode with samples."""
    print("\n" + "="*60)
    print("Testing API Mode")
    print("="*60)
    
    verifier = MathVerifier()
    results = []
    
    for i, sample in enumerate(samples[:5], 1):
        gold = sample['answer']
        prediction = gold  # Use correct answer for demo
        
        result = verifier.verify_answer(
            gold=gold,
            prediction=prediction,
            return_details=True
        )
        
        results.append(result)
        
        print(f"\nSample {i} (ID: {sample['id']}):")
        print(f"  Gold: {gold} | Prediction: {prediction}")
        print(f"  Valid: {result['valid']}")
        if result.get('error_type'):
            print(f"  Error: {result['error_type']}")
    
    return results


def save_results(results, mode):
    """Save test results to file."""
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'{mode}_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


def main():
    """Run all tests."""
    print("="*60)
    print("MathVerifyProject - Sample Testing")
    print("="*60)
    
    # Load samples
    samples = load_mathv_samples(5)
    if not samples:
        print("Error: Could not load samples from MATH-V")
        return
    
    print(f"\nLoaded {len(samples)} samples from MATH-V dataset")
    
    # Test Pipeline Mode
    pipeline_results = test_pipeline_mode(samples)
    save_results(pipeline_results, 'pipeline')
    
    # Test API Mode
    api_results = test_api_mode(samples)
    save_results(api_results, 'api')
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == '__main__':
    main()

