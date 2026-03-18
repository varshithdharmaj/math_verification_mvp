"""
Data loader script for downloading and preparing datasets.

Downloads GSM8K test split and creates sample datasets for testing.
"""

import os
import json
from datasets import load_dataset
from pathlib import Path


def download_gsm8k(num_examples: int = 100):
    """
    Download GSM8K test split.
    
    Args:
        num_examples: Number of examples to download
    """
    print(f"Downloading GSM8K test split ({num_examples} examples)...")
    
    try:
        # Load GSM8K dataset
        dataset = load_dataset("gsm8k", "main", split="test")
        
        # Take first num_examples
        subset = dataset.select(range(min(num_examples, len(dataset))))
        
        # Create directory
        output_dir = Path("data/gsm8k_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        data = []
        for item in subset:
            data.append({
                "question": item["question"],
                "answer": item["answer"]
            })
        
        output_file = output_dir / "test.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved {len(data)} examples to {output_file}")
        print(f"   File size: {output_file.stat().st_size / 1024:.2f} KB")
        
        return output_file
        
    except Exception as e:
        print(f"[ERROR] Error downloading GSM8K: {e}")
        print("   Creating sample dataset instead...")
        return create_sample_dataset()


def create_sample_dataset():
    """Create a sample dataset for testing."""
    print("Creating sample test dataset...")
    
    sample_problems = [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "answer": "Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day at the farmer's market."
        },
        {
            "question": "Solve for x: 2x + 3 = 7",
            "answer": "2x = 7 - 3 = 4, so x = 2"
        },
        {
            "question": "Calculate: (1/2) + (1/4) = ?",
            "answer": "1/2 + 1/4 = 2/4 + 1/4 = 3/4"
        },
        {
            "question": "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
            "answer": "Distance = speed x time = 60 x 2.5 = 150 miles"
        },
        {
            "question": "Simplify: 3x^2 + 2x^2 - 5x + 3x = ?",
            "answer": "3x^2 + 2x^2 - 5x + 3x = 5x^2 - 2x"
        },
        {
            "question": "What is 15% of 200?",
            "answer": "15% of 200 = 0.15 x 200 = 30"
        },
        {
            "question": "A rectangle has length 8 and width 5. What is its area?",
            "answer": "Area = length x width = 8 x 5 = 40"
        },
        {
            "question": "Solve: 3(x + 2) = 15",
            "answer": "3(x + 2) = 15, so x + 2 = 5, therefore x = 3"
        },
        {
            "question": "What is the square root of 144?",
            "answer": "sqrt(144) = 12"
        },
        {
            "question": "If y = 2x + 1 and x = 4, what is y?",
            "answer": "y = 2(4) + 1 = 8 + 1 = 9"
        }
    ]
    
    output_dir = Path("data/test_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "samples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_problems, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Created sample dataset with {len(sample_problems)} problems")
    print(f"   Saved to {output_file}")
    
    return output_file


if __name__ == "__main__":
    print("=" * 60)
    print("MathVerify-Integrated: Data Loader")
    print("=" * 60)
    print()
    
    # Download GSM8K
    gsm8k_file = download_gsm8k(100)
    
    # Create sample dataset
    sample_file = create_sample_dataset()
    
    print()
    print("=" * 60)
    print("[OK] Data preparation complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {gsm8k_file}")
    print(f"  - {sample_file}")
