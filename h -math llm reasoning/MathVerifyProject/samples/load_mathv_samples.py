"""
Load sample problems from MATH-V dataset for testing
"""

import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_mathv_samples(limit=5):
    """
    Load sample problems from MATH-V testmini dataset.
    
    Args:
        limit: Number of samples to load
        
    Returns:
        List of sample problems
    """
    test_file = os.path.join(
        os.path.dirname(__file__), '..', 'MATH-V', 'data', 'testmini.jsonl'
    )
    
    if not os.path.exists(test_file):
        print(f"Warning: MATH-V test file not found: {test_file}")
        return []
    
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if line.strip():
                sample = json.loads(line)
                samples.append(sample)
    
    return samples


if __name__ == '__main__':
    samples = load_mathv_samples(5)
    print(f"Loaded {len(samples)} samples from MATH-V")
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  ID: {sample['id']}")
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Answer: {sample['answer']}")
        print(f"  Subject: {sample.get('subject', 'N/A')}")

