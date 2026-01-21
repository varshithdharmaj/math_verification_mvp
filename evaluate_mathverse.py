"""
MathVerse Evaluation Integration
Evaluates our MVM² system on MathVerse benchmark (ECCV 2024)
"""
import sys
import os

# Add MathVerse to path
mathverse_path = os.path.join(os.path.dirname(__file__), '..', 'external_resources', 'MathVerse')
sys.path.insert(0, mathverse_path)

import json
from typing import Dict, List
from services.orchestrator import MathVerificationOrchestrator

class MathVerseEvaluator:
    """
    Evaluate MVM² on MathVerse benchmark
    MathVerse: 2,612 problems × 6 versions = 15,672 test samples
    """
    
    def __init__(self):
        self.orchestrator = MathVerificationOrchestrator()
        self.results = []
        
    def load_testmini(self):
        """
        Load MathVerse testmini dataset
        788 problems × 5 versions = 3,940 samples
        """
        try:
            from datasets import load_dataset
            
            print("[LOAD] Loading MathVerse testmini dataset...")
            dataset = load_dataset("AI4Math/MathVerse", "testmini")
            
            print(f"[OK] Loaded {len(dataset['testmini'])} test samples")
            return dataset['testmini']
        
        except Exception as e:
            print(f"[ERROR] Failed to load MathVerse: {e}")
            print("[INFO] Install with: pip install datasets")
            return None
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """
        Evaluate a single MathVerse sample
        """
        try:
            # Extract problem details
            problem_text = sample.get('question', '')
            query = sample.get('query_wo', sample.get('query', ''))
            ground_truth = sample.get('answer', '')
            problem_version = sample.get('problem_version', 'unknown')
            
            # Check if image is needed
            has_image = 'image' in sample and sample['image'] is not None
            
            # For text-based versions, extract steps from query
            if problem_version in ['Text Dominant', 'Text Lite', 'Text Only']:
                # Use text-based verification
                steps = [query]  # Simplified - in production, extract steps properly
                result = self.orchestrator.verify(problem_text, steps)
            
            elif has_image:
                # Save image temporarily
                image = sample['image']
                temp_path = f"temp_mathverse_{sample['sample_index']}.png"
                image.save(temp_path)
                
                # Image-based verification
                result = self.orchestrator.verify_from_image(temp_path)
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            else:
                # Fall back to text
                steps = [query]
                result = self.orchestrator.verify(problem_text, steps)
            
            # Extract predicted answer from result
            predicted_answer = self._extract_answer(result)
            
            # Compare with ground truth
            is_correct = self._compare_answers(predicted_answer, ground_truth)
            
            return {
                'sample_index': sample.get('sample_index'),
                'problem_index': sample.get('problem_index'),
                'problem_version': problem_version,
                'subject': sample.get('subject', 'unknown'),
                'level': sample.get('level', 0),
                'predicted': predicted_answer,
                'ground_truth': ground_truth,
                'correct': is_correct,
                'confidence': result.get('overall_confidence', 0),
                'verdict': result.get('final_verdict', 'UNKNOWN')
            }
        
        except Exception as e:
            print(f"[ERROR] Sample {sample.get('sample_index')}: {e}")
            return {
                'sample_index': sample.get('sample_index'),
                'error': str(e),
                'correct': False
            }
    
    def _extract_answer(self, result: Dict) -> str:
        """Extract final answer from verification result"""
        # This is simplified - in production, use Math-Verify's extraction
        if 'final_verdict' in result:
            return result['final_verdict']
        return "UNKNOWN"
    
    def _compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """Compare predicted answer with ground truth"""
        # Simple string comparison for now
        # In production, use Math-Verify's comparison
        return predicted.strip().lower() == ground_truth.strip().lower()
    
    def evaluate_all(self, limit: int = None):
        """
        Evaluate on MathVerse testmini
        """
        dataset = self.load_testmini()
        if not dataset:
            return
        
        total = limit if limit else len(dataset)
        correct = 0
        
        print(f"\n{'='*60}")
        print(f"MathVerse Evaluation - Testing {total} samples")
        print(f"{'='*60}\n")
        
        for i, sample in enumerate(dataset):
            if limit and i >= limit:
                break
            
            print(f"[{i+1}/{total}] Testing sample {sample.get('sample_index')}...")
            
            result = self.evaluate_sample(sample)
            self.results.append(result)
            
            if result.get('correct'):
                correct += 1
            
            # Progress update
            if (i+1) % 10 == 0:
                acc = (correct / (i+1)) * 100
                print(f"  Progress: {i+1}/{total} | Accuracy: {acc:.1f}%\n")
        
        # Final results
        self.print_results()
    
    def print_results(self):
        """Print evaluation results"""
        if not self.results:
            print("[WARNING] No results to display")
            return
        
        total = len(self.results)
        correct = sum(1 for r in self.results if r.get('correct'))
        accuracy = (correct / total) * 100
        
        print(f"\n{'='*60}")
        print(f"MATHVERSE EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total Samples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*60}")
        
        # By version
        versions = {}
        for r in self.results:
            v = r.get('problem_version', 'unknown')
            if v not in versions:
                versions[v] = {'total': 0, 'correct': 0}
            versions[v]['total'] += 1
            if r.get('correct'):
                versions[v]['correct'] += 1
        
        print("\nAccuracy by Version:")
        for v, stats in versions.items():
            acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"  {v:20s}: {acc:5.1f}% ({stats['correct']}/{stats['total']})")
        
        print(f"{'='*60}\n")
    
    def save_results(self, filepath: str = "mathverse_results.json"):
        """Save results to JSON"""
        with open(filepath, 'w') as f:
            json.dump({
                'total': len(self.results),
                'correct': sum(1 for r in self.results if r.get('correct')),
                'accuracy': (sum(1 for r in self.results if r.get('correct')) / len(self.results)) * 100 if self.results else 0,
                'results': self.results
            }, f, indent=2)
        
        print(f"[SAVE] Results saved to {filepath}")


def main():
    """Run MathVerse evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MVM² on MathVerse benchmark")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of samples to test")
    parser.add_argument('--output', type=str, default="mathverse_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    evaluator = MathVerseEvaluator()
    evaluator.evaluate_all(limit=args.limit)
    evaluator.save_results(args.output)


if __name__ == "__main__":
    main()
