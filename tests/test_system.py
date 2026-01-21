"""
Comprehensive system testing
Tests all 5 demo cases + multimodal capabilities
"""
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.orchestrator import MathVerificationOrchestrator
import time

def run_tests():
    orchestrator = MathVerificationOrchestrator()
    
    # Load demo cases
    with open('demo_cases.json') as f:
        data = json.load(f)
    
    print("\n" + "="*70)
    print("MVM² SYSTEM TESTING")
    print("="*70 + "\n")
    
    results = []
    correct = 0
    total_time = 0
    
    for case in data['cases']:
        print(f"Test {case['id']}: {case['name']}")
        print(f"  Category: {case['category']} | Difficulty: {case['difficulty']}")
        
        start = time.time()
        
        # Run verification
        result = orchestrator.verify(case['problem'], case['steps'])
        
        elapsed = time.time() - start
        total_time += elapsed
        
        # Check if correct
        is_correct = result['final_verdict'] == case['expected_verdict']
        if is_correct:
            correct += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        
        print(f"  Expected: {case['expected_verdict']}")
        print(f"  Got: {result['final_verdict']}")
        print(f"  Confidence: {result['overall_confidence']*100:.1f}%")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Status: {status}\n")
        
        results.append({
            'case': case['name'],
            'expected': case['expected_verdict'],
            'got': result['final_verdict'],
            'correct': is_correct,
            'confidence': result['overall_confidence'],
            'time': elapsed
        })
    
    # Summary
    accuracy = (correct / len(data['cases'])) * 100
    avg_time = total_time / len(data['cases'])
    
    print("="*70)
    print(f"RESULTS: {correct}/{len(data['cases'])} tests passed")
    print(f"ACCURACY: {accuracy:.1f}%")
    print(f"AVG TIME: {avg_time:.2f}s per problem")
    print(f"TOTAL TIME: {total_time:.2f}s")
    print("="*70)
    
    # Detailed breakdown
    print("\nDETAILED RESULTS:")
    print("-" * 70)
    for r in results:
        status = "✓" if r['correct'] else "✗"
        print(f"{status} {r['case']:30s} | Expected: {r['expected']:5s} | Got: {r['got']:5s} | {r['confidence']*100:5.1f}% | {r['time']:.2f}s")
    print("-" * 70)
    
    return results, accuracy

if __name__ == "__main__":
    print("🔧 Starting MVM² System Tests...")
    print("⚠️  Make sure all microservices are running:")
    print("   - OCR Service (Port 8001)")
    print("   - SymPy Service (Port 8002)")
    print("   - LLM Service (Port 8003)\n")
    
    input("Press Enter to continue...")
    
    results, acc = run_tests()
    
    # Exit code
    if acc == 100:
        print("\n✅ ALL TESTS PASSED!")
        exit(0)
    else:
        print(f"\n⚠️  {100-acc:.0f}% tests failed")
        exit(1)
