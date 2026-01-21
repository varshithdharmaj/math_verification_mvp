"""
MVM² Benchmark Runner
Unified script to run evaluations on integrated research benchmarks.
"""
import os
import sys
import argparse
import subprocess

def run_mathverse(limit=None):
    """Run MathVerse evaluation"""
    print("\n" + "="*50)
    print("[START] MathVerse Benchmark (ECCV 2024)")
    print("="*50)
    
    cmd = [sys.executable, "evaluate_mathverse.py"]
    if limit:
        cmd.extend(["--limit", str(limit)])
        
    subprocess.run(cmd)

def run_mathv(limit=None):
    """Run MATH-V evaluation"""
    print("\n" + "="*50)
    print("[START] MATH-V Benchmark (NeurIPS 2024)")
    print("="*50)
    
    cmd = [sys.executable, "evaluate_mathv.py"]
    if limit:
        cmd.extend(["--limit", str(limit)])
        
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run MVM2 Research Benchmarks")
    parser.add_argument('benchmark', choices=['mathverse', 'mathv', 'all'], 
                      help="Benchmark to run")
    parser.add_argument('--limit', type=int, default=None, 
                      help="Limit number of samples (for testing)")
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import datasets
    except ImportError:
        print("[ERROR] Missing dependency: 'datasets'")
        print("Please run: pip install datasets")
        return

    if args.benchmark in ['mathverse', 'all']:
        run_mathverse(args.limit)
        
    if args.benchmark in ['mathv', 'all']:
        run_mathv(args.limit)

if __name__ == "__main__":
    main()
