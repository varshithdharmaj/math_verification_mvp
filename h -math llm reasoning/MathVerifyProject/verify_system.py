"""
System Verification Script
Tests all modules and confirms system is ready
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    dependencies = {
        'math_verify': 'Math-Verify',
        'gradio': 'Gradio',
        'rich': 'Rich (CLI colors)',
        'tqdm': 'tqdm',
        'openai': 'OpenAI',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)
    
    return len(missing) == 0, missing


def check_modules():
    """Check if all project modules can be imported."""
    print("\nChecking project modules...")
    modules = [
        ('core_verification', 'MathVerifier'),
        ('benchmark_evaluation', 'MathVEvaluator'),
        ('ocr_input', 'HandwritingTranscriber'),
        ('main_interface', 'MathVerifyCLI'),
    ]
    
    all_ok = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{class_name} - ERROR: {e}")
            all_ok = False
    
    return all_ok


def check_repositories():
    """Check if all repositories are present."""
    print("\nChecking repositories...")
    repos = [
        'Math-Verify',
        'MATH-V',
        'MathVerse',
        'Math_Handwriting_OCR',
        'handwritten-math-transcription'
    ]
    
    all_present = True
    for repo in repos:
        if os.path.exists(repo):
            print(f"  ✓ {repo}")
        else:
            print(f"  ✗ {repo} - NOT FOUND")
            all_present = False
    
    return all_present


def test_verification():
    """Test core verification functionality."""
    print("\nTesting verification...")
    try:
        from core_verification import MathVerifier
        verifier = MathVerifier()
        
        # Test simple verification
        result = verifier.verify_answer("42", "42", return_details=True)
        if isinstance(result, dict) and 'valid' in result:
            print("  ✓ Verification API working")
            return True
        else:
            print("  ✗ Verification API returned unexpected format")
            return False
    except Exception as e:
        print(f"  ✗ Verification test failed: {e}")
        return False


def test_pipeline():
    """Test main pipeline."""
    print("\nTesting pipeline...")
    try:
        from main import MathVerifyPipeline
        pipeline = MathVerifyPipeline()
        print("  ✓ Pipeline initialized")
        return True
    except Exception as e:
        print(f"  ✗ Pipeline test failed: {e}")
        return False


def test_cli():
    """Test CLI interface."""
    print("\nTesting CLI...")
    try:
        from main_interface import MathVerifyCLI
        cli = MathVerifyCLI()
        print("  ✓ CLI initialized")
        return True
    except Exception as e:
        print(f"  ✗ CLI test failed: {e}")
        return False


def test_gradio():
    """Test Gradio interface."""
    print("\nTesting Gradio...")
    try:
        from main_interface import create_gradio_app
        app = create_gradio_app()
        print("  ✓ Gradio app created")
        return True
    except ImportError:
        print("  ⚠ Gradio not available (optional)")
        return True  # Not critical
    except Exception as e:
        print(f"  ✗ Gradio test failed: {e}")
        return False


def main():
    """Run all system checks."""
    print("="*60)
    print("MathVerifyProject - System Verification")
    print("="*60)
    
    all_checks = []
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    all_checks.append(('Dependencies', deps_ok))
    
    # Check modules
    modules_ok = check_modules()
    all_checks.append(('Modules', modules_ok))
    
    # Check repositories
    repos_ok = check_repositories()
    all_checks.append(('Repositories', repos_ok))
    
    # Test functionality
    verif_ok = test_verification()
    all_checks.append(('Verification', verif_ok))
    
    pipeline_ok = test_pipeline()
    all_checks.append(('Pipeline', pipeline_ok))
    
    cli_ok = test_cli()
    all_checks.append(('CLI', cli_ok))
    
    gradio_ok = test_gradio()
    all_checks.append(('Gradio', gradio_ok))
    
    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in all_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ System Ready!")
        print("\nAll modules are functioning correctly.")
        print("\nYou can now use:")
        print("  - Pipeline mode: python main.py --mode pipeline --gold '1/2' --pred '0.5'")
        print("  - CLI mode: python main.py --mode cli verify --gold '1/2' --pred '0.5'")
        print("  - Gradio UI: python main.py --mode gradio")
        print("  - API: from core_verification import MathVerifier")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        if missing:
            print(f"\nMissing dependencies: {', '.join(missing)}")
            print("Install with: pip install -r requirements.txt")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

