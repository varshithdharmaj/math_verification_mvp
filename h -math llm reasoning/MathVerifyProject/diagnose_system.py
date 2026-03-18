"""
Comprehensive System Diagnostic
Checks everything and tells you exactly what's wrong
"""

import sys
import os

print("=" * 70)
print("MathVerifyProject - Complete System Diagnostic")
print("=" * 70)
print()

# Track issues
issues = []
working = []

# 1. Check Python version
print("1. Checking Python version...")
version = sys.version_info
if version.major >= 3 and version.minor >= 8:
    print(f"   ✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
    working.append("Python version")
else:
    print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
    issues.append("Python version too old")

# 2. Check working directory
print("\n2. Checking working directory...")
cwd = os.getcwd()
print(f"   Current directory: {cwd}")
if "MathVerifyProject" in cwd or os.path.exists("main.py"):
    print("   ✓ In correct directory")
    working.append("Directory")
else:
    print("   ⚠ May not be in project root")
    issues.append("Wrong directory")

# 3. Check main.py exists
print("\n3. Checking main.py...")
if os.path.exists("main.py"):
    print("   ✓ main.py exists")
    working.append("main.py")
else:
    print("   ✗ main.py NOT FOUND")
    issues.append("main.py missing")

# 4. Check repositories
print("\n4. Checking repositories...")
repos = {
    "Math-Verify": "Math-Verify",
    "MATH-V": "MATH-V",
    "MathVerse": "MathVerse",
    "handwritten-math-transcription": "handwritten-math-transcription"
}
for name, path in repos.items():
    if os.path.exists(path):
        print(f"   ✓ {name} found")
        working.append(name)
    else:
        print(f"   ✗ {name} NOT FOUND")
        issues.append(f"{name} missing")

# 5. Check core modules
print("\n5. Checking core modules...")
modules_to_check = [
    ("core_verification", "MathVerifier"),
    ("main_interface", "MathVerifyCLI"),
]
for module_name, class_name in modules_to_check:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"   ✓ {module_name}.{class_name}")
        working.append(f"{module_name}")
    except Exception as e:
        print(f"   ✗ {module_name}.{class_name} - {e}")
        issues.append(f"{module_name} import failed")

# 6. Check dependencies
print("\n6. Checking dependencies...")
deps = {
    "sympy": "SymPy",
    "gradio": "Gradio",
    "torch": "PyTorch",
    "numpy": "NumPy",
}
for module, name in deps.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
        working.append(name)
    except ImportError:
        print(f"   ✗ {name} - NOT INSTALLED")
        issues.append(f"{name} missing")

# 7. Test core verification
print("\n7. Testing core verification...")
try:
    from core_verification import MathVerifier
    verifier = MathVerifier()
    result = verifier.verify_answer("1/2", "0.5")
    print(f"   ✓ Verification works! (1/2 vs 0.5 = {result})")
    working.append("Verification")
except Exception as e:
    print(f"   ✗ Verification failed: {e}")
    issues.append(f"Verification error: {e}")

# 8. Test CLI
print("\n8. Testing CLI...")
try:
    from main_interface import MathVerifyCLI
    cli = MathVerifyCLI()
    print("   ✓ CLI initialized")
    working.append("CLI")
except Exception as e:
    print(f"   ✗ CLI failed: {e}")
    issues.append(f"CLI error: {e}")

# 9. Test Gradio
print("\n9. Testing Gradio...")
try:
    import gradio as gr
    from main_interface.gradio_app import create_gradio_app
    app = create_gradio_app()
    print("   ✓ Gradio app created")
    working.append("Gradio")
except Exception as e:
    print(f"   ✗ Gradio failed: {e}")
    issues.append(f"Gradio error: {e}")

# 10. Test pipeline mode
print("\n10. Testing pipeline mode...")
try:
    from main import MathVerifyPipeline
    pipeline = MathVerifyPipeline()
    result = pipeline.process_math_problem("", "0.5", "1/2")
    print(f"   ✓ Pipeline works! (Result: {result['verification']})")
    working.append("Pipeline")
except Exception as e:
    print(f"   ✗ Pipeline failed: {e}")
    issues.append(f"Pipeline error: {e}")

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print(f"\n✓ Working: {len(working)}/{len(working) + len(issues)}")
for item in working:
    print(f"  ✓ {item}")

if issues:
    print(f"\n✗ Issues found: {len(issues)}")
    for item in issues:
        print(f"  ✗ {item}")
else:
    print("\n✓ No issues found! System should be working.")

# Recommendations
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

if issues:
    print("\nTo fix issues:")
    
    if any("missing" in i.lower() or "NOT INSTALLED" in str(i) for i in issues):
        print("\n1. Install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("   pip install sympy gradio torch numpy")
    
    if any("import" in i.lower() for i in issues):
        print("\n2. Check Python path:")
        print("   Make sure you're in the MathVerifyProject directory")
        print("   Current directory:", os.getcwd())
    
    if any("Gradio" in i for i in issues):
        print("\n3. For Gradio issues:")
        print("   pip install gradio")
        print("   Or use CLI instead: python main.py --mode cli verify --gold '1/2' --pred '0.5'")
    
    if any("Verification" in i for i in issues):
        print("\n4. For verification issues:")
        print("   pip install math-verify[antlr4_13_2]")
        print("   pip install latex2sympy2_extended")
else:
    print("\n✓ System is ready!")
    print("\nTry these commands:")
    print("  CLI:    python main.py --mode cli verify --gold '1/2' --pred '0.5'")
    print("  Gradio: python simple_launch.py")
    print("  API:    python -c \"from core_verification import MathVerifier; v = MathVerifier(); print(v.verify_answer('1/2', '0.5'))\"")

print("\n" + "=" * 70)

