"""
Automated Setup and Verification Script
Detects, installs, and verifies all dependencies
"""

import sys
import os
import subprocess

def run_command(cmd, check=True):
    """Run a shell command."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.10+)")
        return False


def install_dependencies():
    """Install all dependencies."""
    print("\nInstalling dependencies...")
    
    # Check if requirements.txt exists
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.exists(req_file):
        print("✗ requirements.txt not found")
        return False
    
    # Install main requirements
    print("Installing main requirements...")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install -r {req_file}", check=False)
    if success:
        print("  ✓ Main requirements installed")
    else:
        print(f"  ⚠ Some packages may have failed: {stderr[:200]}")
    
    # Install Math-Verify with antlr4
    print("Installing Math-Verify...")
    success, _, _ = run_command(
        f"{sys.executable} -m pip install math-verify[antlr4_13_2]",
        check=False
    )
    if success:
        print("  ✓ Math-Verify installed")
    else:
        print("  ⚠ Math-Verify installation had issues (may already be installed)")
    
    return True


def verify_installation():
    """Verify that key packages are installed."""
    print("\nVerifying installation...")
    
    packages = {
        'math_verify': 'Math-Verify',
        'gradio': 'Gradio',
        'rich': 'Rich',
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - Not installed")
            all_ok = False
    
    return all_ok


def main():
    """Main setup and verification."""
    print("="*60)
    print("MathVerifyProject - Automated Setup and Verification")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        print("\nError: Python 3.10+ required")
        return False
    
    # Install dependencies
    install_dependencies()
    
    # Verify installation
    if verify_installation():
        print("\n✅ Dependencies installed and verified!")
        return True
    else:
        print("\n⚠️  Some dependencies may be missing. Try:")
        print("  pip install -r requirements.txt")
        print("  pip install math-verify[antlr4_13_2]")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

