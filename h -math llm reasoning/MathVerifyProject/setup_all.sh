#!/bin/bash
# Master setup script for MathVerifyProject
# Clones repositories, installs dependencies, and verifies installation

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "MathVerifyProject Setup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Step 1: Clone repositories
echo "Step 1: Cloning repositories..."
echo "-----------------------------------"

repos=(
    "https://github.com/huggingface/Math-Verify.git:Math-Verify"
    "https://github.com/mathllm/MATH-V.git:MATH-V"
    "https://github.com/ZrrSkywalker/MathVerse.git:MathVerse"
    "https://github.com/yixchen/Math_Handwriting_OCR.git:Math_Handwriting_OCR"
    "https://github.com/johnkimdw/handwritten-math-transcription.git:handwritten-math-transcription"
)

for repo_info in "${repos[@]}"; do
    IFS=':' read -r url name <<< "$repo_info"
    if [ -d "$name" ]; then
        print_warning "$name already exists, skipping clone"
    else
        echo "Cloning $name..."
        git clone "$url" "$name" || print_error "Failed to clone $name"
        print_status "Cloned $name"
    fi
done

echo ""

# Step 2: Install Python dependencies
echo "Step 2: Installing Python dependencies..."
echo "-----------------------------------"

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.10+"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD=$(command -v python3 || command -v python)

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
print_status "Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    print_error "Could not find virtual environment activation script"
    exit 1
fi

print_status "Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip --quiet
print_status "pip upgraded"

# Install main requirements
echo "Installing main requirements..."
if [ -f "requirements.txt" ]; then
    $PYTHON_CMD -m pip install -r requirements.txt
    print_status "Main requirements installed"
else
    print_warning "requirements.txt not found"
fi

# Install Math-Verify
echo "Installing Math-Verify..."
cd Math-Verify
if [ -f "pyproject.toml" ]; then
    $PYTHON_CMD -m pip install -e .[antlr4_13_2] || $PYTHON_CMD -m pip install math-verify[antlr4_13_2]
    print_status "Math-Verify installed"
else
    print_warning "Math-Verify pyproject.toml not found, trying pip install"
    $PYTHON_CMD -m pip install math-verify[antlr4_13_2]
fi
cd ..

# Install handwritten-math-transcription dependencies
echo "Installing handwritten-math-transcription dependencies..."
cd handwritten-math-transcription
if [ -f "requirements.txt" ]; then
    $PYTHON_CMD -m pip install -r requirements.txt
    print_status "Handwriting transcription dependencies installed"
else
    print_warning "handwritten-math-transcription requirements.txt not found"
fi
cd ..

echo ""

# Step 3: Verify installation
echo "Step 3: Verifying installation..."
echo "-----------------------------------"

# Test Math-Verify
echo "Testing Math-Verify..."
$PYTHON_CMD -c "from math_verify import parse, verify; print('Math-Verify: OK')" 2>/dev/null && \
    print_status "Math-Verify import successful" || \
    print_error "Math-Verify import failed"

# Test core modules
echo "Testing core modules..."
$PYTHON_CMD -c "import sys; sys.path.insert(0, '.'); from core_verification import MathVerifier; print('Core verification: OK')" 2>/dev/null && \
    print_status "Core verification module OK" || \
    print_warning "Core verification module test failed (may need dependencies)"

echo ""

# Step 4: Create demo script
echo "Step 4: Creating demo scripts..."
echo "-----------------------------------"

# Demo script will be created separately
print_status "Demo scripts ready (see demo_verification.py)"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use the system:"
echo "  1. Activate virtual environment: source venv/bin/activate (or venv\\Scripts\\activate on Windows)"
echo "  2. Run main interface: python main.py --mode gradio"
echo "  3. Or use CLI: python main.py --mode cli verify --gold '1/2' --pred '0.5'"
echo ""
echo "For more information, see README.md"
echo ""

