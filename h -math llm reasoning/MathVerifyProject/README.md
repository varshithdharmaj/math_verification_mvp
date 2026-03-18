# MathVerifyProject

A comprehensive mathematical reasoning verification system that integrates multiple open-source research repositories in a modular, microservices-inspired architecture.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd MathVerifyProject

# Install all dependencies
pip install -r requirements.txt

# Install Math-Verify with ANTLR4 support
pip install math-verify[antlr4_13_2]

# Verify system is ready
python verify_system.py
```

**Expected Output:**
```
✅ System Ready!
```

### 2. Run Demo

**Gradio Web Interface (Recommended):**
```bash
python main.py --mode gradio
# Opens at http://localhost:7860
```

**CLI Interface:**
```bash
python main.py --mode cli verify --gold "1/2" --pred "0.5"
```

**Pipeline Mode:**
```bash
python main.py --mode pipeline --gold "1/2" --pred "0.5"
```

**Python API:**
```python
from core_verification import MathVerifier

verifier = MathVerifier()
result = verifier.verify_answer(gold="1/2", prediction="0.5")
print(result)  # True or False
```

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Interface Modes](#interface-modes)
- [Sample Problems](#sample-problems)
- [Project Structure](#project-structure)
- [Integrated Repositories](#integrated-repositories)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project integrates five research repositories to create a unified system for:
- **Mathematical Expression Verification** (Math-Verify)
- **Multimodal Math Benchmark Evaluation** (MATH-V, MathVerse)
- **Handwritten Math OCR** (handwritten-math-transcription, Math_Handwriting_OCR)

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Step 1: Install Dependencies

**Automatic Setup:**
```bash
python setup_and_verify.py
```

**Manual Setup:**
```bash
# Install main requirements
pip install -r requirements.txt

# Install Math-Verify with ANTLR4
pip install math-verify[antlr4_13_2]
```

### Step 2: Verify Installation

```bash
python verify_system.py
```

This will check:
- ✓ All dependencies installed
- ✓ All modules importable
- ✓ All repositories present
- ✓ Core verification working
- ✓ Pipeline functional
- ✓ CLI available
- ✓ Gradio available

**Expected Output:**
```
✅ System Ready!
```

## 🖥️ Interface Modes

### 1. Pipeline Mode

Direct pipeline usage for programmatic access.

**Usage:**
```bash
python main.py --mode pipeline --gold "1/2" --pred "0.5"
```

**Output:**
```
Verification: True
```

**Python Code:**
```python
from main import MathVerifyPipeline

pipeline = MathVerifyPipeline()
result = pipeline.process_math_problem(
    problem_text="What is 1/2?",
    model_answer="0.5",
    gold_answer="1/2"
)
print(f"Verification: {result['verification']}")
```

### 2. CLI Mode

Command-line interface with colored output and error classification.

**Usage:**
```bash
# Single verification
python main.py --mode cli verify --gold "1/2" --pred "0.5"

# Batch verification
python main.py --mode cli batch-verify --gold-file gold.txt --pred-file pred.txt

# Transcribe InkML
python main.py --mode cli transcribe --inkml file.inkml --model model.pth
```

**Features:**
- ✓ Colored output (green/red for correct/incorrect)
- ✓ Error classification
- ✓ Progress bars for batch operations
- ✓ Detailed parsing information

### 3. Gradio UI Mode

Web-based interface with LaTeX rendering and error taxonomy.

**Usage:**
```bash
python main.py --mode gradio
# or
python demo_gradio.py
```

**Features:**
- ✓ LaTeX rendering for math expressions
- ✓ Error taxonomy with colored tags and severity bars
- ✓ Image upload for OCR (handwritten math)
- ✓ Batch verification
- ✓ Step-by-step reasoning display

**Access:**
- Local: http://localhost:7860
- Share: Set `share=True` in `gradio_app.py`

### 4. API Mode

Python API for integration in scripts and notebooks.

**Usage:**
```python
from core_verification import MathVerifier

# Initialize
verifier = MathVerifier()

# Simple verification (returns boolean)
result = verifier.verify_answer(gold="1/2", prediction="0.5")
print(result)  # True or False

# Detailed verification (returns dict)
result = verifier.verify_answer(
    gold="1/2",
    prediction="0.5",
    return_details=True
)
print(result)
# {
#     'valid': True,
#     'gold': '1/2',
#     'prediction': '0.5',
#     'gold_parsed': <SymPy expression>,
#     'pred_parsed': <SymPy expression>,
#     'error_type': None,
#     'details': '...'
# }

# Batch verification
gold_answers = ["1/2", "2+2", "sqrt(4)"]
predictions = ["0.5", "4", "2"]
results = verifier.verify_batch(gold_answers, predictions)
print(results)  # [True, True, True]
```

**Jupyter/Colab:**
See `demo_notebook.ipynb` for interactive examples.

## 📊 Sample Problems

The `samples/` folder contains 5 example problems from the MATH-V dataset.

### Run Sample Tests

```bash
python samples/test_samples.py
```

This will:
- Load 5 samples from MATH-V
- Test Pipeline mode
- Test API mode
- Save results to `samples/outputs/`

### Sample Problems Included

1. **Counting Problem** (ID: 4)
   - Question: "How many different digits can you find in this picture?"
   - Answer: "6"

2. **Arithmetic Problem** (ID: 5)
   - Question: "Which number do you have to write in the last daisy?"
   - Answer: "61"

3. **Arithmetic Problem** (ID: 8)
   - Question: "The sums of the all the three numbers on each side of the triangle are equal..."
   - Answer: "2"

4. **Arithmetic Problem** (ID: 10)
   - Question: "Four people can be seated at a square table..."
   - Answer: "10"

5. **Solid Geometry Problem** (ID: 11)
   - Question: "Mike has built a construction from equal cubes..."
   - Answer: "7"

### View Sample Problems

```python
from samples.load_mathv_samples import load_mathv_samples

samples = load_mathv_samples(5)
for sample in samples:
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
```

## 📁 Project Structure

```
MathVerifyProject/
├── core_verification/          # Main verification pipeline (Math-Verify)
│   ├── __init__.py
│   └── verifier.py            # MathVerifier class wrapper
├── benchmark_evaluation/       # Benchmarking and visualization
│   ├── __init__.py
│   ├── mathv_evaluator.py     # MATH-V evaluator wrapper
│   └── mathverse_evaluator.py # MathVerse evaluator wrapper
├── ocr_input/                # OCR/handwritten input processing
│   ├── __init__.py
│   └── handwriting_transcriber.py  # Handwriting transcriber wrapper
├── main_interface/            # CLI and Gradio demo
│   ├── __init__.py
│   ├── cli.py                # Command-line interface
│   └── gradio_app.py         # Web interface
├── samples/                   # Sample problems and tests
│   ├── sample_problems.json  # 5 sample problems from MATH-V
│   ├── load_mathv_samples.py # Loader script
│   ├── test_samples.py       # Test all interfaces
│   └── outputs/              # Test results
├── Math-Verify/              # Cloned repository
├── MATH-V/                   # Cloned repository
├── MathVerse/                # Cloned repository
├── Math_Handwriting_OCR/     # Cloned repository
├── handwritten-math-transcription/  # Cloned repository
├── main.py                   # Main entry point
├── requirements.txt          # Consolidated dependencies
├── verify_system.py          # System verification script
├── setup_and_verify.py       # Automated setup script
└── README.md                 # This file
```

## 🔗 Integrated Repositories

### 1. Math-Verify (core_verification)
**Repository**: https://github.com/huggingface/Math-Verify.git

**Purpose**: Core mathematical expression verification engine

**Key Features**:
- Robust answer extraction from model outputs
- Advanced parsing capabilities (LaTeX, plain expressions, strings)
- Intelligent expression comparison (numerical, symbolic, sets, matrices)
- Highest accuracy on MATH dataset

**Integration**: `core_verification/verifier.py`
- `MathVerifier` class wraps Math-Verify functionality
- Methods: `verify_answer()`, `parse_expression()`, `verify_batch()`

### 2. MATH-V (benchmark_evaluation)
**Repository**: https://github.com/mathllm/MATH-V.git

**Purpose**: Multimodal mathematical reasoning benchmark evaluation

**Key Features**:
- 3,040 high-quality mathematical problems with visual contexts
- 16 distinct mathematical disciplines
- 5 levels of difficulty

**Integration**: `benchmark_evaluation/mathv_evaluator.py`
- `MathVEvaluator` class for benchmark evaluation

### 3. MathVerse (benchmark_evaluation)
**Repository**: https://github.com/ZrrSkywalker/MathVerse.git

**Purpose**: Visual math problem evaluation with diagram understanding

**Key Features**:
- 2,612 math problems with diagrams
- Six distinct problem versions per question
- Chain-of-Thought (CoT) evaluation strategy

**Integration**: `benchmark_evaluation/mathverse_evaluator.py`
- `MathVerseEvaluator` class for evaluation

### 4. handwritten-math-transcription (ocr_input)
**Repository**: https://github.com/johnkimdw/handwritten-math-transcription.git

**Purpose**: Handwritten mathematical equation transcription to LaTeX

**Key Features**:
- InkML file support
- Neural network-based transcription
- High accuracy on handwritten math

**Integration**: `ocr_input/handwriting_transcriber.py`
- `HandwritingTranscriber` class for OCR

### 5. Math_Handwriting_OCR (optional)
**Repository**: https://github.com/yixchen/Math_Handwriting_OCR.git

**Purpose**: Additional OCR capabilities (optional)

## 📚 Documentation

### Quick References

- **Python API**: See `PYTHON_API.md`
- **CLI Features**: See `CLI_FEATURES.md`
- **Gradio Features**: See `GRADIO_FEATURES.md`
- **Enhanced Features**: See `ENHANCED_FEATURES.md`
- **Quick Start**: See `QUICK_START.md`

### Example Files

- **API Examples**: `api_examples.py`
- **Jupyter Notebook**: `demo_notebook.ipynb`
- **Verification Demo**: `demo_verification.py`

## 🧪 Testing

### Run System Verification

```bash
python verify_system.py
```

### Test Sample Problems

```bash
python samples/test_samples.py
```

### Test Individual Components

```python
# Test verification
from core_verification import MathVerifier
verifier = MathVerifier()
result = verifier.verify_answer("1/2", "0.5")
assert result == True

# Test pipeline
from main import MathVerifyPipeline
pipeline = MathVerifyPipeline()
result = pipeline.process_math_problem("", "0.5", "1/2")
assert result['verification'] == True
```

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
pip install math-verify[antlr4_13_2]
```

**2. Math-Verify Not Found**
```bash
# Ensure Math-Verify is in the path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/Math-Verify/src"
# Or install as package
pip install math-verify[antlr4_13_2]
```

**3. Gradio Not Available**
```bash
# Install Gradio
pip install gradio>=4.0.0
```

**4. Missing Repositories**
- Ensure all repositories are cloned in the project directory
- Check that `Math-Verify/`, `MATH-V/`, etc. exist

### Getting Help

1. Run `python verify_system.py` to diagnose issues
2. Check error messages for specific missing dependencies
3. Review `requirements.txt` for all dependencies

## 📝 License

This project integrates multiple open-source repositories. Please refer to individual repository licenses:
- Math-Verify: See `Math-Verify/LICENSE`
- MATH-V: See `MATH-V/LICENSE`
- MathVerse: See `MathVerse/LICENSE`
- handwritten-math-transcription: See `handwritten-math-transcription/README.md`

## 🙏 Acknowledgments

This project integrates the following excellent open-source repositories:
- [Math-Verify](https://github.com/huggingface/Math-Verify) by HuggingFace
- [MATH-V](https://github.com/mathllm/MATH-V) by MathLLM
- [MathVerse](https://github.com/ZrrSkywalker/MathVerse) by ZrrSkywalker
- [handwritten-math-transcription](https://github.com/johnkimdw/handwritten-math-transcription) by johnkimdw
- [Math_Handwriting_OCR](https://github.com/yixchen/Math_Handwriting_OCR) by yixchen

## 🎯 Next Steps

1. **Run Verification**: `python verify_system.py`
2. **Try Gradio UI**: `python main.py --mode gradio`
3. **Test Samples**: `python samples/test_samples.py`
4. **Explore API**: See `api_examples.py`

---

**Status**: ✅ System Ready!

For detailed usage examples, see the documentation files in the project root.
