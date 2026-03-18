# Quick Start Guide - MathVerifyProject

## ✅ What's Working

1. **Project Structure**: All modules created and organized
2. **Code Integration**: All wrapper classes and interfaces implemented
3. **Math-Verify Installed**: Package is installed (version 0.8.0)
4. **Module Imports**: Core modules can be imported successfully

## ⚠️ Current Status

The system is set up but needs dependency installation for full functionality:

### Required Dependencies

1. **Math-Verify Dependencies**:
   ```bash
   pip install latex2sympy2_extended==1.10.2
   pip install antlr4-python3-runtime==4.13.2
   ```

2. **Other Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Setup Steps

### Option 1: Automated Setup (Recommended)

**Windows:**
```cmd
setup_all.bat
```

**Linux/Mac:**
```bash
chmod +x setup_all.sh
./setup_all.sh
```

### Option 2: Manual Setup

1. **Install Math-Verify with dependencies:**
   ```bash
   pip install math-verify[antlr4_13_2]
   ```

2. **Install other requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install repository-specific dependencies:**
   ```bash
   cd handwritten-math-transcription
   pip install -r requirements.txt
   cd ..
   ```

## 📝 Usage Examples

### 1. Basic Verification (after dependencies installed)

```python
from core_verification import MathVerifier

verifier = MathVerifier()
result = verifier.verify_answer(gold="1/2", prediction="0.5")
print(f"Correct: {result}")
```

### 2. Command Line Interface

```bash
python main.py --mode cli verify --gold "1/2" --pred "0.5"
```

### 3. Gradio Web Interface

```bash
python main.py --mode gradio
```

Then open: http://localhost:7860

### 4. Full Pipeline

```python
from main import MathVerifyPipeline

pipeline = MathVerifyPipeline()
result = pipeline.process_math_problem(
    problem_text="What is 1/2?",
    model_answer="0.5",
    gold_answer="1/2"
)
```

## 🔧 Troubleshooting

### Issue: Parse returns empty list

**Solution**: Install Math-Verify dependencies:
```bash
pip install latex2sympy2_extended==1.10.2
pip install antlr4-python3-runtime==4.13.2
```

### Issue: Import errors

**Solution**: Ensure you're in the MathVerifyProject directory and all paths are correct.

### Issue: OCR model not found

**Solution**: Model checkpoints are in `handwritten-math-transcription/model/`. Use:
```python
transcriber = HandwritingTranscriber(
    model_path="handwritten-math-transcription/model/model_best_0.pth"
)
```

## 📚 Next Steps

1. Install dependencies (see above)
2. Test verification: `python demo_verification.py`
3. Try CLI: `python main.py --mode cli verify --gold "1/2" --pred "0.5"`
4. Launch Gradio: `python main.py --mode gradio`

## 📖 Full Documentation

See `README.md` for complete documentation including:
- Detailed integration plan
- All function signatures
- Data flow architecture
- Configuration options

