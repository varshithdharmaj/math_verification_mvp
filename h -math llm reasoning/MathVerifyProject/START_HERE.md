# 🚀 START HERE - How to Access the Interface

## Quick Start (3 Steps)

### Step 1: Open Terminal
Open PowerShell or Command Prompt in the project folder:
```
C:\Users\Varshith Dharmaj\Downloads\h -math llm reasoning\MathVerifyProject
```

### Step 2: Run This Command
```bash
python launch_with_debug.py
```

### Step 3: Look for This Line
```
Running on local URL:  http://127.0.0.1:7860
```

Then **copy that URL** and paste it into your browser!

---

## If That Doesn't Work

### Option A: Use CLI (Command Line)
```bash
python main.py --mode cli verify --gold "1/2" --pred "0.5"
```

### Option B: Use Python API
```python
from core_verification import MathVerifier
verifier = MathVerifier()
result = verifier.verify_answer(gold="1/2", prediction="0.5")
print(result)
```

---

## Common Issues

**"Can't reach this page"**
- Make sure the server is running (you should see "Running on...")
- Try `http://127.0.0.1:7860` instead of `localhost:7860`
- Check if port 7860 is blocked by firewall

**"Module not found"**
- Run: `python -m pip install gradio`

**Server won't start**
- Check terminal for error messages
- Try: `python test_gradio_simple.py` to test Gradio

---

## Need Help?

1. Run: `python launch_with_debug.py`
2. Copy ALL the output from terminal
3. Share it so I can help diagnose

