# Quick API Guide - MathVerifyProject

## 🚀 One-Line Usage

```python
from core_verification import MathVerifier

verifier = MathVerifier()
result = verifier.verify_answer(gold="1/2", prediction="0.5")
print(result)  # True or False
```

## 📚 Common Patterns

### 1. Simple Verification (Boolean)
```python
verifier = MathVerifier()
is_correct = verifier.verify_answer(gold="1/2", prediction="0.5")
```

### 2. Detailed Verification (Dict)
```python
result = verifier.verify_answer(
    gold="1/2",
    prediction="0.5",
    return_details=True
)
# Returns: {'valid': True, 'gold': '1/2', 'prediction': '0.5', ...}
```

### 3. Batch Verification
```python
gold_answers = ["1/2", "2+2", "sqrt(4)"]
predictions = ["0.5", "4", "2"]
results = verifier.verify_batch(gold_answers, predictions)
```

### 4. Batch with Details
```python
results = verifier.verify_batch(
    gold_answers,
    predictions,
    return_details=True
)
```

## 📖 Full Documentation

See `PYTHON_API.md` for complete API reference and examples.

## 📓 Jupyter Notebook

See `demo_notebook.ipynb` for interactive examples.

## 💡 Examples File

Run `python api_examples.py` to see all examples.

