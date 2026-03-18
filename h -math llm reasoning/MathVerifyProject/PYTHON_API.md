# Python API Documentation

## Quick Start

```python
from core_verification import MathVerifier

verifier = MathVerifier()
result = verifier.verify_answer(gold="1/2", prediction="0.5")
print(result)  # True or False
```

## API Reference

### MathVerifier Class

#### `verify_answer(gold, prediction, return_details=False)`

Verify if a prediction matches the gold answer.

**Parameters:**
- `gold` (str): Gold/correct answer string
- `prediction` (str): Model prediction string
- `return_details` (bool): If True, returns detailed dict instead of boolean

**Returns:**
- If `return_details=False`: `bool` - True if correct, False otherwise
- If `return_details=True`: `dict` with keys:
  - `valid` (bool): Verification result
  - `gold` (str): Original gold answer
  - `prediction` (str): Original prediction
  - `gold_parsed`: Parsed gold expression or None
  - `pred_parsed`: Parsed prediction expression or None
  - `error_type` (str or None): Error classification if incorrect
  - `details` (str): Additional details

**Example:**
```python
# Simple verification
result = verifier.verify_answer(gold="1/2", prediction="0.5")
# Returns: True

# Detailed verification
result = verifier.verify_answer(
    gold="1/2", 
    prediction="0.5",
    return_details=True
)
# Returns: {
#     'valid': True,
#     'gold': '1/2',
#     'prediction': '0.5',
#     'gold_parsed': <SymPy expression>,
#     'pred_parsed': <SymPy expression>,
#     'error_type': None,
#     'details': '...'
# }
```

#### `verify_batch(gold_answers, predictions, return_details=False)`

Verify multiple answers at once.

**Parameters:**
- `gold_answers` (list): List of gold answer strings
- `predictions` (list): List of prediction strings
- `return_details` (bool): If True, returns list of detailed dicts

**Returns:**
- If `return_details=False`: `list[bool]` - List of verification results
- If `return_details=True`: `list[dict]` - List of detailed results

**Example:**
```python
gold_answers = ["1/2", "2+2", "sqrt(4)"]
predictions = ["0.5", "4", "2"]

# Simple batch
results = verifier.verify_batch(gold_answers, predictions)
# Returns: [True, True, True]

# Detailed batch
results = verifier.verify_batch(
    gold_answers, 
    predictions,
    return_details=True
)
# Returns: [{'valid': True, ...}, {'valid': True, ...}, ...]
```

#### `parse_expression(expression, is_gold=False)`

Parse a mathematical expression from text.

**Parameters:**
- `expression` (str): String containing mathematical expression
- `is_gold` (bool): Whether this is a gold answer

**Returns:**
- `list`: List of parsed expressions (SymPy objects or strings)

**Example:**
```python
parsed = verifier.parse_expression("1/2")
# Returns: [<SymPy Rational(1, 2)>]
```

## Usage Examples

### Example 1: Basic Usage

```python
from core_verification import MathVerifier

verifier = MathVerifier()
result = verifier.verify_answer(gold="1/2", prediction="0.5")
print(f"Correct: {result}")
```

### Example 2: Detailed Results

```python
verifier = MathVerifier()
result = verifier.verify_answer(
    gold="1/2",
    prediction="0.5",
    return_details=True
)

print(f"Valid: {result['valid']}")
print(f"Gold Parsed: {result['gold_parsed']}")
print(f"Prediction Parsed: {result['pred_parsed']}")
```

### Example 3: Batch Processing

```python
gold_answers = ["1/2", "2+2", "sqrt(4)"]
predictions = ["0.5", "4", "2"]

results = verifier.verify_batch(gold_answers, predictions)
accuracy = sum(results) / len(results) * 100
print(f"Accuracy: {accuracy:.2f}%")
```

### Example 4: Error Analysis

```python
results = verifier.verify_batch(
    gold_answers,
    predictions,
    return_details=True
)

errors = [r for r in results if not r['valid']]
for error in errors:
    print(f"Error: {error['error_type']}")
    print(f"  Gold: {error['gold']}")
    print(f"  Prediction: {error['prediction']}")
```

### Example 5: Custom Configuration

```python
from math_verify import ExprExtractionConfig, LatexExtractionConfig

verifier = MathVerifier(
    gold_extraction_config=[ExprExtractionConfig()],
    pred_extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
    float_rounding=8,
    numeric_precision=20,
    strict=False
)

result = verifier.verify_answer("1/2", "0.5")
```

### Example 6: Integration with Pandas

```python
import pandas as pd
from core_verification import MathVerifier

verifier = MathVerifier()

df = pd.DataFrame({
    'gold': ["1/2", "2+2", "sqrt(4)"],
    'prediction': ["0.5", "4", "2"]
})

df['correct'] = df.apply(
    lambda row: verifier.verify_answer(
        gold=row['gold'],
        prediction=row['prediction']
    ),
    axis=1
)

print(df)
print(f"Accuracy: {df['correct'].mean() * 100:.2f}%")
```

### Example 7: Full Pipeline

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

## Jupyter/Colab Integration

### Quick Start in Notebook

```python
# Cell 1: Setup
from core_verification import MathVerifier
verifier = MathVerifier()

# Cell 2: Verify
result = verifier.verify_answer(gold="1/2", prediction="0.5", return_details=True)
print(result)

# Cell 3: Batch
gold_answers = ["1/2", "2+2", "sqrt(4)"]
predictions = ["0.5", "4", "2"]
results = verifier.verify_batch(gold_answers, predictions, return_details=True)

for r in results:
    print(f"{'✓' if r['valid'] else '✗'} {r['gold']} vs {r['prediction']}")
```

## Files

- `api_examples.py` - Complete examples file
- `demo_notebook.ipynb` - Jupyter notebook with examples
- `core_verification/verifier.py` - Main API implementation

## See Also

- `README.md` - Complete project documentation
- `CLI_FEATURES.md` - Command-line interface
- `GRADIO_FEATURES.md` - Web interface

