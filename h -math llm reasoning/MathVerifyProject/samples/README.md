# Sample Problems from MATH-V Dataset

This folder contains sample problems from the MATH-V benchmark for testing all interfaces.

## Sample Problems

5 example problems from MATH-V covering different subjects:
- Counting
- Arithmetic
- Solid Geometry

## Files

- `sample_problems.json` - JSON file with 5 sample problems
- `load_mathv_samples.py` - Script to load samples from MATH-V dataset
- `test_samples.py` - Test script that runs all interfaces with samples
- `outputs/` - Directory containing test results

## Usage

### Test All Interfaces:
```bash
python samples/test_samples.py
```

### Load Samples Programmatically:
```python
from samples.load_mathv_samples import load_mathv_samples

samples = load_mathv_samples(5)
for sample in samples:
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
```

## Test Results

Results are saved in `samples/outputs/`:
- `pipeline_results.json` - Pipeline mode results
- `api_results.json` - API mode results

