# Mathematical Reasoning Verification System

A 3-Model Parallel Verification System for mathematical reasoning with weighted consensus mechanism.

## Features

- **3 Parallel Models**: Symbolic (SymPy), LLM Logical, Ensemble (Multi-LLM voting)
- **Weighted Consensus**: 40% Symbolic, 35% LLM Logical, 25% Ensemble
- **Error Detection**: 10+ error types with classification
- **Explanation Generation**: Natural language error explanations
- **Automatic Correction**: Fixes fixable errors automatically
- **Interactive Dashboard**: Streamlit-based UI with real-time processing logs

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Streamlit Dashboard

```bash
streamlit run app.py
```

### Run Tests

```bash
pytest
```

## Architecture

```
INPUT → PARSING → PARALLEL (3 models) → CONSENSUS → OUTPUT
```

## Project Structure

```
.
├── app.py                      # Streamlit dashboard
├── core/                       # Core verification engine
│   ├── verification_engine.py
│   └── __init__.py
├── models/                     # Three verification models
│   ├── symbolic_verifier.py
│   ├── llm_logical_checker.py
│   ├── ensemble_neural_checker.py
│   └── __init__.py
├── consensus/                  # Consensus mechanism
│   ├── consensus_mechanism.py
│   └── __init__.py
├── utils/                      # Utilities
│   ├── error_classifier.py
│   ├── explanation_generator.py
│   ├── error_corrector.py
│   └── __init__.py
├── tests/                      # Unit tests
│   ├── test_symbolic_verifier.py
│   ├── test_llm_logical_checker.py
│   ├── test_ensemble_neural_checker.py
│   ├── test_consensus.py
│   ├── test_error_classifier.py
│   └── __init__.py
└── requirements.txt
```

## Example

**Input:**
- Problem: "Janet has 3 apples. She buys 2 more. She gives 1 away. How many?"
- Steps:
  - "Janet starts with 3 apples"
  - "She buys 2 more: 3 + 2 = 5 apples"
  - "She gives 1 away: 5 - 1 = 6 apples"  # ERROR

**Output:**
- Verdict: ERROR
- Confidence: 91.0%
- Error: Arithmetic error in step 3 (5 - 1 = 6 should be 4)
- Explanation: "You wrote 6, but 5 - 1 actually equals 4..."

## Performance Targets

- Overall Accuracy: 71.5%
- Error Detection: 78.3%
- Processing Time: < 4.1 seconds
- False Positive Rate: 2.1%

