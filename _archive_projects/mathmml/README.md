# Math Verification System

A comprehensive 4-model verification pipeline for mathematical reasoning that combines symbolic computation, logical heuristics, ensemble LLM voting, and a trainable transformer classifier.

## Overview

This system verifies mathematical solution steps using four complementary verifiers:

1. **SymbolicVerifier** (SymPy) - Arithmetic/algebra verification
2. **LLMLogicalChecker** - Heuristic-based logical consistency checks
3. **EnsembleNeuralChecker** - Multi-LLM voting mechanism
4. **MLStepClassifier** - Trainable transformer (RoBERTa/DeBERTa) for step-level error classification

The system computes a weighted consensus from all four models to produce a final verdict with confidence scores.

## Architecture

```
┌─────────┐
│  INPUT  │  Problem + Steps
└────┬────┘
     │
     ▼
┌─────────┐
│ PARSING │  Extract expressions, context
└────┬────┘
     │
     ▼
┌─────────────────────────────────────┐
│      PARALLEL MODELS               │
│  (All 4 run simultaneously)        │
│  ┌──────────┐  ┌──────────┐       │
│  │Symbolic  │  │LLM Logic │       │
│  │Verifier  │  │ Checker  │       │
│  │(40%)     │  │ (35%)    │       │
│  └──────────┘  └──────────┘       │
│  ┌──────────┐  ┌──────────┐       │
│  │Ensemble  │  │ML Class. │       │
│  │Checker   │  │          │       │
│  │(20%)     │  │ (25%)    │       │
│  └──────────┘  └──────────┘       │
└─────────────┬──────────────────────┘
              │
              ▼
     ┌─────────────────┐
     │   CONSENSUS     │  Weighted voting
     │  (error_score) │  Threshold: 0.50
     └────────┬────────┘
              │
              ▼
         ┌─────────┐
         │ OUTPUT  │  Verdict + Confidence
         └─────────┘
```

## Features

- **4-Model Verification**: Parallel execution of complementary verifiers
- **Weighted Consensus**: Configurable weights for model combination
- **Error Taxonomy**: 10+ error types with severity and fixability
- **Auto-Correction**: Automatic fixing for arithmetic and sign errors
- **Explainable AI (XAI)**: Comprehensive explanations for all model decisions
- **Interactive Visualizations**: Charts, graphs, and reasoning chains
- **Streamlit UI**: Interactive dashboard with flowchart and live logs
- **Multi-Model Support**: GPT, Gemini, Llama, Claude integration
- **Colab Support**: Ready-to-run notebooks for training and demo
- **Comprehensive Testing**: Unit tests for all components

## Getting Started

### Installation

```bash
# Clone repository
git clone <repository-url>
cd mathmml

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

1. **Prepare Data**:
```bash
python -c "from src.data.loaders import prepare_training_data; prepare_training_data()"
```

2. **Train Classifier** (optional):
```bash
python scripts/train_classifier.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --model_name roberta-base \
    --output_dir models/checkpoints/
```

3. **Launch Streamlit UI**:
```bash
streamlit run src/ui/streamlit_app.py
```

4. **Run Tests**:
```bash
pytest tests/
```

## Project Structure

```
mathmml/
├── src/
│   ├── data/
│   │   └── loaders.py              # GSM8K/Math500 loaders
│   ├── models/
│   │   ├── symbolic_verifier.py    # SymPy verifier
│   │   ├── llm_logical_checker.py  # Heuristic checker
│   │   ├── ensemble_checker.py     # Multi-LLM voting
│   │   └── ml_step_classifier.py   # Transformer classifier
│   ├── pipeline/
│   │   ├── consensus.py            # Weighted consensus
│   │   └── classifier_head.py      # Custom classifier head
│   ├── utils/
│   │   ├── error_taxonomy.py       # Error types
│   │   ├── explanation.py          # NLG explanations
│   │   ├── correction.py           # Auto-correction
│   │   ├── logging_utils.py        # Session logging
│   │   └── llm_providers.py        # Multi-model support
│   ├── xai/
│   │   ├── explainer.py            # XAI explanation engine
│   │   ├── visualizations.py       # Interactive charts
│   │   └── feature_importance.py   # Feature analysis
│   └── ui/
│       ├── streamlit_app.py        # Dashboard UI
│       └── interactive_flowchart.py # Dynamic flowchart
├── notebooks/
│   ├── 01_data_preview.ipynb       # Data exploration
│   ├── 02_train_ml_classifier.ipynb # Training notebook
│   ├── 03_eval_and_ablation.ipynb # Evaluation
│   └── 04_colab_demo.ipynb        # Colab demo
├── scripts/
│   ├── train_classifier.py         # Training script
│   ├── infer_classifier.py         # Inference script
│   └── colab_launch_streamlit.py   # Colab launcher
├── tests/
│   ├── test_symbolic.py
│   ├── test_llm_logic.py
│   ├── test_ensemble.py
│   ├── test_classifier.py
│   └── test_consensus.py
├── requirements.txt
└── README.md
```

## Explainable AI (XAI)

The system includes comprehensive explainability features:

- **Per-Verifier Explanations**: Understand why each model made its decision
- **Consensus Analysis**: See how models agree/disagree and their weighted contributions
- **Visualizations**: Interactive charts showing probabilities, confidence factors, and contributions
- **Reasoning Chains**: Step-by-step explanation of the decision process

After running a verification in the Streamlit UI, expand the **"🔍 Explainable AI"** section to see detailed explanations with visualizations.

See `src/xai/README.md` for detailed documentation.

## Usage Examples

### Python API

```python
from src.models.symbolic_verifier import SymbolicVerifier
from src.models.llm_logical_checker import LLMLogicalChecker
from src.models.ensemble_checker import EnsembleNeuralChecker
from src.models.ml_step_classifier import MLStepClassifierWrapper
from src.pipeline.consensus import ConsensusEngine

# Initialize verifiers
verifiers = {
    'symbolic': SymbolicVerifier(),
    'llm_logical': LLMLogicalChecker(use_api=False),
    'ensemble': EnsembleNeuralChecker(use_apis=False),
    'ml_classifier': MLStepClassifierWrapper(model_path="models/checkpoints/")
}

# Initialize consensus engine
consensus_engine = ConsensusEngine()

# Verify a step
problem = "Natalia sold clips to 48 of her friends in April."
step = "Natalia sold 48/2 = 24 clips in May."
prev_steps = []

consensus = consensus_engine.verify_step(step, problem, prev_steps, verifiers)
print(f"Verdict: {consensus['final_verdict']}")
print(f"Confidence: {consensus['overall_confidence']:.3f}")
```

### Command Line

```bash
# Inference with classifier
python scripts/infer_classifier.py \
    --model_path models/checkpoints/ \
    --problem "Problem text" \
    --step "Step text" \
    --prev_steps "Previous context"
```

## Model Weights (Default)

- Symbolic: 0.40 (40%)
- LLM Logical: 0.35 (35%)
- Ensemble: 0.20 (20%)
- ML Classifier: 0.25 (25%)

Weights can be customized in `ConsensusEngine` initialization.

## Consensus Algorithm

The system uses a weighted error score with a threshold of 0.50:

```
error_score = Σ(weight × confidence) for all ERROR verdicts
final_verdict = ERROR if error_score > 0.50 else VALID
```

Agreement types:
- **UNANIMOUS ✓✓✓**: All 4 models agree
- **MAJORITY (3/4) ✓✓**: 3 out of 4 models agree
- **MIXED ✓**: 2-2 split or other mixed agreement

## Error Taxonomy

The system detects 10+ error types:

- `correct` - No error
- `arithmetic_error` - Incorrect calculation
- `logical_error` - Logical inconsistency
- `operation_mismatch` - Wrong operation
- `conceptual_error` - Conceptual misunderstanding
- `notation_error` - Notation issues
- `sign_error` - Sign errors
- `unit_error` - Unit mismatches
- `order_ops_error` - Order of operations
- `semantic_error` - Semantic inconsistencies

## Training the Classifier

1. **Prepare Data**:
```python
from src.data.loaders import prepare_training_data
train, val, test = prepare_training_data()
```

2. **Train**:
```bash
python scripts/train_classifier.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --model_name roberta-base \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 5
```

3. **Evaluate**:
Use `notebooks/03_eval_and_ablation.ipynb` for metrics and confusion matrices.

## Google Colab

### Training Notebook
Open `notebooks/02_train_ml_classifier.ipynb` in Colab to:
- Install dependencies
- Prepare data
- Train the classifier
- Test inference

### Demo Notebook
Open `notebooks/04_colab_demo.ipynb` to:
- Launch Streamlit app
- Get public ngrok URL
- Test with example problems

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test:
```bash
pytest tests/test_symbolic.py
```

## Limitations

- Symbolic verifier works best with simple arithmetic/algebra
- LLM checker requires API keys for full functionality (mock mode available)
- ML classifier needs training data for optimal performance
- Auto-correction is limited to arithmetic and sign errors

## Roadmap

- [ ] Support for more complex mathematical domains (calculus, geometry)
- [ ] Improved auto-correction for logical errors
- [ ] Real-time model fine-tuning
- [ ] Multi-language support
- [ ] Integration with more LLM providers

## Citation

If you use this system, please cite:

```bibtex
@software{mathmml2024,
  title={Math Verification System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/mathmml}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- GSM8K dataset: [Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)
- HuggingFace Transformers
- SymPy for symbolic computation
