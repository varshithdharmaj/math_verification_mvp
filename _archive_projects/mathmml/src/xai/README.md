# Explainable AI (XAI) Module

This module provides comprehensive explainability features for the Math Verification System, helping users understand why the models made their decisions.

## Features

### 1. **Per-Verifier Explanations**
- **Symbolic Verifier**: Explains arithmetic evaluation and result matching
- **LLM Logical Checker**: Shows which heuristics were checked and what patterns were found
- **Ensemble Checker**: Displays voting breakdown and agreement analysis
- **ML Classifier**: Shows class probabilities and top predictions

### 2. **Consensus Explanations**
- Weighted contribution analysis
- Agreement type reasoning (UNANIMOUS/MAJORITY/MIXED)
- Verifier influence ranking
- Confidence breakdown

### 3. **Visualizations**
- Confidence factor charts
- Class probability distributions
- Verifier contribution graphs
- Agreement pie charts
- Reasoning chain visualization

### 4. **Feature Importance** (Advanced)
- Token-level attention analysis for ML models
- Input importance scoring
- Highlighted important tokens

## Usage

### In Streamlit UI

After running a verification, expand the **"🔍 Explainable AI - Why this verdict?"** section to see:

1. **Overview Tab**: Consensus summary and reasoning
2. **Per-Verifier Tab**: Detailed explanations for each model
3. **Consensus Tab**: Weighted analysis and contributions
4. **Visualizations Tab**: Interactive charts and graphs

### In Python Code

```python
from src.xai.explainer import XAIExplainer

# Initialize explainer
explainer = XAIExplainer()

# Explain a verifier's decision
explanation = explainer.explain_verifier_decision(
    verifier_name='symbolic',
    step='5 + 3 = 8',
    problem='Add 5 and 3',
    result=verifier_result,
    prev_steps=[]
)

# Explain consensus
consensus_explanation = explainer.explain_consensus(
    consensus=consensus_result,
    verifier_results=all_verifier_results
)
```

## Explanation Structure

Each explanation includes:

- **Reasoning**: Natural language explanation of the decision
- **Evidence**: Specific facts or patterns that led to the verdict
- **Confidence Factors**: Breakdown of what contributes to confidence
- **Key Factors**: Most important factors in the decision
- **Visualizations**: Charts showing probabilities and contributions

## Example Output

```python
{
    'verifier': 'symbolic',
    'verdict': 'VALID',
    'reasoning': [
        "The symbolic verifier verified the arithmetic by evaluating the expression..."
    ],
    'evidence': [
        "Expression 5 + 3 = 8.0 is correct"
    ],
    'confidence_factors': {
        'expression_validity': 0.95,
        'result_match': 0.9,
        'calculation_precision': 0.92
    },
    'key_factors': [
        "Arithmetic calculation verified"
    ]
}
```

## Benefits

1. **Transparency**: Understand why each model made its decision
2. **Trust**: Build confidence in the system's verdicts
3. **Debugging**: Identify which verifiers are contributing most
4. **Education**: Learn about mathematical reasoning verification
5. **Improvement**: Identify areas where models need better training

## Future Enhancements

- SHAP values for feature importance
- LIME explanations for local interpretability
- Counterfactual examples ("What if...")
- Attention heatmaps for transformer models
- Rule extraction from neural models

