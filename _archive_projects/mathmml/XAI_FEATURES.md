# Explainable AI (XAI) Features

## Overview

The Math Verification System now includes comprehensive Explainable AI features that help users understand **why** each model made its decision, not just **what** the decision was.

## Key Features

### 1. **Per-Verifier Explanations**

Each verifier provides detailed explanations:

#### Symbolic Verifier
- Shows the extracted expression
- Displays calculated vs. claimed result
- Explains arithmetic verification process
- Confidence factors: expression validity, result match, calculation precision

#### LLM Logical Checker
- Lists which heuristics were checked
- Shows detected patterns (contradictions, operation mismatches, etc.)
- Explains logical consistency analysis
- Confidence factors: heuristic match, pattern confidence

#### Ensemble Checker
- Displays voting breakdown (X/Y models voted ERROR/VALID)
- Shows agreement analysis
- Explains majority/minority reasoning
- Confidence factors: agreement ratio, consensus strength

#### ML Classifier
- Shows top 3 predicted classes with probabilities
- Displays class probability distribution
- Explains prediction confidence
- Confidence factors: prediction confidence, class separation, model certainty

### 2. **Consensus Explanations**

- **Agreement Analysis**: Explains why verdict is UNANIMOUS/MAJORITY/MIXED
- **Weighted Contributions**: Shows which verifiers had the most influence
- **Reasoning Chain**: Step-by-step explanation of how consensus was reached
- **Contributing Factors**: Highlights key factors that led to the final verdict

### 3. **Interactive Visualizations**

#### Charts Available:
1. **Confidence Breakdown**: Bar chart showing confidence factors
2. **Class Probabilities**: Distribution of ML classifier predictions
3. **Verifier Contributions**: Weighted contributions to consensus
4. **Agreement Visualization**: Pie chart showing ERROR vs VALID votes
5. **Reasoning Chain**: Textual step-by-step reasoning

### 4. **Feature Importance** (Advanced)

- Token-level attention analysis for transformer models
- Input importance scoring
- Highlighted important tokens in the input

## How to Use

### In Streamlit UI

1. Run a verification as usual
2. Scroll down to the **"🔍 Explainable AI - Why this verdict?"** expander
3. Explore the 4 tabs:
   - **Overview**: Quick summary
   - **Per-Verifier**: Detailed explanations for each model
   - **Consensus**: Weighted analysis
   - **Visualizations**: Interactive charts

### In Python Code

```python
from src.xai.explainer import XAIExplainer

explainer = XAIExplainer()

# Explain individual verifier
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
    verifier_results=all_results
)
```

## Example Explanation

For a step "5 + 3 = 9" (incorrect):

**Symbolic Verifier Explanation:**
- Verdict: ERROR
- Reasoning: "The symbolic verifier detected an arithmetic error by evaluating the mathematical expression."
- Evidence: "Calculation error: 5 + 3 = 8.0, but step claims 9.0"
- Key Factors: "Arithmetic calculation mismatch"
- Confidence Factors:
  - Expression validity: 0.95
  - Result match: 0.1
  - Calculation precision: 0.95

**Consensus Explanation:**
- Final Verdict: ERROR
- Agreement Type: MAJORITY
- Reasoning: "A majority of verifiers agreed on the verdict."
- Contributing Factors:
  - "symbolic (ERROR) had strong influence"
  - "Weighted error score (0.525) indicates an error was detected"

## Benefits

1. **Transparency**: See exactly why each model made its decision
2. **Trust**: Build confidence in the system's verdicts
3. **Debugging**: Identify which verifiers are contributing most
4. **Education**: Learn about mathematical reasoning verification
5. **Improvement**: Identify areas where models need better training

## Technical Details

- **No additional dependencies** for basic explanations
- **Plotly** for interactive visualizations (already in requirements)
- **Lightweight**: Explanations generated on-the-fly
- **Extensible**: Easy to add new explanation types

## Future Enhancements

- SHAP values for feature importance
- LIME explanations for local interpretability
- Counterfactual examples
- Attention heatmaps for transformer models
- Rule extraction from neural models

