# MultiMath-System: A Multi-Modal, Adaptive Ensemble Framework for Mathematical Reasoning

## System Architecture Overview

MultiMath-System is a novel neuro-symbolic and multi-modal integration pipeline designed to push the boundaries of AI mathematical reasoning. By combining state-of-the-art specialized datasets with an adaptive training curriculum, the system tackles textual, visual, and handwritten mathematical problems.

### Directory Structure

```text
MultiMath-System/
├── input_module/          # Handles text (Math-Verify), vision (MathVision), and OCR (Math_Handwriting_OCR)
├── reasoning_module/      # Ensemble Multi-LLM routing, SymPy symbolic evaluation, and Consensus Engine
├── evaluation_module/     # Adaptive evaluation metrics, ablation framework, and divergence matrix calculation
├── datasets/              # Cloned submodules containing MATH-V, MathVerse, MathVision, and OpenMathReasoning
├── experiments/           # Training scripts, metadata mapping, and result visualizations
└── docs/                  # Paper drafts and extensive documentation 
```

### Integrated Resources (The "HOW")
Our foundation brings together cutting-edge community repositories:
1. **Math-Verify (Hugging Face)**: Used strictly for stringent answer correctness parsing and evaluation logic.
2. **MATH-V (mathllm)**: Benchmarking and staged curriculum dataset (Levels 1 through 5).
3. **MathVerse (ZrrSkywalker)**: Crucial for visual mathematical understanding and spatial reasoning.
4. **Handwriting OCR Pipelines (yixchen, johnkimdw)**: Translates raw noisy handwriting into formal mathematical syntax (LaTeX representation) before reaching the models.
5. **MathVision & OpenMathReasoning**: Mixed-modality training pools sampled adaptively across our training phases.

### Novel Contributions (The "WHY")
Instead of simply throwing all data at an LLM, MultiMath-System proposes:
1. **Adaptive Training Curriculum**: Staging complexity logically (Arithmetic -> Algebra -> Advanced Math -> Visual Math) forcing the model to acquire robust symbolic rules before tackling vision.
2. **Cross-modality Error Interception**: Identifying whether a failure occurred in translation (OCR), logical syntax (SymPy), or spatial reasoning (MathVerse metrics).
3. **Inter-Agent Divergence Matrices**: A novel metric for multi-agent systems to mathematically quantify *hallucination risk* by measuring the delta between reasoning traces.
