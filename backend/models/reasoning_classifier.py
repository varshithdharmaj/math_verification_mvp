"""
Reasoning classifier training and inference utilities.
Trains a small classifier that predicts correctness from pipeline features.
"""
import os
from typing import List

import joblib


FEATURE_NAMES = [
    "symbolic_score",
    "logical_score",
    "avg_consensus",
    "hallucination_rate",
    "num_flagged_steps",
    "ocr_confidence",
    "steps_count",
    "reasoning_length",
    "consensus_total_steps"
]


def build_feature_vector(
    symbolic_score: float,
    logical_score: float,
    avg_consensus: float,
    hallucination_rate: float,
    num_flagged_steps: int,
    ocr_confidence: float,
    steps_count: int,
    reasoning_length: int,
    consensus_total_steps: int
) -> List[float]:
    return [
        float(symbolic_score),
        float(logical_score),
        float(avg_consensus),
        float(hallucination_rate),
        float(num_flagged_steps),
        float(ocr_confidence),
        float(steps_count),
        float(reasoning_length),
        float(consensus_total_steps)
    ]


def default_model_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "checkpoints",
        "reasoning_classifier.joblib"
    )


def load_model_or_none(model_path: str = None):
    path = model_path or default_model_path()
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None
