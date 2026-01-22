"""
Evaluate the reasoning classifier on evaluation CSV.
Reports accuracy, ROC-AUC, and calibration (Brier score).
"""
import argparse
import csv
from typing import List, Dict

import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from backend.models.reasoning_classifier import build_feature_vector, default_model_path


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_bool(value: str) -> int:
    if isinstance(value, bool):
        return int(value)
    v = str(value).strip().lower()
    return 1 if v in ["1", "true", "yes"] else 0


def build_features_from_row(row: Dict[str, str]) -> List[float]:
    return build_feature_vector(
        symbolic_score=float(row.get("symbolic_score", 0.0)),
        logical_score=float(row.get("logical_score", 0.0)),
        avg_consensus=float(row.get("avg_consensus", 0.0)),
        hallucination_rate=float(row.get("hallucination_rate", 0.0)),
        num_flagged_steps=int(float(row.get("num_flagged_steps", 0))),
        ocr_confidence=float(row.get("ocr_confidence", 1.0)),
        steps_count=int(float(row.get("steps_count", 0))),
        reasoning_length=int(float(row.get("reasoning_length", 0))),
        consensus_total_steps=int(float(row.get("consensus_total_steps", 0)))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="evaluation_results.csv", help="Evaluation CSV")
    parser.add_argument("--model", type=str, default="", help="Model path")
    parser.add_argument("--split", type=str, default="test", help="Split name to evaluate")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    rows = [r for r in rows if r.get("split") == args.split]

    if not rows:
        raise ValueError("No rows found for requested split.")

    X = [build_features_from_row(r) for r in rows]
    y = [parse_bool(r.get("is_correct", 0)) for r in rows]

    model_path = args.model or default_model_path()
    model = joblib.load(model_path)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_proba)
    except Exception:
        auc = 0.0
    brier = brier_score_loss(y, y_proba)

    print(f"[RESULT] Split: {args.split}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Brier: {brier:.4f}")


if __name__ == "__main__":
    main()
