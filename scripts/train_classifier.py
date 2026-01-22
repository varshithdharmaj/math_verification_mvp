"""
Train a small reasoning classifier from evaluation CSV.
Saves the model to backend/models/checkpoints/reasoning_classifier.joblib
"""
import argparse
import csv
import os
from typing import List, Dict

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from backend.models.reasoning_classifier import FEATURE_NAMES, build_feature_vector, default_model_path


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
    parser.add_argument("--model_out", type=str, default="", help="Output model path")
    parser.add_argument("--split_train", type=str, default="train", help="Train split name")
    parser.add_argument("--split_val", type=str, default="val", help="Val split name")
    args = parser.parse_args()

    rows = load_csv(args.csv)

    train_rows = [r for r in rows if r.get("split") == args.split_train]
    val_rows = [r for r in rows if r.get("split") == args.split_val]

    if not train_rows:
        raise ValueError("No training rows found. Check split names or CSV content.")

    X_train = [build_features_from_row(r) for r in train_rows]
    y_train = [parse_bool(r.get("is_correct", 0)) for r in train_rows]

    X_val = [build_features_from_row(r) for r in val_rows] if val_rows else []
    y_val = [parse_bool(r.get("is_correct", 0)) for r in val_rows] if val_rows else []

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    model_path = args.model_out or default_model_path()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)

    print(f"[OK] Saved model to {model_path}")

    # Optional validation
    if X_val:
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, y_pred)
        try:
            auc = roc_auc_score(y_val, y_proba)
        except Exception:
            auc = 0.0
        print(f"[VAL] Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
    else:
        print("[WARN] No validation rows found. Skipping validation.")


if __name__ == "__main__":
    main()
