"""
Offline evaluation utilities for the MVM² verification pipeline using Hugging Face `evaluate`.

This module is intentionally decoupled from Gradio so it can be run:
- Locally (e.g. `python evaluation_module.py`)
- Inside a Hugging Face Space (from a separate evaluation tab or button)

The core idea:
- You supply a list of ground-truth labels (e.g. correct final numeric answers).
- You supply a list of model predictions produced by the MVM² consensus engine.
- We compute accuracy, F1, precision, recall, and confusion matrix.
- We emit both a structured JSON-friendly dict and an optional Markdown report string.
"""

from __future__ import annotations

import json
from typing import List, Any, Dict, Tuple

import evaluate


def compute_classification_metrics(
    y_true: List[Any],
    y_pred: List[Any],
) -> Dict[str, Any]:
    """
    Compute core classification metrics for scalar labels using Hugging Face `evaluate`.

    Inputs:
    - y_true: list of ground-truth labels (ints, strings, etc.).
    - y_pred: list of predicted labels from your MVM² pipeline.

    Returns:
    - metrics: dict containing accuracy, F1, precision, recall, errors, and confusion matrix.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length ({len(y_true)}) must match y_pred length ({len(y_pred)})."
        )

    # Load evaluation metrics from Hugging Face.
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    confusion_metric = evaluate.load("confusion_matrix")

    # NOTE: For non-binary labels, evaluate will infer the label space from the data.
    acc_result = accuracy_metric.compute(references=y_true, predictions=y_pred)
    f1_result = f1_metric.compute(references=y_true, predictions=y_pred, average="macro")
    precision_result = precision_metric.compute(
        references=y_true, predictions=y_pred, average="macro"
    )
    recall_result = recall_metric.compute(
        references=y_true, predictions=y_pred, average="macro"
    )
    confusion_result = confusion_metric.compute(
        references=y_true,
        predictions=y_pred,
    )

    total_samples = len(y_true)
    errors = int(sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp))

    metrics = {
        "accuracy": acc_result.get("accuracy", 0.0),
        "f1_macro": f1_result.get("f1", 0.0),
        "precision_macro": precision_result.get("precision", 0.0),
        "recall_macro": recall_result.get("recall", 0.0),
        "total_samples": total_samples,
        "errors": errors,
        "confusion_matrix": {
            "labels": confusion_result.get("labels", []),
            "matrix": confusion_result.get("confusion_matrix", []),
        },
    }
    return metrics


def build_markdown_report(metrics: Dict[str, Any]) -> str:
    """
    Build a Markdown string summarizing the evaluation results.

    IMPORTANT:
    - This returns a string only. You decide whether to write it to disk
      or render it in the UI. This respects the project's preference to
      avoid static .md guide files in the repo.
    """
    cm = metrics.get("confusion_matrix", {})
    labels = cm.get("labels", [])
    matrix = cm.get("matrix", [])

    # Render confusion matrix as a Markdown table if labels are available.
    cm_lines: List[str] = []
    if labels and matrix:
        header = "| True \\ Pred | " + " | ".join(str(l) for l in labels) + " |"
        sep = "|" + "|".join([" --- " for _ in range(len(labels) + 1)]) + "|"
        cm_lines.append(header)
        cm_lines.append(sep)
        for true_label, row in zip(labels, matrix):
            row_str = " | ".join(str(v) for v in row)
            cm_lines.append(f"| {true_label} | {row_str} |")

    md_lines = [
        "# MVM² Evaluation Report",
        "",
        "## Summary Metrics",
        f"- **Total Samples**: `{metrics.get('total_samples', 0)}`",
        f"- **Errors**: `{metrics.get('errors', 0)}`",
        f"- **Accuracy**: `{metrics.get('accuracy', 0.0):.4f}`",
        f"- **Macro F1**: `{metrics.get('f1_macro', 0.0):.4f}`",
        f"- **Macro Precision**: `{metrics.get('precision_macro', 0.0):.4f}`",
        f"- **Macro Recall**: `{metrics.get('recall_macro', 0.0):.4f}`",
    ]

    if cm_lines:
        md_lines.append("")
        md_lines.append("## Confusion Matrix")
        md_lines.extend(cm_lines)

    return "\n".join(md_lines)


def evaluate_from_labels(
    y_true: List[Any],
    y_pred: List[Any],
    save_json_path: str | None = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Convenience wrapper:
    - Computes metrics from given labels.
    - Optionally saves the JSON report to disk.
    - Returns both the metrics dict and a Markdown report string.

    Parameters:
    - y_true: ground-truth labels.
    - y_pred: model predictions.
    - save_json_path: if not None, metrics are written to this JSON file.
    """
    metrics = compute_classification_metrics(y_true, y_pred)
    md_report = build_markdown_report(metrics)

    if save_json_path is not None:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

    return metrics, md_report


if __name__ == "__main__":
    """
    Minimal CLI-style sanity check:
    - Replace `y_true` and `y_pred` with your own benchmark labels.
    - Run: `python evaluation_module.py`
    """
    example_y_true = [1, 0, 1, 1, 0]
    example_y_pred = [1, 0, 0, 1, 0]

    metrics_out, md = evaluate_from_labels(
        example_y_true,
        example_y_pred,
        save_json_path="evaluation_report.json",
    )

    print("Structured metrics:")
    print(json.dumps(metrics_out, indent=2))
    print("\nMarkdown summary:")
    print(md)

