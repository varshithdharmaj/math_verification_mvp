"""
Evaluate restricted image segment pipeline.
Computes OCR accuracy vs ground truth LaTeX and reasoning metrics.
"""
import argparse
import csv
import json
import os
import time
from difflib import SequenceMatcher
from typing import List, Dict, Any

from backend.core.orchestrator import MathVerificationOrchestrator


MODES = [
    "single_llm_only",
    "llm_plus_sympy",
    "multi_agent_no_classifier",
    "multi_agent_no_ocr_conf",
    "multi_agent_with_classifier",
    "full_mvm2"
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def normalize_text(s: str) -> str:
    return str(s or "").strip().lower().replace(" ", "")


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def eval_dataset(dataset_path: str, output_csv: str, limit: int = 0):
    data = load_jsonl(dataset_path)
    if limit > 0:
        data = data[:limit]

    orchestrator = MathVerificationOrchestrator()
    rows = []

    for idx, sample in enumerate(data):
        image_path = sample.get("image_path", "")
        latex_gt = sample.get("latex_expression", "")
        gt_answer = sample.get("ground_truth_answer", "")
        problem_id = sample.get("problem_id", f"img_{idx}")

        if not os.path.exists(image_path):
            continue

        for mode in MODES:
            start = time.time()
            result = orchestrator.verify_from_image(image_path, mode=mode)
            latency = (time.time() - start) * 1000

            predicted_answer = result.get("final_answer", "")
            is_correct = normalize_text(predicted_answer) == normalize_text(gt_answer)

            ocr_text = result.get("ocr_normalized_text", "")
            ocr_acc = similarity(normalize_text(ocr_text), normalize_text(latex_gt))

            consensus_stats = result.get("consensus_stats", {})
            avg_cons = consensus_stats.get("avg_consensus", 0.0)
            hall_rate = consensus_stats.get("hallucination_rate", 0.0)

            rows.append({
                "problem_id": problem_id,
                "mode": mode,
                "latency_ms": round(latency, 2),
                "ocr_accuracy": round(ocr_acc, 4),
                "ocr_confidence": round(result.get("ocr_confidence", 0.0), 4),
                "answer_accuracy": is_correct,
                "reasoning_validity": result.get("final_verdict", ""),
                "hallucination_rate": round(hall_rate, 4),
                "avg_consensus": round(avg_cons, 4)
            })

    # Save CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="JSONL dataset path")
    parser.add_argument("--output", type=str, required=True, help="CSV output")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit")
    args = parser.parse_args()

    eval_dataset(args.dataset, args.output, args.limit)


if __name__ == "__main__":
    main()
