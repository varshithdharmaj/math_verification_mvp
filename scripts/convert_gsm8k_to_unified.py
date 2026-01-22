"""
Convert GSM8K JSON to unified dataset schema.
"""
import argparse
import json
import os
from typing import List, Dict


def load_gsm8k(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_answer(answer_text: str) -> str:
    """
    GSM8K answers end with '#### <final_answer>'.
    """
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


def extract_steps(answer_text: str) -> List[str]:
    """
    Convert GSM8K reasoning into step list.
    """
    parts = [p.strip() for p in answer_text.split("\n") if p.strip()]
    # Remove final answer line
    steps = [p for p in parts if not p.startswith("####")]
    return steps


def convert(gsm_data: List[Dict], split: str) -> List[Dict]:
    unified = []
    for i, row in enumerate(gsm_data):
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        unified.append({
            "problem_id": f"gsm8k_{split}_{i}",
            "input_type": "text",
            "input_text": question,
            "image_path": "",
            "ground_truth_answer": extract_answer(answer),
            "split": split,
            "steps": extract_steps(answer)
        })
    return unified


def save_json(path: str, data: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="GSM8K JSON input file")
    parser.add_argument("--split", type=str, default="test", help="Split name")
    parser.add_argument("--output", type=str, required=True, help="Output unified JSON file")
    args = parser.parse_args()

    gsm = load_gsm8k(args.input)
    unified = convert(gsm, args.split)
    save_json(args.output, unified)
    print(f"[OK] Wrote {len(unified)} samples to {args.output}")


if __name__ == "__main__":
    main()
