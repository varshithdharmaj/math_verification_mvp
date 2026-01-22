"""
Build unified datasets from local sources.
Creates train/val/test JSON files using a unified schema.
"""
import argparse
import json
import os
import random
from typing import Dict, Any, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert legacy or custom dataset records into unified schema.
    Supported legacy fields:
    - type (text/image/handwritten)
    - input_text_or_path
    - ground_truth_answer
    """
    input_type = sample.get("input_type") or sample.get("type") or "text"
    input_type = input_type.lower()
    problem_id = sample.get("problem_id") or sample.get("id") or ""
    ground_truth = sample.get("ground_truth_answer") or sample.get("answer") or ""

    input_text = sample.get("input_text")
    image_path = sample.get("image_path")

    if not input_text and not image_path:
        legacy = sample.get("input_text_or_path", "")
        if input_type in ["image", "handwritten"]:
            image_path = legacy
        else:
            input_text = legacy

    unified = {
        "problem_id": problem_id,
        "input_type": "image" if input_type in ["image", "handwritten"] else "text",
        "input_text": input_text or "",
        "image_path": image_path or "",
        "ground_truth_answer": ground_truth,
        "split": ""
    }
    return unified


def split_dataset(items: List[Dict[str, Any]], train_ratio: float, val_ratio: float, seed: int):
    random.seed(seed)
    items = items[:]
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def assign_split(items: List[Dict[str, Any]], split_name: str):
    for item in items:
        item["split"] = split_name
    return items


def save_json(path: str, data: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="datasets/sample_data.json", help="Source dataset JSON")
    parser.add_argument("--image_manifest", type=str, default="", help="Optional image manifest JSON")
    parser.add_argument("--out_dir", type=str, default="datasets/unified", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_data = load_json(args.source)
    unified_items = [normalize_sample(s) for s in source_data]

    if args.image_manifest:
        image_data = load_json(args.image_manifest)
        unified_items.extend([normalize_sample(s) for s in image_data])

    train, val, test = split_dataset(unified_items, args.train_ratio, args.val_ratio, args.seed)
    assign_split(train, "train")
    assign_split(val, "val")
    assign_split(test, "test")

    save_json(os.path.join(args.out_dir, "train.json"), train)
    save_json(os.path.join(args.out_dir, "val.json"), val)
    save_json(os.path.join(args.out_dir, "test.json"), test)

    print(f"[OK] Wrote unified dataset to {args.out_dir}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")


if __name__ == "__main__":
    main()
