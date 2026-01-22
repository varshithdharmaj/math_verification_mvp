"""
Prepare a restricted handwritten math dataset for algebra/calculus segment.
Input: JSON/CSV with image_path and latex_expression.
Output: JSONL/CSV with sympy expression and ground truth.
"""
import argparse
import csv
import json
import os
import re
from typing import List, Dict, Any

import sympy as sp

try:
    from sympy.parsing.latex import parse_latex
    LATEX_PARSER_AVAILABLE = True
except Exception:
    LATEX_PARSER_AVAILABLE = False


def load_rows(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".json") or path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text.startswith("["):
                items = json.loads(text)
            else:
                for line in text.splitlines():
                    if line.strip():
                        items.append(json.loads(line))
        return items
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    raise ValueError("Unsupported input format")


def normalize_latex(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("\\cdot", "*").replace("\\times", "*")
    expr = expr.replace("\\div", "/")
    return expr


def is_in_scope(expr: str) -> bool:
    if not expr:
        return False
    if len(expr) > 120:
        return False
    if re.search(r'\\(sum|prod|matrix|det|cases)', expr):
        return False
    return True


def parse_latex_to_sympy(expr: str):
    expr = normalize_latex(expr)
    if LATEX_PARSER_AVAILABLE:
        try:
            return parse_latex(expr)
        except Exception:
            pass
    try:
        return sp.sympify(expr)
    except Exception:
        return None


def compute_ground_truth(sym_expr):
    x = sp.symbols("x")
    try:
        # If equation, solve for x
        if isinstance(sym_expr, sp.Equality):
            sol = sp.solve(sym_expr, x)
            return sol
        # If integral, evaluate if definite
        if sym_expr.has(sp.Integral):
            return sp.simplify(sym_expr.doit())
        # If limit, evaluate
        if sym_expr.has(sp.Limit):
            return sp.simplify(sym_expr.doit())
        # Otherwise simplify
        return sp.simplify(sym_expr)
    except Exception:
        return None


def difficulty(expr: str) -> str:
    ops = len(re.findall(r'[\+\-\*/\^=]', expr))
    if len(expr) < 30 and ops <= 3:
        return "simple"
    if len(expr) < 70 and ops <= 7:
        return "medium"
    return "hard"


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to dataset manifest (json/jsonl/csv)")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--output_csv", type=str, default="", help="Optional CSV output")
    parser.add_argument("--limit", type=int, default=1000, help="Max samples to keep")
    args = parser.parse_args()

    raw = load_rows(args.input)
    filtered = []

    for idx, row in enumerate(raw):
        latex = row.get("latex_expression") or row.get("latex") or row.get("expression") or ""
        image_path = row.get("image_path") or row.get("path") or row.get("image") or ""
        if not latex or not image_path:
            continue
        if not is_in_scope(latex):
            continue
        sym_expr = parse_latex_to_sympy(latex)
        if sym_expr is None:
            continue
        gt = compute_ground_truth(sym_expr)
        filtered.append({
            "problem_id": f"img_{idx}",
            "image_path": image_path,
            "latex_expression": latex,
            "sympy_expression": str(sym_expr),
            "ground_truth_answer": str(gt) if gt is not None else "",
            "difficulty": difficulty(latex)
        })
        if len(filtered) >= args.limit:
            break

    write_jsonl(args.output_jsonl, filtered)
    if args.output_csv:
        write_csv(args.output_csv, filtered)

    print(f"[OK] Wrote {len(filtered)} samples")


if __name__ == "__main__":
    main()
