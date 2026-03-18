"""Data loaders for GSM8K and Math500 datasets with preprocessing."""

import re
import random
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

# Error taxonomy labels
ERROR_LABELS = [
    "correct",
    "arithmetic_error",
    "logical_error",
    "operation_mismatch",
    "conceptual_error",
    "notation_error",
    "sign_error",
    "unit_error",
    "order_ops_error",
    "semantic_error"
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(ERROR_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def parse_gsm8k_answer(answer: str) -> List[str]:
    """Extract steps from GSM8K answer format.
    
    Args:
        answer: Answer string with steps and calculator annotations
        
    Returns:
        List of step strings
    """
    # Remove calculator annotations like <<48/2=24>>
    answer = re.sub(r'<<[^>]+>>', '', answer)
    # Split by newlines and filter empty
    steps = [s.strip() for s in answer.split('\n') if s.strip()]
    # Remove final answer line (#### number)
    steps = [s for s in steps if not s.startswith('####')]
    return steps


def inject_arithmetic_error(step: str, prob: float = 0.3) -> Tuple[str, str]:
    """Inject arithmetic error by modifying numbers.
    
    Args:
        step: Original step text
        prob: Probability of injecting error
        
    Returns:
        (modified_step, error_label)
    """
    if random.random() > prob:
        return step, "correct"
    
    # Find numbers and operations
    numbers = re.findall(r'\d+\.?\d*', step)
    if len(numbers) < 2:
        return step, "correct"
    
    # Randomly modify a number
    num_idx = random.randint(0, len(numbers) - 1)
    original_num = numbers[num_idx]
    try:
        num_val = float(original_num)
        # Add/subtract small random amount
        error = random.choice([-1, 1]) * random.uniform(0.1, 0.5) * num_val
        new_val = num_val + error
        new_num = str(int(new_val)) if new_val.is_integer() else f"{new_val:.2f}"
        modified = step.replace(original_num, new_num, 1)
        return modified, "arithmetic_error"
    except:
        return step, "correct"


def inject_sign_error(step: str, prob: float = 0.2) -> Tuple[str, str]:
    """Inject sign error by flipping + to - or vice versa."""
    if random.random() > prob:
        return step, "correct"
    
    if '+' in step:
        modified = step.replace('+', '-', 1)
        return modified, "sign_error"
    elif '-' in step and not step.startswith('-'):
        modified = step.replace('-', '+', 1)
        return modified, "sign_error"
    return step, "correct"


def inject_operation_error(step: str, prob: float = 0.2) -> Tuple[str, str]:
    """Inject operation mismatch error."""
    if random.random() > prob:
        return step, "correct"
    
    ops = {'+': '-', '-': '+', '*': '/', '/': '*'}
    for op, replacement in ops.items():
        if op in step:
            modified = step.replace(op, replacement, 1)
            return modified, "operation_mismatch"
    return step, "correct"


def load_gsm8k(data_dir: str = ".", split: str = "train", config: str = "main") -> List[Dict]:
    """Load GSM8K dataset from parquet files.
    
    Args:
        data_dir: Directory containing main/ and socratic/ folders
        split: "train" or "test"
        config: "main" or "socratic"
        
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    data_path = Path(data_dir) / config / f"{split}-00000-of-00001.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"GSM8K data not found at {data_path}")
    
    df = pd.read_parquet(data_path)
    return df.to_dict('records')


def load_math500(csv_path: str = "math_500_test.csv") -> List[Dict]:
    """Load Math500 dataset from CSV.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of dicts with problem data
    """
    if not Path(csv_path).exists():
        print(f"Warning: Math500 file not found at {csv_path}, returning empty list")
        return []
    
    df = pd.read_csv(csv_path)
    # Assume columns: question, answer (or similar)
    records = df.to_dict('records')
    return records


def create_step_dataset(
    problems: List[Dict],
    inject_errors: bool = True,
    error_ratio: float = 0.5
) -> List[Dict]:
    """Create step-level dataset from problems.
    
    Args:
        problems: List of problem dicts with 'question' and 'answer'
        inject_errors: Whether to inject synthetic errors
        error_ratio: Ratio of error examples to correct examples
        
    Returns:
        List of step examples with:
        - problem: str
        - step_text: str
        - prev_steps_context: str (last 3 steps)
        - label: str (error type)
        - step_idx: int
    """
    dataset = []
    
    for problem in problems:
        question = problem.get('question', '')
        answer = problem.get('answer', '')
        
        steps = parse_gsm8k_answer(answer)
        
        for idx, step in enumerate(steps):
            # Get previous steps context (max 3)
            prev_steps = steps[max(0, idx-3):idx]
            prev_context = " ".join(prev_steps)
            
            # Create correct example
            dataset.append({
                'problem': question,
                'step_text': step,
                'prev_steps_context': prev_context,
                'label': 'correct',
                'step_idx': idx,
                'original': True
            })
            
            # Inject errors if enabled
            if inject_errors and random.random() < error_ratio:
                error_funcs = [
                    inject_arithmetic_error,
                    inject_sign_error,
                    inject_operation_error
                ]
                error_func = random.choice(error_funcs)
                modified_step, error_label = error_func(step)
                
                if error_label != "correct":
                    dataset.append({
                        'problem': question,
                        'step_text': modified_step,
                        'prev_steps_context': prev_context,
                        'label': error_label,
                        'step_idx': idx,
                        'original': False
                    })
    
    return dataset


def prepare_training_data(
    gsm8k_dir: str = ".",
    math500_path: str = "math_500_test.csv",
    train_split: str = "train",
    test_split: str = "test",
    output_dir: str = "data/processed",
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Prepare train/val/test splits for classifier training.
    
    Args:
        gsm8k_dir: Directory with GSM8K data
        math500_path: Path to Math500 CSV
        train_split: GSM8K train split name
        test_split: GSM8K test split name
        output_dir: Output directory for processed data
        seed: Random seed
        
    Returns:
        (train_data, val_data, test_data)
    """
    random.seed(seed)
    
    # Load datasets
    gsm8k_train = load_gsm8k(gsm8k_dir, train_split, "main")
    gsm8k_test = load_gsm8k(gsm8k_dir, test_split, "main")
    math500 = load_math500(math500_path)
    
    # Create step datasets
    train_steps = create_step_dataset(gsm8k_train, inject_errors=True, error_ratio=0.5)
    test_steps = create_step_dataset(gsm8k_test, inject_errors=True, error_ratio=0.5)
    
    if math500:
        math500_steps = create_step_dataset(math500, inject_errors=True, error_ratio=0.5)
        train_steps.extend(math500_steps)
    
    # Split train into train/val (80/20)
    random.shuffle(train_steps)
    val_size = int(len(train_steps) * 0.2)
    val_data = train_steps[:val_size]
    train_data = train_steps[val_size:]
    
    # Save processed data
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_path = Path(output_dir) / f"{split_name}.json"
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {len(split_data)} examples to {output_path}")
    
    return train_data, val_data, test_steps


if __name__ == "__main__":
    # Quick test
    train, val, test = prepare_training_data()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

