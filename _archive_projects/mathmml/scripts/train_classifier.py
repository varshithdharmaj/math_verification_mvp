"""CLI script for training ML classifier."""

import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ml_step_classifier import MLStepClassifier
from src.data.loaders import LABEL_TO_ID, ID_TO_LABEL


class StepDataset(Dataset):
    """Dataset for step classification."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """Initialize dataset.
        
        Args:
            data_path: Path to JSON data file
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input
        problem = item['problem']
        prev_context = item.get('prev_steps_context', '')
        step = item['step_text']
        
        input_text = f"{problem} [SEP] {prev_context} [SEP] {step}"
        
        # Tokenize
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label = LABEL_TO_ID.get(item['label'], 0)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dict of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description="Train ML step classifier")
    parser.add_argument("--train_data", type=str, default="data/processed/train.json",
                       help="Path to training data")
    parser.add_argument("--val_data", type=str, default="data/processed/val.json",
                       help="Path to validation data")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                       help="Base model name")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset = StepDataset(args.train_data, tokenizer)
    val_dataset = StepDataset(args.val_data, tokenizer)
    
    # Initialize model
    num_labels = len(LABEL_TO_ID)
    model = MLStepClassifier(args.model_name, num_labels)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        fp16=torch.cuda.is_available()
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

