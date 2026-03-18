"""MLStepClassifier using HuggingFace Transformers."""

import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer
)
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path


class MLStepClassifier(nn.Module):
    """Transformer-based step classifier for error detection."""
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 10,
        dropout: float = 0.1
    ):
        """Initialize classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of error classes
            dropout: Dropout rate
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            labels: Ground truth labels
            
        Returns:
            Loss and logits
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits}


class MLStepClassifierWrapper:
    """Wrapper for MLStepClassifier with inference utilities."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "roberta-base",
        num_labels: int = 10,
        device: str = "cpu"
    ):
        """Initialize wrapper.
        
        Args:
            model_path: Path to trained model checkpoint
            model_name: Base model name
            num_labels: Number of classes
            device: Device to use
        """
        self.device = torch.device(device)
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if model_path and Path(model_path).exists():
            self.model = MLStepClassifier(model_name, num_labels)
            state_dict = torch.load(Path(model_path) / "pytorch_model.bin", map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            # Initialize untrained model
            self.model = MLStepClassifier(model_name, num_labels)
            print(f"Initialized untrained model: {model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        try:
            from src.data.loaders import ID_TO_LABEL
            self.id_to_label = ID_TO_LABEL
        except ImportError:
            # Fallback if import fails
            self.id_to_label = {i: f"label_{i}" for i in range(num_labels)}
    
    def format_input(self, problem: str, prev_steps: str, current_step: str) -> str:
        """Format input for model.
        
        Args:
            problem: Problem statement
            prev_steps: Previous steps context
            current_step: Current step text
            
        Returns:
            Formatted input string
        """
        # Limit prev_steps to last 3
        prev_parts = prev_steps.split() if prev_steps else []
        prev_context = " ".join(prev_parts[-50:])  # Limit tokens
        
        input_text = f"{problem} [SEP] {prev_context} [SEP] {current_step}"
        return input_text
    
    def infer(self, problem: str, prev_steps: str, current_step: str) -> Dict:
        """Run inference on a step.
        
        Args:
            problem: Problem statement
            prev_steps: Previous steps context
            current_step: Current step text
            
        Returns:
            Dict with 'label', 'prob_vector', 'confidence'
        """
        # Format input
        input_text = self.format_input(problem, prev_steps, current_step)
        
        # Tokenize
        encoded = self.tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            probs_np = probs.cpu().numpy()[0]
        
        # Get predicted label
        pred_id = np.argmax(probs_np)
        pred_label = self.id_to_label.get(pred_id, "correct")
        
        # Confidence
        if pred_label == "correct":
            confidence = float(probs_np[pred_id])
        else:
            # Confidence is max of error class probabilities
            error_probs = [probs_np[i] for i, label in self.id_to_label.items() if label != "correct"]
            confidence = float(max(error_probs)) if error_probs else 0.5
        
        return {
            'label': pred_label,
            'prob_vector': probs_np.tolist(),
            'confidence': confidence,
            'all_probs': {self.id_to_label[i]: float(probs_np[i]) for i in range(self.num_labels)}
        }
    
    def verify(self, step: str, problem: str = "", prev_steps: List[str] = None) -> Dict:
        """Main verification method.
        
        Args:
            step: Step text to verify
            problem: Problem statement
            prev_steps: Previous steps
            
        Returns:
            Verification result dict
        """
        if prev_steps is None:
            prev_steps = []
        
        prev_context = " ".join(prev_steps[-3:])
        
        result = self.infer(problem, prev_context, step)
        
        # Map to verdict
        if result['label'] == 'correct':
            verdict = 'VALID'
            confidence = result['confidence']
        else:
            verdict = 'ERROR'
            confidence = result['confidence']
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'error_type': result['label'] if result['label'] != 'correct' else None,
            'details': f"ML classifier: {result['label']} (confidence: {confidence:.3f})",
            'prob_vector': result['prob_vector'],
            'all_probs': result['all_probs']
        }
    
    def save(self, save_path: str):
        """Save model and tokenizer.
        
        Args:
            save_path: Directory to save to
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), save_dir / "pytorch_model.bin")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # Save config
        self.model.config.save_pretrained(save_dir)
        
        print(f"Model saved to {save_dir}")

