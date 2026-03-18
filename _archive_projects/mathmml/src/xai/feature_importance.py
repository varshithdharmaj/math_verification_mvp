"""Feature importance analysis for ML models."""

import numpy as np
from typing import Dict, List, Tuple
import torch


def analyze_input_importance(
    step: str,
    problem: str,
    prev_steps: str,
    tokenizer,
    model,
    device: str = "cpu"
) -> Dict:
    """Analyze which parts of the input are most important for the prediction.
    
    Args:
        step: Current step text
        problem: Problem statement
        prev_steps: Previous steps context
        tokenizer: Tokenizer instance
        model: Model instance
        device: Device to use
        
    Returns:
        Dict with token importance scores
    """
    # Format input
    input_text = f"{problem} [SEP] {prev_steps} [SEP] {step}"
    
    # Tokenize
    encoded = tokenizer(
        input_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Get attention weights if available
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Try to extract attention weights
        if hasattr(outputs, 'attentions') and outputs.attentions:
            # Use attention weights from last layer
            attention = outputs.attentions[-1]  # [batch, heads, seq, seq]
            # Average across heads
            attention_avg = attention.mean(dim=1).squeeze(0)  # [seq, seq]
            # Get attention to [CLS] token (first token)
            cls_attention = attention_avg[0, :].cpu().numpy()
        else:
            # Fallback: use uniform attention
            cls_attention = np.ones(input_ids.shape[1]) / input_ids.shape[1]
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    # Map attention to tokens
    token_importance = {}
    for i, token in enumerate(tokens):
        if token not in ['[PAD]', '[CLS]', '[SEP]']:
            token_importance[token] = float(cls_attention[i])
    
    return {
        'tokens': tokens,
        'importance_scores': cls_attention.tolist(),
        'token_importance': token_importance,
        'top_important_tokens': sorted(
            token_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    }


def highlight_important_tokens(
    step: str,
    importance_analysis: Dict,
    top_k: int = 5
) -> str:
    """Create HTML with highlighted important tokens.
    
    Args:
        step: Step text
        importance_analysis: Output from analyze_input_importance
        top_k: Number of top tokens to highlight
        
    Returns:
        HTML string with highlighted tokens
    """
    top_tokens = importance_analysis.get('top_important_tokens', [])[:top_k]
    important_tokens = {token: score for token, score in top_tokens}
    
    # Simple highlighting - in practice, would need better token matching
    highlighted = step
    for token, score in important_tokens.items():
        if token in highlighted:
            highlighted = highlighted.replace(
                token,
                f'<mark style="background-color: yellow; opacity: {min(score * 2, 1.0)}">{token}</mark>'
            )
    
    return highlighted

