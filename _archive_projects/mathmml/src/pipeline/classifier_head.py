"""Optional custom classifier head for error taxonomy."""

import torch
import torch.nn as nn
from typing import Dict, List


class ErrorTaxonomyHead(nn.Module):
    """Custom classifier head with hierarchical error taxonomy."""
    
    def __init__(self, hidden_size: int, num_base_classes: int = 10):
        """Initialize taxonomy head.
        
        Args:
            hidden_size: Input feature size
            num_base_classes: Number of base error classes
        """
        super().__init__()
        self.num_base_classes = num_base_classes
        
        # Base classification
        self.base_classifier = nn.Linear(hidden_size, num_base_classes)
        
        # Severity prediction (low, medium, high)
        self.severity_classifier = nn.Linear(hidden_size, 3)
        
        # Fixability prediction (auto-fixable, manual-review)
        self.fixability_classifier = nn.Linear(hidden_size, 2)
    
    def forward(self, hidden_states):
        """Forward pass.
        
        Args:
            hidden_states: Input features [batch_size, hidden_size]
            
        Returns:
            Dict with base_logits, severity_logits, fixability_logits
        """
        base_logits = self.base_classifier(hidden_states)
        severity_logits = self.severity_classifier(hidden_states)
        fixability_logits = self.fixability_classifier(hidden_states)
        
        return {
            'base_logits': base_logits,
            'severity_logits': severity_logits,
            'fixability_logits': fixability_logits
        }

