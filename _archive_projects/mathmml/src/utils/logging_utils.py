"""Logging utilities for session state and live logs."""

import time
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Log levels with emoji indicators."""
    PENDING = "⏳"
    SUCCESS = "✓"
    ERROR = "❌"
    WARNING = "⚠️"
    CALCULATION = "🔢"
    REASONING = "🧠"
    MODEL = "🤖"
    METRICS = "📊"


class SessionLogger:
    """Logger for session state and live logs."""
    
    def __init__(self):
        """Initialize logger."""
        self.logs: List[Dict] = []
        self.session_start = datetime.now()
        self.stats = {
            'total_steps': 0,
            'valid_steps': 0,
            'error_steps': 0,
            'verifications': 0
        }
    
    def log(
        self,
        message: str,
        level: LogLevel = LogLevel.PENDING,
        details: Optional[Dict] = None
    ):
        """Add log entry.
        
        Args:
            message: Log message
            level: Log level
            details: Additional details dict
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'message': message,
            'details': details or {}
        }
        self.logs.append(entry)
    
    def log_verification_start(self, step: str, problem: str):
        """Log start of verification.
        
        Args:
            step: Step being verified
            problem: Problem statement
        """
        self.log(
            f"Starting verification for step: {step[:50]}...",
            LogLevel.PENDING,
            {'step': step, 'problem': problem}
        )
        self.stats['verifications'] += 1
    
    def log_model_result(self, model_name: str, verdict: str, confidence: float):
        """Log model result.
        
        Args:
            model_name: Name of model
            verdict: VALID or ERROR
            confidence: Confidence score
        """
        self.log(
            f"{model_name}: {verdict} (confidence: {confidence:.3f})",
            LogLevel.MODEL,
            {'model': model_name, 'verdict': verdict, 'confidence': confidence}
        )
    
    def log_consensus(self, consensus: Dict):
        """Log consensus result.
        
        Args:
            consensus: Consensus result dict
        """
        verdict = consensus.get('final_verdict')
        confidence = consensus.get('overall_confidence')
        agreement = consensus.get('agreement_type')
        
        self.log(
            f"Consensus: {verdict} (confidence: {confidence:.3f}, agreement: {agreement})",
            LogLevel.SUCCESS if verdict == 'VALID' else LogLevel.ERROR,
            consensus
        )
        
        if verdict == 'VALID':
            self.stats['valid_steps'] += 1
        else:
            self.stats['error_steps'] += 1
        self.stats['total_steps'] += 1
    
    def get_recent_logs(self, n: int = 10) -> List[Dict]:
        """Get recent log entries.
        
        Args:
            n: Number of recent entries
            
        Returns:
            List of log dicts
        """
        return self.logs[-n:]
    
    def get_stats(self) -> Dict:
        """Get session statistics.
        
        Returns:
            Stats dict
        """
        return {
            **self.stats,
            'session_duration': str(datetime.now() - self.session_start),
            'total_logs': len(self.logs)
        }
    
    def clear(self):
        """Clear all logs."""
        self.logs = []
        self.stats = {
            'total_steps': 0,
            'valid_steps': 0,
            'error_steps': 0,
            'verifications': 0
        }
        self.session_start = datetime.now()

