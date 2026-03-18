"""LLMLogicalChecker using heuristics and optional LLM API."""

import re
import os
from typing import Dict, List, Optional
import random
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm_providers import LLMProvider


class LLMLogicalChecker:
    """Checks logical consistency using heuristics and optional LLM API."""
    
    def __init__(
        self,
        use_api: bool = False,
        api_provider: str = "openai",
        model: str = None,
        confidence_base: float = 0.82
    ):
        """Initialize checker.
        
        Args:
            use_api: Whether to use LLM API (requires API keys)
            api_provider: "openai", "gemini", "llama", "anthropic"
            model: Specific model name (e.g., "gpt-4", "gemini-pro", "llama2")
            confidence_base: Base confidence level
        """
        self.use_api = use_api
        self.api_provider = api_provider
        self.confidence_base = confidence_base
        self.llm_provider = None
        
        if use_api:
            try:
                self.llm_provider = LLMProvider(api_provider, model)
                if not self.llm_provider.client:
                    print(f"Warning: {api_provider} client not available, using mock mode")
                    self.use_api = False
            except Exception as e:
                print(f"Warning: Failed to initialize {api_provider}: {e}, using mock mode")
                self.use_api = False
    
    def detect_contradiction(self, step: str, prev_steps: List[str]) -> bool:
        """Detect logical contradictions.
        
        Args:
            step: Current step
            prev_steps: Previous steps
            
        Returns:
            True if contradiction detected
        """
        # Check for contradiction patterns
        contradiction_patterns = [
            (r'\b(but|however|yet)\b.*\b(but|however|yet)\b', 'double_contradiction'),
            (r'\b(and|also)\b.*\b(but|however)\b', 'contradiction'),
            (r'\bnot\b.*\b(but|however)\b', 'negation_contradiction')
        ]
        
        full_text = " ".join(prev_steps + [step]).lower()
        for pattern, _ in contradiction_patterns:
            if re.search(pattern, full_text):
                return True
        
        return False
    
    def detect_operation_mismatch(self, step: str, problem: str) -> bool:
        """Detect operation mismatch with problem requirements.
        
        Args:
            step: Current step
            problem: Problem statement
            
        Returns:
            True if mismatch detected
        """
        # Extract operations from problem
        problem_ops = set(re.findall(r'\b(add|subtract|multiply|divide|sum|difference|product|quotient)\b', problem.lower()))
        
        # Extract operations from step
        step_ops = set(re.findall(r'[+\-*/]', step))
        
        # Simple heuristic: if problem mentions "add" but step uses subtraction
        if 'add' in problem.lower() or 'sum' in problem.lower():
            if '-' in step_ops and '+' not in step_ops:
                return True
        
        if 'subtract' in problem.lower() or 'difference' in problem.lower():
            if '+' in step_ops and '-' not in step_ops:
                return True
        
        return False
    
    def detect_circular_reasoning(self, step: str, prev_steps: List[str]) -> bool:
        """Detect circular reasoning patterns.
        
        Args:
            step: Current step
            prev_steps: Previous steps
            
        Returns:
            True if circular reasoning detected
        """
        # Check if step repeats a previous conclusion
        step_conclusion = re.search(r'=\s*([\d.]+)', step)
        if step_conclusion:
            conclusion = step_conclusion.group(1)
            for prev in prev_steps:
                if conclusion in prev and '=' in prev:
                    # Check if it's using its own result
                    if step != prev:
                        return True
        
        return False
    
    def detect_semantic_inconsistency(self, step: str, problem: str) -> bool:
        """Detect semantic inconsistencies.
        
        Args:
            step: Current step
            problem: Problem statement
            
        Returns:
            True if inconsistency detected
        """
        # Extract units from problem
        problem_units = set(re.findall(r'\b(dollars?|cents?|miles?|hours?|minutes?|pounds?|kg|grams?)\b', problem.lower()))
        
        # Extract units from step
        step_units = set(re.findall(r'\b(dollars?|cents?|miles?|hours?|minutes?|pounds?|kg|grams?)\b', step.lower()))
        
        # If problem has units but step doesn't match
        if problem_units and step_units:
            if not problem_units.intersection(step_units):
                return True
        
        return False
    
    def check_with_llm(self, step: str, problem: str, prev_steps: List[str]) -> Optional[Dict]:
        """Check using LLM API (if enabled).
        
        Args:
            step: Current step
            problem: Problem statement
            prev_steps: Previous steps
            
        Returns:
            LLM result dict or None
        """
        if not self.use_api or not self.llm_provider:
            return None
        
        try:
            result = self.llm_provider.check_step(step, problem, prev_steps)
            return result
        except Exception as e:
            print(f"LLM API error: {e}")
            return None
    
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
        
        errors = []
        error_types = []
        
        # Run heuristic checks
        if self.detect_contradiction(step, prev_steps):
            errors.append("Contradiction detected")
            error_types.append("logical_error")
        
        if self.detect_operation_mismatch(step, problem):
            errors.append("Operation mismatch with problem")
            error_types.append("operation_mismatch")
        
        if self.detect_circular_reasoning(step, prev_steps):
            errors.append("Circular reasoning detected")
            error_types.append("logical_error")
        
        if self.detect_semantic_inconsistency(step, problem):
            errors.append("Semantic inconsistency (units)")
            error_types.append("unit_error")
        
        # Try LLM check if enabled
        llm_result = self.check_with_llm(step, problem, prev_steps)
        if llm_result and llm_result.get('has_error'):
            errors.append(llm_result.get('reason', 'LLM detected error'))
            error_types.append(llm_result.get('error_type', 'logical_error'))
        
        if errors:
            return {
                'verdict': 'ERROR',
                'confidence': self.confidence_base + 0.05,
                'error_type': error_types[0] if error_types else 'logical_error',
                'details': "; ".join(errors)
            }
        else:
            return {
                'verdict': 'VALID',
                'confidence': self.confidence_base,
                'error_type': None,
                'details': 'No logical errors detected'
            }

