"""EnsembleNeuralChecker using multi-LLM voting."""

import random
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm_providers import LLMProvider, get_available_providers


class EnsembleNeuralChecker:
    """Ensemble checker using multiple LLM votes."""
    
    def __init__(
        self,
        num_models: int = 3,
        use_apis: bool = False,
        model_configs: List[Dict] = None,
        confidence_base: float = 0.75
    ):
        """Initialize ensemble checker.
        
        Args:
            num_models: Number of LLM models to simulate/query
            use_apis: Whether to use real APIs (otherwise mock)
            model_configs: List of dicts with 'provider' and 'model' keys
                          e.g., [{'provider': 'openai', 'model': 'gpt-4'}, 
                                 {'provider': 'gemini', 'model': 'gemini-pro'}]
            confidence_base: Base confidence level
        """
        self.num_models = num_models
        self.use_apis = use_apis
        self.confidence_base = confidence_base
        self.llm_providers = []
        
        if use_apis and model_configs:
            for config in model_configs[:num_models]:
                try:
                    provider = LLMProvider(
                        provider=config.get('provider', 'openai'),
                        model=config.get('model')
                    )
                    if provider.client:
                        self.llm_providers.append(provider)
                except Exception as e:
                    print(f"Warning: Failed to initialize {config}: {e}")
        
        # If no providers initialized, use available ones
        if use_apis and not self.llm_providers:
            available = get_available_providers()
            for provider_name in available[:num_models]:
                try:
                    provider = LLMProvider(provider=provider_name)
                    if provider.client:
                        self.llm_providers.append(provider)
                except:
                    pass
    
    def mock_llm_check(self, step: str, problem: str, prev_steps: List[str]) -> Dict:
        """Mock LLM check (for testing without APIs).
        
        Args:
            step: Step text
            problem: Problem statement
            prev_steps: Previous steps
            
        Returns:
            Mock result dict
        """
        # Simple heuristics for mock
        has_error = False
        error_type = None
        
        # Check for obvious errors
        if '=' in step:
            # Check if left and right sides are reasonable
            parts = step.split('=')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                # Simple check: if numbers don't match pattern
                import re
                left_nums = re.findall(r'\d+', left)
                right_nums = re.findall(r'\d+', right)
                if left_nums and right_nums:
                    # Mock: 20% chance of detecting error
                    if random.random() < 0.2:
                        has_error = True
                        error_type = "arithmetic_error"
        
        return {
            'has_error': has_error,
            'error_type': error_type,
            'confidence': random.uniform(0.7, 0.9)
        }
    
    def query_llm(self, step: str, problem: str, prev_steps: List[str], model_id: int) -> Dict:
        """Query a single LLM (mock or real).
        
        Args:
            step: Step text
            problem: Problem statement
            prev_steps: Previous steps
            model_id: Model identifier
            
        Returns:
            LLM result dict
        """
        if not self.use_apis or not self.llm_providers:
            return self.mock_llm_check(step, problem, prev_steps)
        
        # Use real LLM if available
        provider_idx = model_id % len(self.llm_providers) if self.llm_providers else 0
        provider = self.llm_providers[provider_idx]
        
        try:
            result = provider.check_step(step, problem, prev_steps)
            return {
                'has_error': result.get('has_error', False),
                'error_type': result.get('error_type'),
                'confidence': result.get('confidence', 0.8)
            }
        except Exception as e:
            print(f"Error querying {provider.provider}: {e}")
            return self.mock_llm_check(step, problem, prev_steps)
    
    def verify(self, step: str, problem: str = "", prev_steps: List[str] = None) -> Dict:
        """Main verification method using ensemble voting.
        
        Args:
            step: Step text to verify
            problem: Problem statement
            prev_steps: Previous steps
            
        Returns:
            Verification result dict
        """
        if prev_steps is None:
            prev_steps = []
        
        # Query all models
        results = []
        actual_num = len(self.llm_providers) if self.use_apis and self.llm_providers else self.num_models
        for i in range(actual_num):
            result = self.query_llm(step, problem, prev_steps, i)
            results.append(result)
        
        # Count votes
        num_models_used = len(results)
        error_votes = sum(1 for r in results if r.get('has_error', False))
        valid_votes = num_models_used - error_votes
        
        # Calculate average confidence
        avg_confidence = sum(r.get('confidence', 0.5) for r in results) / len(results)
        
        # Majority vote
        if error_votes > valid_votes:
            # Get most common error type
            error_types = [r.get('error_type') for r in results if r.get('has_error')]
            error_type = max(set(error_types), key=error_types.count) if error_types else 'logical_error'
            
            # Confidence based on agreement
            agreement_ratio = error_votes / num_models_used if num_models_used > 0 else 0.5
            confidence = self.confidence_base + (agreement_ratio * 0.15)
            
            return {
                'verdict': 'ERROR',
                'confidence': min(confidence, 0.95),
                'error_type': error_type,
                'details': f'Majority vote: {error_votes}/{num_models_used} models detected error',
                'votes': {'error': error_votes, 'valid': valid_votes, 'total': num_models_used}
            }
        else:
            # Confidence based on agreement
            agreement_ratio = valid_votes / num_models_used if num_models_used > 0 else 0.5
            confidence = self.confidence_base + (agreement_ratio * 0.15)
            
            return {
                'verdict': 'VALID',
                'confidence': min(confidence, 0.95),
                'error_type': None,
                'details': f'Majority vote: {valid_votes}/{num_models_used} models found no error',
                'votes': {'error': error_votes, 'valid': valid_votes, 'total': num_models_used}
            }

