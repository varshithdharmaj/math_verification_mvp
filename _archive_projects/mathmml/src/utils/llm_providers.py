"""Unified LLM provider interface for multiple models (GPT, Gemini, Llama)."""

import os
import json
from typing import Dict, Optional, List
import requests

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars


class LLMProvider:
    """Unified interface for different LLM providers."""
    
    def __init__(self, provider: str, model: str = None):
        """Initialize provider.
        
        Args:
            provider: "openai", "gemini", "llama" (via Ollama), "anthropic"
            model: Specific model name (e.g., "gpt-4", "gemini-pro", "llama2")
        """
        self.provider = provider
        self.model = model or self._get_default_model()
        self.client = None
        self._init_client()
    
    def _get_default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-3.5-turbo",
            "gemini": "gemini-pro",
            "llama": "llama2",
            "anthropic": "claude-3-sonnet-20240229"
        }
        return defaults.get(self.provider, "gpt-3.5-turbo")
    
    def _init_client(self):
        """Initialize API client."""
        if self.provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                else:
                    print(f"Warning: OPENAI_API_KEY not found for {self.provider}")
            except ImportError:
                print(f"Warning: openai package not installed")
        
        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.client = genai
                else:
                    print(f"Warning: GOOGLE_API_KEY or GEMINI_API_KEY not found")
            except ImportError:
                print(f"Warning: google-generativeai package not installed")
        
        elif self.provider == "llama":
            # Ollama runs locally, no API key needed
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.client = "ollama"  # Marker for Ollama
        
        elif self.provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
                else:
                    print(f"Warning: ANTHROPIC_API_KEY not found")
            except ImportError:
                print(f"Warning: anthropic package not installed")
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 500) -> Optional[str]:
        """Generate response from LLM.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or None
        """
        if not self.client:
            return None
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif self.provider == "gemini":
                model = self.client.GenerativeModel(self.model)
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens
                    }
                )
                return response.text
            
            elif self.provider == "llama":
                # Ollama API
                url = f"{self.ollama_base_url}/api/generate"
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                response = requests.post(url, json=payload, timeout=30)
                if response.status_code == 200:
                    return response.json().get("response")
                else:
                    print(f"Ollama error: {response.status_code}")
                    return None
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        
        except Exception as e:
            print(f"Error calling {self.provider} ({self.model}): {e}")
            return None
        
        return None
    
    def check_step(self, step: str, problem: str, prev_steps: List[str]) -> Dict:
        """Check a step for errors using LLM.
        
        Args:
            step: Current step text
            problem: Problem statement
            prev_steps: Previous steps
            
        Returns:
            Dict with has_error, error_type, reason, confidence
        """
        context = "\n".join(prev_steps[-3:]) if prev_steps else ""
        prompt = f"""You are a mathematical reasoning verifier. Analyze if the following step contains any errors.

Problem: {problem}

Previous steps:
{context}

Current step: {step}

Does this step contain any logical errors, contradictions, or inconsistencies? 
Respond with ONLY valid JSON in this exact format:
{{"has_error": true/false, "error_type": "type", "reason": "brief explanation", "confidence": 0.0-1.0}}"""

        response_text = self.generate(prompt, temperature=0.0)
        
        if not response_text:
            return {
                'has_error': False,
                'error_type': None,
                'reason': 'LLM call failed',
                'confidence': 0.5
            }
        
        # Try to parse JSON from response
        try:
            # Extract JSON from response (in case there's extra text)
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = json.loads(response_text)
            
            return {
                'has_error': result.get('has_error', False),
                'error_type': result.get('error_type', 'logical_error'),
                'reason': result.get('reason', ''),
                'confidence': result.get('confidence', 0.8)
            }
        except:
            # Fallback: try to infer from text
            has_error = any(word in response_text.lower() for word in ['error', 'incorrect', 'wrong', 'mistake'])
            return {
                'has_error': has_error,
                'error_type': 'logical_error' if has_error else None,
                'reason': response_text[:200],
                'confidence': 0.7
            }


def get_available_providers() -> List[str]:
    """Get list of available providers based on installed packages and API keys.
    
    Returns:
        List of provider names
    """
    available = []
    
    # Check OpenAI
    try:
        import openai
        if os.getenv("OPENAI_API_KEY"):
            available.append("openai")
    except:
        pass
    
    # Check Gemini
    try:
        import google.generativeai
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            available.append("gemini")
    except:
        pass
    
    # Check Ollama (Llama)
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            available.append("llama")
    except:
        pass
    
    # Check Anthropic
    try:
        import anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            available.append("anthropic")
    except:
        pass
    
    return available

