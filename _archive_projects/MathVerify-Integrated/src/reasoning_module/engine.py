"""
Reasoning engine for LLM-based mathematical problem solving.

This module uses transformer models to generate step-by-step solutions
to mathematical problems using chain-of-thought reasoning.
"""

import logging
import re
from typing import Dict, List, Optional, Any
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    LLM-based reasoning engine for mathematical problem solving.
    
    Uses transformer models (LLaMA/Mistral) to generate step-by-step solutions
    with chain-of-thought prompting.
    
    Attributes:
        model_name: Name of the model to use (default: "meta-llama/Llama-2-7b-hf")
        max_steps: Maximum number of reasoning steps (default: 10)
        device: Device to run model on ("cuda" or "cpu")
        model: Loaded model instance
        tokenizer: Loaded tokenizer instance
        generator: Pipeline for text generation
        
    Example:
        >>> engine = ReasoningEngine(model_name="meta-llama/Llama-2-7b-hf")
        >>> result = engine.generate_solution("Solve: 2x + 3 = 7")
        >>> print(result['final_answer'])
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: Optional[str] = None,
        max_steps: int = 10
    ):
        """
        Initialize the reasoning engine.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            max_steps: Maximum number of reasoning steps
        """
        self.model_name = model_name
        self.max_steps = max_steps
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Initialize model
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Model loading failed. Using fallback mode.")
            self.model = None
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name} on {self.device}")
            
            # Try to load with pipeline first (simpler)
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Pipeline loading failed: {e}")
            # Fallback: try loading model and tokenizer separately
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                logger.info("Model loaded via fallback method")
            except Exception as e2:
                logger.error(f"Fallback loading also failed: {e2}")
                raise
    
    def _create_cot_prompt(self, problem: str) -> str:
        """
        Create a chain-of-thought prompt for the problem.
        
        Args:
            problem: Mathematical problem statement
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Solve this mathematical problem step by step. Show your reasoning clearly.

Problem: {problem}

Solution:
Step 1:"""
        return prompt
    
    def generate_solution(self, problem: str) -> Dict[str, Any]:
        """
        Generate step-by-step solution for a mathematical problem.
        
        Args:
            problem: Problem statement
            
        Returns:
            Dictionary with keys:
                - steps: List of step dictionaries with 'number', 'content', 'rationale'
                - final_answer: Final answer string
                - confidence: Confidence score (0.0-1.0)
        """
        if not self.generator and not self.model:
            # Fallback: return mock solution if model not loaded
            logger.warning("Model not loaded, returning mock solution")
            return {
                "steps": [
                    {
                        "number": 1,
                        "content": "Analyze the problem",
                        "rationale": "Model not available, using fallback"
                    }
                ],
                "final_answer": "Unable to generate solution (model not loaded)",
                "confidence": 0.0
            }
        
        try:
            prompt = self._create_cot_prompt(problem)
            
            # Generate solution
            if self.generator:
                # Use pipeline
                output = self.generator(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.generator.tokenizer.eos_token_id,
                )
                generated_text = output[0]['generated_text']
            else:
                # Use model directly
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract steps from generated text
            steps = self._parse_steps(generated_text, prompt)
            final_answer = self._extract_final_answer(generated_text)
            
            return {
                "steps": steps,
                "final_answer": final_answer,
                "confidence": self._calculate_confidence(steps, final_answer)
            }
            
        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return {
                "steps": [],
                "final_answer": f"Error: {str(e)}",
                "confidence": 0.0
            }
    
    def generate_single_step(
        self,
        problem: str,
        previous_steps: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Generate the next reasoning step given previous steps.
        
        Args:
            problem: Original problem statement
            previous_steps: List of previous step dictionaries
            
        Returns:
            Dictionary with 'content' and 'rationale' keys
        """
        # Build context from previous steps
        context = "\n".join([
            f"Step {i+1}: {step.get('content', '')}"
            for i, step in enumerate(previous_steps)
        ])
        
        prompt = f"""Problem: {problem}

Previous steps:
{context}

Next step:"""
        
        try:
            if self.generator:
                output = self.generator(
                    prompt,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=1,
                )
                generated = output[0]['generated_text']
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        do_sample=True,
                    )
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the new step
            new_content = generated[len(prompt):].strip()
            
            return {
                "content": new_content,
                "rationale": "Generated from previous context"
            }
        except Exception as e:
            logger.error(f"Error generating single step: {e}")
            return {
                "content": f"Error: {str(e)}",
                "rationale": "Failed to generate step"
            }
    
    def _parse_steps(self, generated_text: str, original_prompt: str) -> List[Dict[str, Any]]:
        """
        Parse generated text into structured steps.
        
        Args:
            generated_text: Full generated text
            original_prompt: Original prompt (to remove from output)
            
        Returns:
            List of step dictionaries
        """
        # Remove the prompt from generated text
        solution_text = generated_text[len(original_prompt):].strip()
        
        steps = []
        # Look for step markers: "Step 1:", "Step 2:", etc.
        step_pattern = r'Step\s+(\d+):\s*(.*?)(?=Step\s+\d+:|$)'
        matches = re.findall(step_pattern, solution_text, re.DOTALL | re.IGNORECASE)
        
        for step_num, content in matches:
            steps.append({
                "number": int(step_num),
                "content": content.strip(),
                "rationale": "Generated by reasoning engine"
            })
        
        # If no step markers found, split by newlines
        if not steps:
            lines = solution_text.split('\n')
            for i, line in enumerate(lines[:self.max_steps], 1):
                if line.strip():
                    steps.append({
                        "number": i,
                        "content": line.strip(),
                        "rationale": "Parsed from generated text"
                    })
        
        return steps
    
    def _extract_final_answer(self, generated_text: str) -> str:
        """
        Extract final answer from generated text.
        
        Args:
            generated_text: Full generated text
            
        Returns:
            Final answer string
        """
        # Look for common answer patterns
        patterns = [
            r'Final\s+answer[:\s]+(.*?)(?:\n|$)',
            r'Answer[:\s]+(.*?)(?:\n|$)',
            r'Therefore[,\s]+(.*?)(?:\n|$)',
            r'=\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, generated_text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: return last line
        lines = generated_text.split('\n')
        for line in reversed(lines):
            if line.strip() and not line.strip().startswith('Step'):
                return line.strip()
        
        return "Answer not found"
    
    def _calculate_confidence(
        self,
        steps: List[Dict[str, Any]],
        final_answer: str
    ) -> float:
        """
        Calculate confidence score for the solution.
        
        Args:
            steps: List of reasoning steps
            final_answer: Final answer string
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not steps:
            return 0.0
        
        # Base confidence on number of steps and answer presence
        step_score = min(len(steps) / self.max_steps, 1.0)
        answer_score = 1.0 if final_answer and "error" not in final_answer.lower() else 0.0
        
        # Combined confidence
        confidence = (step_score * 0.5 + answer_score * 0.5)
        return round(confidence, 2)

