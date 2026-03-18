"""
Main integration pipeline for MathVerify-Integrated system.

This module integrates all components: input processing, reasoning, verification,
and error classification into a complete end-to-end pipeline.
"""

import logging
from typing import Dict, List, Optional, Any
from src.input_module.processor import InputProcessor
from src.reasoning_module.engine import ReasoningEngine
from src.verification_module.verifier import SymbolicVerifier
from src.verification_module.error_taxonomy import ErrorTaxonomy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathVerifyPipeline:
    """
    Complete end-to-end pipeline for mathematical reasoning with verification.
    
    Integrates:
    - Input processing and normalization
    - LLM-based reasoning
    - Symbolic verification
    - Error classification and correction
    
    Attributes:
        input_processor: InputProcessor instance
        reasoning_engine: ReasoningEngine instance
        verifier: SymbolicVerifier instance
        error_taxonomy: ErrorTaxonomy instance
    
    Example:
        >>> pipeline = MathVerifyPipeline()
        >>> result = pipeline.process_problem("Solve: 2x + 3 = 7")
        >>> print(result['final_answer'])
        >>> print(result['errors'])
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: Optional[str] = None
    ):
        """
        Initialize the complete pipeline.
        
        Args:
            model_name: Model name for reasoning engine
            device: Device to run model on
        """
        logger.info("Initializing MathVerifyPipeline...")
        
        self.input_processor = InputProcessor()
        self.reasoning_engine = ReasoningEngine(model_name=model_name, device=device)
        self.verifier = SymbolicVerifier()
        self.error_taxonomy = ErrorTaxonomy()
        
        logger.info("Pipeline initialized successfully")
    
    def process_problem(self, problem: str) -> Dict[str, Any]:
        """
        Process a mathematical problem through the complete pipeline.
        
        Executes:
        1. Input validation and normalization
        2. Reasoning step generation
        3. Step-by-step verification
        4. Error classification
        5. Error correction attempts
        
        Args:
            problem: Mathematical problem statement
            
        Returns:
            Dictionary with keys:
                - problem: Original problem
                - solution: Solution dictionary with steps
                - verification: List of verification results
                - errors: Error report dictionary
                - final_answer: Final answer string
                - confidence: Confidence score (0.0-1.0)
        """
        logger.info(f"Processing problem: {problem[:50]}...")
        
        # Step 1: Input processing
        if not self.input_processor.validate_input(problem):
            return {
                "problem": problem,
                "solution": {"steps": [], "final_answer": "", "confidence": 0.0},
                "verification": [],
                "errors": {"total_errors": 1, "by_type": {"NOTATION_ERROR": 1}},
                "final_answer": "Invalid input",
                "confidence": 0.0
            }
        
        normalized_problem = self.input_processor.normalize_expression(problem)
        components = self.input_processor.extract_problem_components(problem)
        
        # Step 2: Generate solution
        logger.info("Generating solution...")
        solution = self.reasoning_engine.generate_solution(normalized_problem)
        
        # Step 3: Verify and correct steps
        logger.info("Verifying steps...")
        verified_steps = self.verify_and_correct(solution.get("steps", []))
        
        # Step 4: Classify errors
        errors = self._classify_all_errors(verified_steps)
        
        # Step 5: Calculate final confidence
        confidence = self._calculate_final_confidence(
            solution.get("confidence", 0.0),
            errors
        )
        
        return {
            "problem": problem,
            "solution": {
                "steps": verified_steps,
                "final_answer": solution.get("final_answer", ""),
                "confidence": confidence
            },
            "verification": [step.get("verification", {}) for step in verified_steps],
            "errors": errors,
            "final_answer": solution.get("final_answer", ""),
            "confidence": confidence
        }
    
    def verify_and_correct(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify each step and attempt corrections on failures.
        
        Args:
            steps: List of reasoning step dictionaries
            
        Returns:
            List of verified steps with corrections applied
        """
        verified_steps = []
        context = {}
        
        for i, step in enumerate(steps):
            step_content = step.get("content", "")
            step_number = step.get("number", i + 1)
            
            # Verify the step
            verification_result = self.verifier.verify_step(
                step_content,
                context=context
            )
            
            # Classify error if verification failed
            error_type = None
            correction = None
            
            if not verification_result.get("valid", True):
                error_type = self.error_taxonomy.classify_error(
                    step_content,
                    verification_result
                )
                correction = self.error_taxonomy.suggest_correction(
                    error_type,
                    step_content
                )
                
                # Attempt automatic correction for simple cases
                corrected_content = self._attempt_correction(
                    step_content,
                    error_type,
                    verification_result
                )
                
                if corrected_content != step_content:
                    # Re-verify corrected step
                    new_verification = self.verifier.verify_step(
                        corrected_content,
                        context=context
                    )
                    if new_verification.get("valid", False):
                        step_content = corrected_content
                        verification_result = new_verification
                        error_type = None
                        correction = "Auto-corrected"
            
            # Build verified step
            verified_step = {
                "number": step_number,
                "content": step_content,
                "rationale": step.get("rationale", ""),
                "verification": verification_result,
                "error_type": error_type,
                "correction": correction,
                "is_valid": verification_result.get("valid", True)
            }
            
            verified_steps.append(verified_step)
            
            # Update context for next step
            context["previous_steps"] = verified_steps
        
        return verified_steps
    
    def _attempt_correction(
        self,
        step_content: str,
        error_type: str,
        verification_result: Dict
    ) -> str:
        """
        Attempt automatic correction of a step based on error type.
        
        Args:
            step_content: Original step content
            error_type: Classified error type
            verification_result: Verification result
            
        Returns:
            Corrected step content (or original if correction not possible)
        """
        # Only attempt corrections for calculation errors with clear fixes
        if error_type == "CALCULATION_ERROR":
            # Try to extract and recalculate simple arithmetic
            import re
            # Pattern: number operator number = result
            pattern = r'(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)'
            match = re.search(pattern, step_content)
            
            if match:
                num1, op, num2, wrong_result = match.groups()
                num1, num2 = int(num1), int(num2)
                
                # Calculate correct result
                if op == '+':
                    correct = num1 + num2
                elif op == '-':
                    correct = num1 - num2
                elif op == '*':
                    correct = num1 * num2
                elif op == '/':
                    if num2 != 0:
                        correct = num1 / num2
                    else:
                        return step_content  # Division by zero, can't fix
                else:
                    return step_content
                
                # Replace wrong result with correct one
                corrected = step_content.replace(
                    f"{num1} {op} {num2} = {wrong_result}",
                    f"{num1} {op} {num2} = {correct}"
                )
                return corrected
        
        # For other error types, return original (manual correction needed)
        return step_content
    
    def _classify_all_errors(self, verified_steps: List[Dict]) -> Dict[str, Any]:
        """
        Classify all errors found in verified steps.
        
        Args:
            verified_steps: List of verified step dictionaries
            
        Returns:
            Error report dictionary
        """
        errors = []
        
        for step in verified_steps:
            if not step.get("is_valid", True):
                error_type = step.get("error_type", "UNKNOWN")
                errors.append({
                    "type": error_type,
                    "step_number": step.get("number"),
                    "content": step.get("content", ""),
                    "correction": step.get("correction", "")
                })
        
        return self.error_taxonomy.generate_report(errors)
    
    def _calculate_final_confidence(
        self,
        base_confidence: float,
        errors: Dict[str, Any]
    ) -> float:
        """
        Calculate final confidence score considering errors.
        
        Args:
            base_confidence: Base confidence from reasoning engine
            errors: Error report dictionary
            
        Returns:
            Adjusted confidence score (0.0-1.0)
        """
        total_errors = errors.get("total_errors", 0)
        
        # Reduce confidence based on number of errors
        error_penalty = min(total_errors * 0.1, 0.5)  # Max 50% penalty
        
        final_confidence = max(0.0, base_confidence - error_penalty)
        
        return round(final_confidence, 2)

