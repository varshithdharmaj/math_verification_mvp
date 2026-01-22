"""
Orchestrator Module
Bridge between the benchmark scripts (which expect MathVerificationOrchestrator) 
and the new modular backend services.
"""
import os
import asyncio
from typing import Dict, List, Any, Optional

# Import new backend services
from backend.core.representation_service import to_canonical_expression
from backend.core.verification_service import verify_step_by_step
from backend.core.classifier_service import classify_and_score
from backend.core.ocr_service import ocr_engine

class MathVerificationOrchestrator:
    """
    Orchestrates the verification pipeline.
    This replaces the legacy services/orchestrator.py
    """
    
    def __init__(self):
        pass
        
    def verify(self, problem_text: str, steps: List[str], mode: str = "full_mvm2") -> Dict[str, Any]:
        """
        Verify math solution from text input (Problem + Steps)
        Synchronous wrapper around async backend logic.
        """
        return asyncio.run(self._verify_async(problem_text, steps, mode))
        
    async def _verify_async(self, problem_text: str, steps: List[str], mode: str) -> Dict[str, Any]:
        """
        Async verification logic mirroring backend.main.solve_text
        """
        # 1. Representation (Manual construction since we have structural input)
        canonical_problem = to_canonical_expression(problem_text)
        canonical_steps = [to_canonical_expression(s) for s in steps]
        
        canonical_input = {
            "problem": canonical_problem,
            "steps": canonical_steps,
            "format": "canonical_latex",
            "raw_problem": problem_text,
            "raw_steps": steps
        }
        
        # 2. Verification (SymPy + LLM)
        verification_result = await verify_step_by_step(canonical_input, mode=mode)
        
        # 3. Classification & Scoring
        ocr_confidence = 1.0  # Text input defaults to high confidence
        
        # Disable calibration if requested mode implies it
        use_calibration = True
        if mode == "multi_agent_no_ocr_conf":
            use_calibration = False

        use_classifier = True
        if mode in ["single_llm_only", "llm_plus_sympy", "multi_agent_no_classifier"]:
            use_classifier = False
            
        final_result = await classify_and_score(
            verification_result,
            ocr_confidence,
            use_ocr_calibration=use_calibration,
            use_classifier=use_classifier
        )
        
        # Add keys expected by benchmark scripts
        final_result['overall_confidence'] = final_result.get('final_confidence', 0.0)
        final_result['final_verdict'] = final_result.get('verdict', "UNKNOWN")
        
        final_result["ocr_confidence"] = ocr_confidence
        final_result["verification"] = verification_result
        final_result["input_type"] = "text"
        final_result["mode"] = mode
        return final_result

    def verify_from_image(self, image_path: str, mode: str = "full_mvm2") -> Dict[str, Any]:
        """
        Verify math solution from image path.
        Synchronous wrapper around async backend logic.
        """
        return asyncio.run(self._verify_from_image_async(image_path, mode))
        
    async def _verify_from_image_async(self, image_path: str, mode: str) -> Dict[str, Any]:
        """
        Async verification logic mirroring backend.main.solve_image
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        # 1. OCR (get text and confidence)
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_bytes))
        ocr_data = ocr_engine.extract_text(image)
        
        problem_text = ocr_data.get('problem', '')
        steps = ocr_data.get('steps', [])
        ocr_confidence = ocr_data.get('ocr_confidence', 0.5)
        
        # 2. Representation
        canonical_problem = to_canonical_expression(problem_text)
        canonical_steps = [to_canonical_expression(s) for s in steps]
        
        canonical_input = {
            "problem": canonical_problem,
            "steps": canonical_steps,
            "format": "canonical_latex",
            "raw_problem": problem_text,
            "raw_steps": steps
        }
        
        # 3. Verification
        verification_result = await verify_step_by_step(canonical_input, mode=mode)
        
        # 4. Classification
        use_calibration = True
        if mode == "multi_agent_no_ocr_conf":
            use_calibration = False

        use_classifier = True
        if mode in ["single_llm_only", "llm_plus_sympy", "multi_agent_no_classifier"]:
            use_classifier = False
            
        final_result = await classify_and_score(
            verification_result,
            ocr_confidence,
            use_ocr_calibration=use_calibration,
            use_classifier=use_classifier
        )
        
        # Add keys expected by benchmarks
        final_result['overall_confidence'] = final_result.get('final_confidence', 0.0)
        final_result['final_verdict'] = final_result.get('verdict', "UNKNOWN")
        
        final_result["ocr_confidence"] = ocr_confidence
        final_result["ocr_text"] = ocr_data.get("extracted_text", "")
        final_result["ocr_normalized_text"] = ocr_data.get("normalized_text", "")
        final_result["verification"] = verification_result
        final_result["input_type"] = "image"
        final_result["mode"] = mode
        return final_result
