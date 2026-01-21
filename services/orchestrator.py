"""
Main Orchestrator - MULTIMODAL COORDINATOR
Coordinates all microservices and implements novel consensus algorithm
"""
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from services.ml_classifier import predict_errors

class MathVerificationOrchestrator:
    def __init__(self):
        self.ocr_url = "http://localhost:8001/extract"
        self.sympy_url = "http://localhost:8005/verify"
        self.llm_url = "http://localhost:8003/verify"

    def verify_from_image(self, image_path: str) -> Dict:
        """
        MULTIMODAL PIPELINE: Image → OCR → Verification → Consensus
        This is the novel contribution!
        """
        print("[INFO] Processing image input...")
        
        # Step 1: OCR Extraction
        with open(image_path, 'rb') as f:
            ocr_response = requests.post(
                self.ocr_url,
                files={'file': f},
                timeout=30
            )
        
        if ocr_response.status_code != 200:
            return {'error': 'OCR failed', 'details': ocr_response.text}
        
        ocr_data = ocr_response.json()
        problem = ocr_data['problem']
        steps = ocr_data['steps']
        ocr_confidence = ocr_data['ocr_confidence']
        
        print(f"[OK] OCR Complete - Confidence: {ocr_confidence*100:.1f}%")
        print(f"   Problem: {problem}")
        print(f"   Steps: {len(steps)} detected")
        
        # Step 2: Verification with OCR confidence
        return self.verify(problem, steps, ocr_confidence, source='image')
    
    def verify(self, 
               problem: str, 
               steps: List[str], 
               ocr_confidence: float = 1.0,
               source: str = 'text') -> Dict:
        """
        Verify solution using all microservices
        Implements NOVEL weighted consensus algorithm
        """
        start = time.time()
        
        print(f"[INFO] Starting verification (source: {source})...")
        
        # Parallel execution of all verifiers
        with ThreadPoolExecutor(max_workers=3) as executor:
            f1 = executor.submit(self._call_sympy, steps, problem)  # Pass problem for Math-Verify
            f2 = executor.submit(self._call_llm, problem, steps)
            f3 = executor.submit(self._call_ml_classifier, steps)  # REAL ML now!
            
            # Collect results
            sympy_result = f1.result()
            llm_result = f2.result()
            ml_result = f3.result()
        
        print("[OK] All verifiers complete")
        
        # NOVEL: Weighted consensus with OCR-aware calibration
        consensus = self._weighted_consensus({
            'symbolic': sympy_result,
            'llm': llm_result,
            'ml_classifier': ml_result
        }, ocr_confidence)
        
        # Metadata
        consensus['problem'] = problem
        consensus['steps'] = steps
        consensus['processing_time'] = time.time() - start
        consensus['input_source'] = source
        consensus['ocr_confidence'] = ocr_confidence if source == 'image' else None
        
        return consensus
    
    def _call_sympy(self, steps: List[str], problem: str = "") -> Dict:
        """Call Enhanced SymPy verification service with Math-Verify"""
        try:
            response = requests.post(
                self.sympy_url,
                json={'steps': steps, 'problem': problem, 'use_math_verify': True},
                timeout=5
            )
            return response.json()
        except Exception as e:
            print(f"[WARN] SymPy service failed: {e}")
            return {
                'model': 'symbolic',
                'model_name': '[Symbolic] Symbolic (Offline)',
                'verdict': 'UNKNOWN',
                'confidence': 0.0,
                'errors': []
            }
    
    def _call_llm(self, problem: str, steps: List[str]) -> Dict:
        """Call LLM ensemble service"""
        try:
            response = requests.post(
                self.llm_url,
                json={'problem': problem, 'steps': steps},
                timeout=15
            )
            return response.json()
        except Exception as e:
            print(f"[WARN] LLM service failed: {e}")
            return {
                'model': 'ensemble',
                'model_name': '[LLM] LLM (Offline)',
                'verdict': 'UNKNOWN',
                'confidence': 0.0
            }
    
    def _call_ml_classifier(self, steps: List[str]) -> Dict:
        """
        Call REAL ML classifier (TF-IDF + Naive Bayes)
        Trained on mathematical error patterns
        """
        try:
            result = predict_errors(steps)
            return result
        except Exception as e:
            print(f"[WARN] ML classifier failed: {e}")
            return {
                'model': 'ml_classifier',
                'model_name': '[ML] ML Classifier (Offline)',
                'verdict': 'UNKNOWN',
                'confidence': 0.0
            }
    
    def _weighted_consensus(self, results: Dict, ocr_confidence: float) -> Dict:
        """
        NOVEL CONTRIBUTION: Adaptive weighted consensus with OCR calibration
        
        This is the key innovation of your research!
        """
        # Weights based on model complementarity
        weights = {
            'symbolic': 0.40,      # Highest: deterministic
            'llm': 0.35,           # High: semantic reasoning
            'ml_classifier': 0.25  # Medium: learned patterns
        }
        
        # Calculate weighted error score
        error_score = 0
        for model, result in results.items():
            if result.get('verdict') == 'ERROR':
                confidence = result.get('confidence', 0)
                error_score += weights[model] * confidence
        
        # Threshold: >0.50 = ERROR
        final_verdict = "ERROR" if error_score > 0.50 else "VALID"
        
        # Agreement analysis
        verdicts = [r.get('verdict') for r in results.values()]
        unique_verdicts = set(v for v in verdicts if v != 'UNKNOWN')
        
        if len(unique_verdicts) == 1:
            agreement = "UNANIMOUS (3/3)"
            conf_boost = 1.1
        elif verdicts.count(final_verdict) >= 2:
            agreement = "MAJORITY (2/3)"
            conf_boost = 1.0
        else:
            agreement = "MIXED"
            conf_boost = 0.8
        
        # Calculate overall confidence
        agreeing = [r for r in results.values() if r.get('verdict') == final_verdict]
        if agreeing:
            overall_conf = sum(r.get('confidence', 0) for r in agreeing) / len(agreeing)
            overall_conf = min(overall_conf * conf_boost, 0.99)
        else:
            overall_conf = 0.5
        
        # NOVEL: OCR-aware calibration
        # If OCR confidence is low, reduce final confidence
        if ocr_confidence < 0.85:
            calibration_factor = 0.9 + 0.1 * ocr_confidence
            overall_conf *= calibration_factor
            print(f"[INFO] OCR calibration applied: {calibration_factor:.2f}x")
        
        # Collect all errors from symbolic verifier
        all_errors = []
        for result in results.values():
            all_errors.extend(result.get('errors', []))
        
        return {
            'final_verdict': final_verdict,
            'overall_confidence': overall_conf,
            'error_score': error_score,
            'agreement_type': agreement,
            'individual_results': results,
            'all_errors': all_errors,
            'weights_used': weights
        }
