"""Weighted consensus mechanism for 4-model verification."""

from typing import Dict, List, Optional
from enum import Enum


class AgreementType(Enum):
    """Types of model agreement."""
    UNANIMOUS = "UNANIMOUS ✓✓✓"
    MAJORITY = "MAJORITY (3/4) ✓✓"
    MIXED = "MIXED ✓"


class ConsensusEngine:
    """Computes weighted consensus from multiple verifiers."""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ):
        """Initialize consensus engine.
        
        Args:
            weights: Dict mapping verifier names to weights
                    Default: Symbolic=0.40, LLM_Logical=0.35, Ensemble=0.20, ML_Classifier=0.25 (Spec-specified)
        """
        if weights is None:
            # Spec-specified weights: Symbolic 40%, LLM 35%, Ensemble 20%, ML 25%
            self.weights = {
                'symbolic': 0.40,
                'llm_logical': 0.35,
                'ensemble': 0.20,
                'ml_classifier': 0.25
            }
        else:
            self.weights = weights
        
        # Preserve raw weights for reporting while using normalized weights internally
        total = sum(self.weights.values()) or 1.0
        self.normalized_weights = {k: v / total for k, v in self.weights.items()}
        self.threshold = 0.50  # Spec-defined error score threshold
    
    def compute_consensus(
        self,
        results: Dict[str, Dict]
    ) -> Dict:
        """Compute weighted consensus from verifier results.
        
        Args:
            results: Dict mapping verifier names to their result dicts
                    Each result should have 'verdict' (VALID/ERROR) and 'confidence'
        
        Returns:
            Consensus result with:
            - final_verdict: VALID or ERROR
            - overall_confidence: float
            - agreement_type: UNANIMOUS/MAJORITY/MIXED
            - error_score: weighted error score
            - breakdown: per-verifier contributions
        """
        if not results:
            return {
                'final_verdict': 'UNKNOWN',
                'overall_confidence': 0.0,
                'agreement_type': AgreementType.MIXED.value,
                'error_score': 0.0,
                'breakdown': {}
            }
        
        # Compute error score (spec algorithm)
        error_score = 0.0
        breakdown = {}
        confidences = []
        verdicts = []
        
        for verifier_name, result in results.items():
            raw_weight = self.weights.get(verifier_name, 0.0)
            weight = self.normalized_weights.get(verifier_name, 0.0)
            verdict = result.get('verdict', 'UNKNOWN')
            confidence = result.get('confidence', 0.5)
            
            confidences.append(confidence)
            verdicts.append(verdict)
            
            # Contribution to error score (spec: only ERROR verdicts contribute)
            if verdict == 'ERROR':
                contribution_weight = raw_weight
                
                # Boost high-confidence symbolic verifier contributions
                if verifier_name == 'symbolic' and confidence >= 0.9:
                    contribution_weight *= 1.5
                
                contribution = contribution_weight * confidence
                error_score += contribution
            else:
                contribution = 0.0  # VALID verdicts don't contribute to error_score
            
            breakdown[verifier_name] = {
                'verdict': verdict,
                'confidence': confidence,
                'weight': raw_weight,
                'normalized_weight': weight,
                'effective_weight': contribution / confidence if (confidence and contribution) else raw_weight,
                'contribution': contribution,
                'error_type': result.get('error_type'),
                'details': result.get('details', '')
            }
        
        # Determine agreement type (spec format)
        error_count = sum(1 for v in verdicts if v == 'ERROR')
        valid_count = sum(1 for v in verdicts if v == 'VALID')
        total = len(verdicts)
        
        if error_count == total or valid_count == total:
            agreement_type = AgreementType.UNANIMOUS
        elif error_count >= 3 or valid_count >= 3:  # 3/4 majority
            agreement_type = AgreementType.MAJORITY
        else:
            agreement_type = AgreementType.MIXED
        
        # Final verdict - use error_score threshold (spec: error_score > 0.50)
        if error_score > self.threshold:
            final_verdict = 'ERROR'
        else:
            final_verdict = 'VALID'
        
        # Overall confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        if agreement_type == AgreementType.UNANIMOUS:
            agreeing_confs = [c for v, c in zip(verdicts, confidences) if v == final_verdict]
            base_conf = sum(agreeing_confs) / len(agreeing_confs) if agreeing_confs else avg_confidence
            overall_confidence = min(base_conf + 0.2, 0.99)
        elif agreement_type == AgreementType.MAJORITY:
            # Average of agreeing models
            if final_verdict == 'ERROR':
                agreeing_confs = [c for v, c in zip(verdicts, confidences) if v == 'ERROR']
            else:
                agreeing_confs = [c for v, c in zip(verdicts, confidences) if v == 'VALID']
            if agreeing_confs:
                base_conf = sum(agreeing_confs) / len(agreeing_confs)
                overall_confidence = min(base_conf + 0.1, 0.95)
            else:
                overall_confidence = avg_confidence
        else:  # MIXED
            overall_confidence = max(avg_confidence * 0.75, min(confidences) if confidences else 0.5)
        
        # Aggregate error types
        error_types = [r.get('error_type') for r in results.values() if r.get('verdict') == 'ERROR' and r.get('error_type')]
        primary_error_type = max(set(error_types), key=error_types.count) if error_types else None
        
        return {
            'final_verdict': final_verdict,
            'overall_confidence': overall_confidence,
            'agreement_type': agreement_type.value,
            'error_score': error_score,
            'primary_error_type': primary_error_type,
            'breakdown': breakdown,
            'verdict_counts': {'error': error_count, 'valid': valid_count, 'total': total}
        }
    
    def verify_step(
        self,
        step: str,
        problem: str,
        prev_steps: List[str],
        verifiers: Dict[str, any]
    ) -> Dict:
        """Run all verifiers sequentially and compute consensus.
        
        Note: For parallel execution, use VerificationOrchestrator instead.
        
        Args:
            step: Step text to verify
            problem: Problem statement
            prev_steps: Previous steps
            verifiers: Dict mapping names to verifier instances
            
        Returns:
            Full consensus result with per-verifier results
        """
        # Run all verifiers sequentially (for backward compatibility)
        results = {}
        for name, verifier in verifiers.items():
            try:
                result = verifier.verify(step, problem, prev_steps)
                results[name] = result
            except Exception as e:
                # Fallback on error
                results[name] = {
                    'verdict': 'UNKNOWN',
                    'confidence': 0.0,
                    'error_type': None,
                    'details': f'Error: {str(e)}'
                }
        
        # Compute consensus
        consensus = self.compute_consensus(results)
        consensus['per_verifier_results'] = results
        
        return consensus

