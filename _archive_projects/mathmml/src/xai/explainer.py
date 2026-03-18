"""Main explainer class for generating explanations across all verifiers."""

from typing import Dict, List, Optional, Any
import numpy as np


class XAIExplainer:
    """Explainable AI explainer for the verification system."""
    
    def __init__(self):
        """Initialize XAI explainer."""
        pass
    
    def explain_verifier_decision(
        self,
        verifier_name: str,
        step: str,
        problem: str,
        result: Dict,
        prev_steps: List[str] = None
    ) -> Dict:
        """Generate explanation for a verifier's decision.
        
        Args:
            verifier_name: Name of the verifier
            step: Step text
            problem: Problem statement
            result: Verifier result dict
            prev_steps: Previous steps
            
        Returns:
            Explanation dict with reasoning, evidence, and confidence breakdown
        """
        if prev_steps is None:
            prev_steps = []
        
        explanations = {
            'symbolic': self._explain_symbolic,
            'llm_logical': self._explain_llm_logical,
            'ensemble': self._explain_ensemble,
            'ml_classifier': self._explain_ml_classifier
        }
        
        explainer = explanations.get(verifier_name, self._explain_generic)
        return explainer(step, problem, result, prev_steps)
    
    def _explain_symbolic(self, step: str, problem: str, result: Dict, prev_steps: List[str]) -> Dict:
        """Explain symbolic verifier decision."""
        verdict = result.get('verdict', 'UNKNOWN')
        details = result.get('details', '')
        confidence = result.get('confidence', 0.5)
        
        explanation = {
            'verifier': 'symbolic',
            'verdict': verdict,
            'reasoning': [],
            'evidence': [],
            'confidence_factors': {},
            'key_factors': []
        }
        
        if verdict == 'ERROR':
            explanation['reasoning'].append(
                "The symbolic verifier detected an arithmetic error by evaluating the mathematical expression."
            )
            if 'Calculation error' in details:
                # Extract the numbers from details
                explanation['evidence'].append(details)
                explanation['key_factors'].append("Arithmetic calculation mismatch")
            else:
                explanation['evidence'].append(details)
        elif verdict == 'VALID':
            explanation['reasoning'].append(
                "The symbolic verifier verified the arithmetic by evaluating the expression and confirming the result matches."
            )
            explanation['evidence'].append(details)
            explanation['key_factors'].append("Arithmetic calculation verified")
        
        explanation['confidence_factors'] = {
            'expression_validity': 0.95 if 'evaluates correctly' in details else 0.7,
            'result_match': 0.9 if verdict == 'VALID' else 0.1,
            'calculation_precision': confidence
        }
        
        return explanation
    
    def _explain_llm_logical(self, step: str, problem: str, result: Dict, prev_steps: List[str]) -> Dict:
        """Explain LLM logical checker decision."""
        verdict = result.get('verdict', 'UNKNOWN')
        details = result.get('details', '')
        error_type = result.get('error_type')
        
        explanation = {
            'verifier': 'llm_logical',
            'verdict': verdict,
            'reasoning': [],
            'evidence': [],
            'confidence_factors': {},
            'key_factors': [],
            'heuristics_checked': []
        }
        
        if verdict == 'ERROR':
            explanation['reasoning'].append(
                "The logical checker identified inconsistencies using pattern matching and heuristics."
            )
            if 'Contradiction' in details:
                explanation['heuristics_checked'].append('contradiction_detection')
                explanation['key_factors'].append("Logical contradiction found")
            if 'Operation mismatch' in details:
                explanation['heuristics_checked'].append('operation_mismatch')
                explanation['key_factors'].append("Operation doesn't match problem requirements")
            if 'Circular reasoning' in details:
                explanation['heuristics_checked'].append('circular_reasoning')
                explanation['key_factors'].append("Circular reasoning detected")
        else:
            explanation['reasoning'].append(
                "The logical checker found no logical inconsistencies in the step."
            )
            explanation['heuristics_checked'] = [
                'contradiction_detection',
                'operation_mismatch',
                'circular_reasoning',
                'semantic_consistency'
            ]
        
        explanation['evidence'] = details.split('; ') if ';' in details else [details]
        explanation['confidence_factors'] = {
            'heuristic_match': 0.8 if verdict == 'ERROR' else 0.6,
            'pattern_confidence': result.get('confidence', 0.5)
        }
        
        return explanation
    
    def _explain_ensemble(self, step: str, problem: str, result: Dict, prev_steps: List[str]) -> Dict:
        """Explain ensemble checker decision."""
        verdict = result.get('verdict', 'UNKNOWN')
        votes = result.get('votes', {})
        details = result.get('details', '')
        
        explanation = {
            'verifier': 'ensemble',
            'verdict': verdict,
            'reasoning': [],
            'evidence': [],
            'confidence_factors': {},
            'key_factors': [],
            'voting_breakdown': votes
        }
        
        error_votes = votes.get('error', 0)
        valid_votes = votes.get('valid', 0)
        total = votes.get('total', 1)
        
        explanation['reasoning'].append(
            f"The ensemble checker used majority voting from {total} models: "
            f"{error_votes} voted ERROR, {valid_votes} voted VALID."
        )
        
        if error_votes > valid_votes:
            explanation['key_factors'].append(f"Majority ({error_votes}/{total}) detected error")
        else:
            explanation['key_factors'].append(f"Majority ({valid_votes}/{total}) found no error")
        
        explanation['evidence'] = [details]
        explanation['confidence_factors'] = {
            'agreement_ratio': max(error_votes, valid_votes) / total if total > 0 else 0.5,
            'consensus_strength': result.get('confidence', 0.5)
        }
        
        return explanation
    
    def _explain_ml_classifier(self, step: str, problem: str, result: Dict, prev_steps: List[str]) -> Dict:
        """Explain ML classifier decision."""
        verdict = result.get('verdict', 'UNKNOWN')
        label = result.get('error_type', 'correct')
        all_probs = result.get('all_probs', {})
        prob_vector = result.get('prob_vector', [])
        
        explanation = {
            'verifier': 'ml_classifier',
            'verdict': verdict,
            'reasoning': [],
            'evidence': [],
            'confidence_factors': {},
            'key_factors': [],
            'class_probabilities': all_probs,
            'top_predictions': []
        }
        
        # Get top 3 predictions
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        explanation['top_predictions'] = [
            {'class': cls, 'probability': prob} 
            for cls, prob in sorted_probs[:3]
        ]
        
        explanation['reasoning'].append(
            f"The ML classifier predicted '{label}' with confidence {result.get('confidence', 0):.3f}."
        )
        
        if label != 'correct':
            explanation['key_factors'].append(f"Predicted error type: {label}")
            explanation['reasoning'].append(
                f"The model identified this as a {label} based on learned patterns from training data."
            )
        else:
            explanation['key_factors'].append("No error detected by ML model")
            explanation['reasoning'].append(
                "The model found no error patterns matching the training data."
            )
        
        explanation['evidence'] = [
            f"Top prediction: {sorted_probs[0][0]} ({sorted_probs[0][1]:.3f})"
        ]
        if len(sorted_probs) > 1:
            explanation['evidence'].append(
                f"Second: {sorted_probs[1][0]} ({sorted_probs[1][1]:.3f})"
            )
        else:
            explanation['evidence'].append("Second: N/A")
        
        explanation['confidence_factors'] = {
            'prediction_confidence': result.get('confidence', 0.5),
            'class_separation': sorted_probs[0][1] - (sorted_probs[1][1] if len(sorted_probs) > 1 else 0),
            'model_certainty': max(prob_vector) if prob_vector else 0.5
        }
        
        return explanation
    
    def _explain_generic(self, step: str, problem: str, result: Dict, prev_steps: List[str]) -> Dict:
        """Generic explanation for unknown verifiers."""
        return {
            'verifier': 'unknown',
            'verdict': result.get('verdict', 'UNKNOWN'),
            'reasoning': [result.get('details', 'No explanation available')],
            'evidence': [],
            'confidence_factors': {},
            'key_factors': []
        }
    
    def explain_consensus(
        self,
        consensus: Dict,
        verifier_results: Dict[str, Dict]
    ) -> Dict:
        """Generate explanation for consensus decision.
        
        Args:
            consensus: Consensus result dict
            verifier_results: Individual verifier results
            
        Returns:
            Comprehensive explanation of consensus
        """
        verdict = consensus.get('final_verdict', 'UNKNOWN')
        agreement = consensus.get('agreement_type', 'MIXED')
        error_score = consensus.get('error_score', 0)
        breakdown = consensus.get('breakdown', {})
        
        explanation = {
            'final_verdict': verdict,
            'agreement_type': agreement,
            'reasoning': [],
            'contributing_factors': [],
            'verifier_explanations': {},
            'weighted_analysis': {}
        }
        
        # Explain agreement type
        if agreement == 'UNANIMOUS':
            explanation['reasoning'].append(
                "All verifiers agreed on the verdict, providing high confidence."
            )
        elif agreement == 'MAJORITY':
            explanation['reasoning'].append(
                "A majority of verifiers agreed on the verdict."
            )
        else:
            explanation['reasoning'].append(
                "Verifiers had mixed opinions, requiring weighted consensus."
            )
        
        # Analyze each verifier's contribution
        for name, info in breakdown.items():
            weight = info.get('weight', 0)
            contribution = info.get('contribution', 0)
            verdict_v = info.get('verdict', 'UNKNOWN')
            
            explanation['weighted_analysis'][name] = {
                'weight': weight,
                'contribution': contribution,
                'verdict': verdict_v,
                'impact': 'high' if abs(contribution) > 0.2 else 'medium' if abs(contribution) > 0.1 else 'low'
            }
            
            if abs(contribution) > 0.15:
                explanation['contributing_factors'].append(
                    f"{name} ({verdict_v}) had {'strong' if abs(contribution) > 0.2 else 'moderate'} influence"
                )
        
        # Generate per-verifier explanations
        for name, result in verifier_results.items():
            explanation['verifier_explanations'][name] = self.explain_verifier_decision(
                name, '', '', result
            )
        
        # Final reasoning
        if error_score > 0:
            explanation['reasoning'].append(
                f"The weighted error score ({error_score:.3f}) indicates an error was detected."
            )
        else:
            explanation['reasoning'].append(
                f"The weighted error score ({error_score:.3f}) indicates no significant error."
            )
        
        return explanation

