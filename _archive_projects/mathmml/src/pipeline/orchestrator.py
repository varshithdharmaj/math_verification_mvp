"""Pipeline orchestrator for parallel verification execution."""

import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.pipeline.consensus import ConsensusEngine


class VerificationOrchestrator:
    """Orchestrates parallel execution of all verifiers."""
    
    def __init__(
        self,
        consensus_engine: Optional[ConsensusEngine] = None,
        max_workers: int = 4
    ):
        """Initialize orchestrator.
        
        Args:
            consensus_engine: Consensus engine instance (creates new if None)
            max_workers: Max parallel workers (default: 4 for 4 verifiers)
        """
        self.consensus_engine = consensus_engine or ConsensusEngine()
        self.max_workers = max_workers
    
    def run_verification_parallel(
        self,
        problem: str,
        steps: List[str],
        verifiers: Dict[str, any],
        callback: Optional[callable] = None
    ) -> Dict:
        """Run all verifiers in parallel and compute consensus.
        
        Args:
            problem: Problem statement
            steps: List of solution steps
            verifiers: Dict mapping names to verifier instances
            callback: Optional callback function(status, result) for progress updates
        
        Returns:
            Full verification result with:
            - problem: Original problem
            - steps: Original steps
            - model_results: Per-verifier results
            - consensus: Consensus result
            - processing_time: Total time in seconds
        """
        start_time = time.time()
        
        # Track previous steps for context
        prev_steps = []
        all_step_results = []
        
        # Process each step
        for step_idx, step in enumerate(steps):
            if callback:
                callback('status', f'Processing step {step_idx + 1}/{len(steps)}')
            
            # Run all verifiers in parallel for this step
            step_results = self._run_verifiers_parallel(
                step, problem, prev_steps, verifiers, callback
            )
            
            # Compute consensus for this step
            consensus = self.consensus_engine.compute_consensus(step_results)
            consensus['per_verifier_results'] = step_results
            
            all_step_results.append({
                'step_number': step_idx + 1,
                'step_text': step,
                'consensus': consensus,
                'verifier_results': step_results
            })
            
            # Update previous steps for next iteration
            prev_steps.append(step)
        
        # Aggregate results across all steps
        processing_time = time.time() - start_time
        
        # Determine overall verdict (ERROR if any step has error)
        overall_verdict = 'VALID'
        overall_confidence = 0.0
        all_errors = []
        
        for step_result in all_step_results:
            consensus = step_result['consensus']
            if consensus['final_verdict'] == 'ERROR':
                overall_verdict = 'ERROR'
            all_errors.extend([
                {
                    **error,
                    'step_number': step_result['step_number']
                }
                for verifier_result in step_result['verifier_results'].values()
                for error in verifier_result.get('errors', [])
            ])
        
        # Calculate overall confidence as average
        confidences = [
            step_result['consensus']['overall_confidence']
            for step_result in all_step_results
        ]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        if callback:
            callback('complete', {'verdict': overall_verdict, 'confidence': overall_confidence})
        
        return {
            'problem': problem,
            'steps': steps,
            'step_results': all_step_results,
            'final_verdict': overall_verdict,
            'overall_confidence': overall_confidence,
            'all_errors': all_errors,
            'processing_time': processing_time,
            'individual_verdicts': {
                name: 'ERROR' if any(
                    step_result['verifier_results'].get(name, {}).get('verdict') == 'ERROR'
                    for step_result in all_step_results
                ) else 'VALID'
                for name in verifiers.keys()
            },
            'individual_confidences': {
                name: sum(
                    step_result['verifier_results'].get(name, {}).get('confidence', 0.0)
                    for step_result in all_step_results
                ) / len(all_step_results) if all_step_results else 0.0
                for name in verifiers.keys()
            }
        }
    
    def _run_verifiers_parallel(
        self,
        step: str,
        problem: str,
        prev_steps: List[str],
        verifiers: Dict[str, any],
        callback: Optional[callable] = None
    ) -> Dict[str, Dict]:
        """Run all verifiers in parallel for a single step.
        
        Args:
            step: Step text to verify
            problem: Problem statement
            prev_steps: Previous steps
            verifiers: Dict mapping names to verifier instances
            callback: Optional callback for progress
        
        Returns:
            Dict mapping verifier names to their results
        """
        results = {}
        
        def run_verifier(name: str, verifier: any) -> tuple:
            """Run a single verifier and return (name, result)."""
            try:
                if callback:
                    callback('verifier_start', name)
                result = verifier.verify(step, problem, prev_steps)
                if callback:
                    callback('verifier_complete', name)
                return (name, result)
            except Exception as e:
                if callback:
                    callback('verifier_error', {'name': name, 'error': str(e)})
                return (name, {
                    'verdict': 'UNKNOWN',
                    'confidence': 0.0,
                    'error_type': None,
                    'details': f'Error: {str(e)}',
                    'errors': []
                })
        
        # Execute all verifiers in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(run_verifier, name, verifier): name
                for name, verifier in verifiers.items()
            }
            
            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result
        
        return results
    
    def verify_step(
        self,
        step: str,
        problem: str,
        prev_steps: List[str],
        verifiers: Dict[str, any]
    ) -> Dict:
        """Verify a single step using all verifiers in parallel.
        
        Args:
            step: Step text to verify
            problem: Problem statement
            prev_steps: Previous steps
            verifiers: Dict mapping names to verifier instances
        
        Returns:
            Consensus result with per-verifier results
        """
        step_results = self._run_verifiers_parallel(step, problem, prev_steps, verifiers)
        consensus = self.consensus_engine.compute_consensus(step_results)
        consensus['per_verifier_results'] = step_results
        return consensus

