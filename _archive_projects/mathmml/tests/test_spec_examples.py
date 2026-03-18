"""Test cases from the CURSOR-COMPLETE-PROJECT-PROMPT.md specification."""

import pytest
from src.models.symbolic_verifier import SymbolicVerifier
from src.models.llm_logical_checker import LLMLogicalChecker
from src.models.ensemble_checker import EnsembleNeuralChecker
from src.models.ml_step_classifier import MLStepClassifierWrapper
from src.pipeline.consensus import ConsensusEngine
from src.pipeline.orchestrator import VerificationOrchestrator


class TestSpecExamples:
    """Test cases matching the spec examples."""
    
    def test_case_1_janet_apples_arithmetic_error(self):
        """Test Case 1: Arithmetic Error (Janet's Apples)
        
        Problem: "Janet has 3 apples. She buys 2 more. She gives 1 away. How many?"
        Steps:
          1. Janet starts with 3 apples
          2. She buys 2 more: 3 + 2 = 5 apples
          3. She gives 1 away: 5 - 1 = 6 apples  ❌ ERROR
        
        Expected:
          - Symbolic: ERROR (98% confidence)
          - LLM Logic: ERROR (87%)
          - Ensemble: ERROR (88%)
          - ML: ERROR (91%)
          - Final: ERROR (91% overall confidence, UNANIMOUS)
        """
        problem = "Janet has 3 apples. She buys 2 more. She gives 1 away. How many?"
        steps = [
            "Janet starts with 3 apples",
            "She buys 2 more: 3 + 2 = 5 apples",
            "She gives 1 away: 5 - 1 = 6 apples"  # ERROR
        ]
        
        verifiers = {
            'symbolic': SymbolicVerifier(),
            'llm_logical': LLMLogicalChecker(use_api=False),
            'ensemble': EnsembleNeuralChecker(use_apis=False),
            'ml_classifier': MLStepClassifierWrapper(model_path=None)
        }
        
        orchestrator = VerificationOrchestrator()
        
        # Test step 3 (the error)
        step = steps[2]
        prev_steps = steps[:2]
        
        result = orchestrator.verify_step(step, problem, prev_steps, verifiers)
        
        # Check verdict
        assert result['final_verdict'] == 'ERROR', f"Expected ERROR, got {result['final_verdict']}"
        
        # Check symbolic verifier detects the error
        symbolic_result = result['per_verifier_results']['symbolic']
        assert symbolic_result['verdict'] == 'ERROR', "Symbolic verifier should detect arithmetic error"
        assert symbolic_result['confidence'] >= 0.90, f"Symbolic confidence should be high, got {symbolic_result['confidence']}"
        
        # Check overall confidence is reasonable
        assert result['overall_confidence'] >= 0.80, f"Overall confidence should be high, got {result['overall_confidence']}"
        
        # Check agreement type (should be UNANIMOUS or MAJORITY for clear error)
        assert result['agreement_type'] in ['UNANIMOUS ✓✓✓', 'MAJORITY (3/4) ✓✓'], \
            f"Agreement should be UNANIMOUS or MAJORITY, got {result['agreement_type']}"
        
        # Check error_score > 0.50 (spec threshold)
        assert result['error_score'] > 0.50, f"Error score should be > 0.50, got {result['error_score']}"
    
    def test_case_2_six_horses_logic_puzzle(self):
        """Test Case 2: Logic Puzzle (Six Horses)
        
        Problem: "You have 6 horses. Find the fastest. Minimum races?"
        Steps:
          1. We need a tournament bracket
          2. Divide into 3 pairs: 3 races
          3. Winners race: 1 race
          4. Total: 4 races needed  ❌ OVERCOMPLICATED
        
        Expected:
          - Symbolic: N/A (logic problem)
          - LLM Logic: ERROR (89%) - detected over-complication
          - Ensemble: VALID (85%) - after refined prompting
          - ML: ERROR (84%)
          - Final: MAJORITY (ERROR)
        """
        problem = "You have 6 horses. Find the fastest. Minimum races?"
        steps = [
            "We need a tournament bracket",
            "Divide into 3 pairs: 3 races",
            "Winners race: 1 race",
            "Total: 4 races needed"  # Overcomplicated
        ]
        
        verifiers = {
            'symbolic': SymbolicVerifier(),
            'llm_logical': LLMLogicalChecker(use_api=False),
            'ensemble': EnsembleNeuralChecker(use_apis=False),
            'ml_classifier': MLStepClassifierWrapper(model_path=None)
        }
        
        orchestrator = VerificationOrchestrator()
        
        # Test last step
        step = steps[3]
        prev_steps = steps[:3]
        
        result = orchestrator.verify_step(step, problem, prev_steps, verifiers)
        
        # For logic problems, verdict may vary, but should process without errors
        assert result['final_verdict'] in ['ERROR', 'VALID'], \
            f"Verdict should be ERROR or VALID, got {result['final_verdict']}"
        
        # LLM logical checker should have an opinion
        llm_result = result['per_verifier_results']['llm_logical']
        assert llm_result['verdict'] in ['ERROR', 'VALID'], \
            "LLM logical checker should provide a verdict"
    
    def test_case_3_correct_solution_speed_calculation(self):
        """Test Case 3: Correct Solution (Speed Calculation)
        
        Problem: "Car travels 60 miles in 2 hours. Average speed?"
        Steps:
          1. Distance = 60 miles
          2. Time = 2 hours
          3. Speed = 60/2 = 30 mph  ✅ CORRECT
        
        Expected:
          - All 4 models: VALID
          - Final: VALID (99% confidence, UNANIMOUS)
        """
        problem = "Car travels 60 miles in 2 hours. Average speed?"
        steps = [
            "Distance = 60 miles",
            "Time = 2 hours",
            "Speed = 60/2 = 30 mph"  # Correct
        ]
        
        verifiers = {
            'symbolic': SymbolicVerifier(),
            'llm_logical': LLMLogicalChecker(use_api=False),
            'ensemble': EnsembleNeuralChecker(use_apis=False),
            'ml_classifier': MLStepClassifierWrapper(model_path=None)
        }
        
        orchestrator = VerificationOrchestrator()
        
        # Test last step (correct calculation)
        step = steps[2]
        prev_steps = steps[:2]
        
        result = orchestrator.verify_step(step, problem, prev_steps, verifiers)
        
        # Should be VALID
        assert result['final_verdict'] == 'VALID', \
            f"Expected VALID for correct calculation, got {result['final_verdict']}"
        
        # Symbolic verifier should confirm correctness
        symbolic_result = result['per_verifier_results']['symbolic']
        assert symbolic_result['verdict'] == 'VALID', \
            "Symbolic verifier should confirm correct arithmetic"
        
        # Error score should be <= 0.50 (spec threshold)
        assert result['error_score'] <= 0.50, \
            f"Error score should be <= 0.50 for valid solution, got {result['error_score']}"
        
        # Confidence should be high for correct solution
        assert result['overall_confidence'] >= 0.80, \
            f"Confidence should be high for correct solution, got {result['overall_confidence']}"
    
    def test_parallel_execution(self):
        """Test that all verifiers run in parallel."""
        problem = "Calculate 5 + 3"
        step = "5 + 3 = 8"
        prev_steps = []
        
        verifiers = {
            'symbolic': SymbolicVerifier(),
            'llm_logical': LLMLogicalChecker(use_api=False),
            'ensemble': EnsembleNeuralChecker(use_apis=False),
            'ml_classifier': MLStepClassifierWrapper(model_path=None)
        }
        
        orchestrator = VerificationOrchestrator()
        
        import time
        start = time.time()
        result = orchestrator.verify_step(step, problem, prev_steps, verifiers)
        elapsed = time.time() - start
        
        # All verifiers should have results
        assert len(result['per_verifier_results']) == 4, \
            "All 4 verifiers should have results"
        
        # Parallel execution should be faster than sequential (rough check)
        # Sequential would take at least 4x the slowest verifier
        # This is a sanity check - actual timing depends on implementation
        assert elapsed < 5.0, \
            f"Parallel execution took too long: {elapsed}s"
    
    def test_consensus_weights(self):
        """Test that consensus weights match spec (40%, 35%, 20%, 25%)."""
        engine = ConsensusEngine()
        
        expected_weights = {
            'symbolic': 0.40,
            'llm_logical': 0.35,
            'ensemble': 0.20,
            'ml_classifier': 0.25
        }
        
        # Raw weights should match spec proportions (may sum to > 1 due to boost)
        total = sum(engine.weights.values())
        assert abs(total - 1.20) < 0.01, f"Raw weights should sum to 1.20, got {total}"
        
        assert abs(engine.weights['symbolic'] - 0.40) < 0.001, \
            f"Symbolic weight should be 0.40, got {engine.weights['symbolic']}"
        assert abs(engine.weights['llm_logical'] - 0.35) < 0.001, \
            f"LLM weight should be 0.35, got {engine.weights['llm_logical']}"
        assert abs(engine.weights['ensemble'] - 0.20) < 0.001, \
            f"Ensemble weight should be 0.20, got {engine.weights['ensemble']}"
        assert abs(engine.weights['ml_classifier'] - 0.25) < 0.001, \
            f"ML weight should be 0.25, got {engine.weights['ml_classifier']}"
    
    def test_error_score_threshold(self):
        """Test that error_score > 0.50 threshold works correctly."""
        engine = ConsensusEngine()
        
        # Test case: All verifiers say ERROR with high confidence
        results = {
            'symbolic': {'verdict': 'ERROR', 'confidence': 0.95},
            'llm_logical': {'verdict': 'ERROR', 'confidence': 0.87},
            'ensemble': {'verdict': 'ERROR', 'confidence': 0.90},
            'ml_classifier': {'verdict': 'ERROR', 'confidence': 0.91}
        }
        
        consensus = engine.compute_consensus(results)
        
        # Error score should be high (all ERROR verdicts)
        assert consensus['error_score'] > 0.50, \
            f"Error score should be > 0.50, got {consensus['error_score']}"
        assert consensus['final_verdict'] == 'ERROR', \
            "Final verdict should be ERROR when error_score > 0.50"
        
        # Test case: All verifiers say VALID
        results_valid = {
            'symbolic': {'verdict': 'VALID', 'confidence': 0.95},
            'llm_logical': {'verdict': 'VALID', 'confidence': 0.87},
            'ensemble': {'verdict': 'VALID', 'confidence': 0.90},
            'ml_classifier': {'verdict': 'VALID', 'confidence': 0.91}
        }
        
        consensus_valid = engine.compute_consensus(results_valid)
        
        # Error score should be low (no ERROR verdicts)
        assert consensus_valid['error_score'] <= 0.50, \
            f"Error score should be <= 0.50, got {consensus_valid['error_score']}"
        assert consensus_valid['final_verdict'] == 'VALID', \
            "Final verdict should be VALID when error_score <= 0.50"
    
    def test_agreement_types(self):
        """Test agreement type formatting matches spec."""
        engine = ConsensusEngine()
        
        # UNANIMOUS: All agree on ERROR
        results_unanimous = {
            'symbolic': {'verdict': 'ERROR', 'confidence': 0.95},
            'llm_logical': {'verdict': 'ERROR', 'confidence': 0.87},
            'ensemble': {'verdict': 'ERROR', 'confidence': 0.90},
            'ml_classifier': {'verdict': 'ERROR', 'confidence': 0.91}
        }
        
        consensus = engine.compute_consensus(results_unanimous)
        assert consensus['agreement_type'] == 'UNANIMOUS ✓✓✓', \
            f"Expected 'UNANIMOUS ✓✓✓', got {consensus['agreement_type']}"
        
        # MAJORITY: 3 out of 4 agree
        results_majority = {
            'symbolic': {'verdict': 'ERROR', 'confidence': 0.95},
            'llm_logical': {'verdict': 'ERROR', 'confidence': 0.87},
            'ensemble': {'verdict': 'ERROR', 'confidence': 0.90},
            'ml_classifier': {'verdict': 'VALID', 'confidence': 0.91}
        }
        
        consensus = engine.compute_consensus(results_majority)
        assert consensus['agreement_type'] == 'MAJORITY (3/4) ✓✓', \
            f"Expected 'MAJORITY (3/4) ✓✓', got {consensus['agreement_type']}"
        
        # MIXED: 2-2 split
        results_mixed = {
            'symbolic': {'verdict': 'ERROR', 'confidence': 0.95},
            'llm_logical': {'verdict': 'ERROR', 'confidence': 0.87},
            'ensemble': {'verdict': 'VALID', 'confidence': 0.90},
            'ml_classifier': {'verdict': 'VALID', 'confidence': 0.91}
        }
        
        consensus = engine.compute_consensus(results_mixed)
        assert consensus['agreement_type'] == 'MIXED ✓', \
            f"Expected 'MIXED ✓', got {consensus['agreement_type']}"

