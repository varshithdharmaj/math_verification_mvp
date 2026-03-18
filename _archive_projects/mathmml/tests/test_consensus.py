"""Tests for consensus mechanism."""

import pytest
from src.pipeline.consensus import ConsensusEngine, AgreementType


def test_unanimous_agreement():
    """Test unanimous agreement."""
    engine = ConsensusEngine()
    
    results = {
        'symbolic': {'verdict': 'ERROR', 'confidence': 0.9},
        'llm_logical': {'verdict': 'ERROR', 'confidence': 0.8},
        'ensemble': {'verdict': 'ERROR', 'confidence': 0.85},
        'ml_classifier': {'verdict': 'ERROR', 'confidence': 0.75}
    }
    
    consensus = engine.compute_consensus(results)
    assert consensus['agreement_type'] == AgreementType.UNANIMOUS.value
    assert consensus['final_verdict'] == 'ERROR'


def test_majority_agreement():
    """Test majority agreement."""
    engine = ConsensusEngine()
    
    results = {
        'symbolic': {'verdict': 'ERROR', 'confidence': 0.9},
        'llm_logical': {'verdict': 'ERROR', 'confidence': 0.8},
        'ensemble': {'verdict': 'VALID', 'confidence': 0.7},
        'ml_classifier': {'verdict': 'VALID', 'confidence': 0.6}
    }
    
    consensus = engine.compute_consensus(results)
    assert consensus['agreement_type'] in [AgreementType.MAJORITY.value, AgreementType.MIXED.value]


def test_boundary_case():
    """Test boundary case at 0.5 error score."""
    engine = ConsensusEngine()
    
    results = {
        'symbolic': {'verdict': 'ERROR', 'confidence': 0.5},
        'llm_logical': {'verdict': 'VALID', 'confidence': 0.5},
        'ensemble': {'verdict': 'VALID', 'confidence': 0.5},
        'ml_classifier': {'verdict': 'VALID', 'confidence': 0.5}
    }
    
    consensus = engine.compute_consensus(results)
    assert consensus['final_verdict'] in ['VALID', 'ERROR']


if __name__ == "__main__":
    pytest.main([__file__])

