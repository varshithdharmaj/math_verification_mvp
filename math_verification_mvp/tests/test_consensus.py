"""
Unit tests for Consensus Mechanism
"""

import pytest
from consensus.consensus_mechanism import compute_neurosymbolic_consensus

def test_unanimous_valid():
    """Test all agents agreeing with valid steps"""
    agent_results = {
        "Agent1": {
            "reasoning_trace": ["1 + 1 = 2", "2 * 2 = 4"],
            "confidence_explanation": "I am confident.",
            "final_answer": "4"
        },
        "Agent2": {
            "reasoning_trace": ["3 - 1 = 2", "1 + 1 = 2", "2 * 2 = 4"],
            "confidence_explanation": "Looks good.",
            "final_answer": "4"
        }
    }
    
    consensus = compute_neurosymbolic_consensus(agent_results)
    
    assert consensus["final_verdict"] == "VALID"
    assert "Agent1" in consensus["divergence_scores"]
    assert consensus["chosen_answer"] == "4"


def test_error_hallucination():
    """Test hallucination detection and ERROR verdict"""
    agent_results = {
        "Agent1": {
            "reasoning_trace": ["x = 2", "x = 3", "x = 4", "x = 5", "x = 6"], # long trace
            "confidence_explanation": "Possible hallucination or error here.",
            "final_answer": "5"
        },
        "Agent2": {
            "reasoning_trace": ["x = 2"], # short trace
            "confidence_explanation": "Confident.",
            "final_answer": "2"
        }
    }
    
    consensus = compute_neurosymbolic_consensus(agent_results)
    
    assert consensus["final_verdict"] == "ERROR"
    assert len(consensus["hallucination_alerts"]) > 0


def test_empty_results():
    """Test with minimal/empty steps"""
    agent_results = {
        "Agent1": {
            "reasoning_trace": [],
            "confidence_explanation": "guess",
            "final_answer": "0"
        }
    }
    
    consensus = compute_neurosymbolic_consensus(agent_results)
    
    # Empty steps give 0.0 for symbolic, "guess" gives 0.3 for logical. 
    # Weighted score will be low.
    assert consensus["final_verdict"] == "ERROR"
    assert consensus["chosen_answer"] == "0"
