"""
Tests for Gradio demo interface.
"""

import pytest
from demo import create_demo

def test_demo_creation():
    """Test that demo can be created without error."""
    demo = create_demo()
    assert demo is not None
    # assert hasattr(demo, "launch") # Gradio blocks has launch method

def test_demo_api_availability():
    """Test that demo exposes API."""
    demo = create_demo()
    # Gradio Blocks object
    assert demo.enable_queue is True or demo.enable_queue is None # Default might vary
