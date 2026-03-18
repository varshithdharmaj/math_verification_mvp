"""
Main Interface Module
Provides CLI and Gradio interface for the integrated system
"""

from main_interface.cli import MathVerifyCLI

try:
    from main_interface.gradio_app import create_gradio_app
    __all__ = ['MathVerifyCLI', 'create_gradio_app']
except ImportError:
    __all__ = ['MathVerifyCLI']
    def create_gradio_app():
        raise ImportError("Gradio is not installed. Please install it with: pip install gradio")

