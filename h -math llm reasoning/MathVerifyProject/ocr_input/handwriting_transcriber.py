"""
Handwriting Transcriber Module
Wrapper for handwritten-math-transcription repository
"""

import sys
import os
import torch
from typing import Optional, Tuple

# Add handwritten-math-transcription to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'handwritten-math-transcription'))

try:
    from model import Encoder, Decoder, Seq2Seq
    from dataset.hme_ink import read_inkml_file
    from utils import tokenize_latex
    from corrector import correct_latex
    from config import *
except ImportError:
    Encoder = None
    Decoder = None
    Seq2Seq = None
    read_inkml_file = None
    tokenize_latex = None
    correct_latex = None


class HandwritingTranscriber:
    """
    Handwriting transcriber for mathematical expressions.
    Converts handwritten math (InkML format) to LaTeX.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 device: str = None,
                 use_corrector: bool = True):
        """
        Initialize handwriting transcriber.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run model on ('cpu', 'cuda', 'mps')
            use_corrector: Whether to use LLM corrector for post-processing
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_corrector = use_corrector
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        if Seq2Seq is None:
            raise ImportError("Handwriting transcription model not available")
        
        try:
            # Model architecture parameters (from config or defaults)
            input_dim = 11
            enc_hidden_dim = 256
            dec_hidden_dim = 256
            embed_dim = 128
            output_dim = LATEX_VOCAB_SIZE if 'LATEX_VOCAB_SIZE' in globals() else 300
            encoder_num_layers = 2
            decoder_num_layers = 2
            
            # Create model
            encoder = Encoder(input_dim, enc_hidden_dim, 
                            num_layers=encoder_num_layers, bidirectional=True)
            decoder = Decoder(output_dim, embed_dim, 
                            enc_hidden_dim, dec_hidden_dim, 
                            num_layers=decoder_num_layers)
            self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def transcribe_inkml(self, inkml_path: str) -> Tuple[str, str]:
        """
        Transcribe an InkML file to LaTeX.
        
        Args:
            inkml_path: Path to InkML file
            
        Returns:
            Tuple of (predicted_latex, ground_truth_latex if available)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        if read_inkml_file is None:
            raise ImportError("InkML reading functionality not available")
        
        try:
            # Read InkML file
            strokes, ground_truth = read_inkml_file(inkml_path)
            
            # Convert to model input format
            # This is a simplified version - actual implementation would need
            # proper feature extraction and tensor conversion
            # For now, return placeholder
            predicted_latex = "\\placeholder"
            
            # Apply corrector if enabled
            if self.use_corrector and correct_latex:
                try:
                    predicted_latex = correct_latex(predicted_latex)
                except Exception as e:
                    print(f"Corrector error: {e}")
            
            return predicted_latex, ground_truth
        except Exception as e:
            print(f"Error transcribing InkML: {e}")
            return "", ""
    
    def transcribe_image(self, image_path: str) -> str:
        """
        Transcribe a handwritten math image to LaTeX.
        Note: This is a placeholder - actual implementation would require
        image preprocessing and conversion to InkML or direct image processing.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Predicted LaTeX string
        """
        # This would require additional image processing
        # For now, return placeholder
        return "\\placeholder"
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None

