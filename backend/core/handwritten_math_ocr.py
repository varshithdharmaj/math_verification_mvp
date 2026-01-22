"""
Handwritten Math OCR Wrapper
Integrates johnkimdw/handwritten-math-transcription model
"""
import sys
import os
import torch
import numpy as np
from PIL import Image
from typing import Dict

# Add handwritten-math-transcription to path
# Since we are in backend/, we go up one level to root, then to handwritten-math-transcription
HMT_PATH = os.path.join(os.path.dirname(__file__), "..", "handwritten-math-transcription")
sys.path.insert(0, HMT_PATH)

# Import config values directly to avoid path issues
try:
    # Try to import from the repository
    from config import LATEX_VOCAB, LATEX_VOCAB_REVERSE, DEVICE
    from model import Encoder, Decoder, Seq2Seq
    from dataset.hme_dataset import HMEDataset
    HMT_AVAILABLE = True
    print("[OK] Handwritten math transcription imports successful")
except ImportError as e:
    print(f"[WARN] Handwritten math transcription model not available: {e}")
    HMT_AVAILABLE = False
    LATEX_VOCAB = {}
    LATEX_VOCAB_REVERSE = {}
    DEVICE = None


class HandwrittenMathOCR:
    """
    Wrapper for handwritten math OCR model
    Converts images to LaTeX using seq2seq model
    """
    
    def __init__(self):
        self.model = None
        self.device = None
        self.model_loaded = False
        
    def load_model(self):
        """Lazy load the model"""
        if self.model_loaded or not HMT_AVAILABLE:
            return
            
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Model architecture (from main.py)
            input_dim = 11  # Feature dimension from InkML
            encoder = Encoder(input_dim=input_dim)
            decoder = Decoder(
                output_dim=len(LATEX_VOCAB),
                embed_dim=64,
                encoder_hidden_dim=128,
                decoder_hidden_dim=128
            )
            self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)
            
            # Try multiple model paths (in order of preference)
            model_paths = [
                os.path.join(HMT_PATH, "model", "model_best_crc_full_data.pth"),  # Best: trained on full data
                os.path.join(HMT_PATH, "model_v3_0.pth"),  # Alternative
                os.path.join(HMT_PATH, "model_v3_1.pth"),  # Alternative
            ]
            
            model_loaded_from = None
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                        self.model.eval()
                        self.model_loaded = True
                        model_loaded_from = model_path
                        print(f"[OK] Handwritten Math OCR model loaded from {os.path.basename(model_path)}")
                        break
                    except Exception as e:
                        print(f"[WARN] Failed to load {os.path.basename(model_path)}: {e}")
                        continue
            
            if not self.model_loaded:
                print(f"[ERROR] No valid pretrained model found in {HMT_PATH}")
                
        except Exception as e:
            print(f"[ERROR] Failed to load handwritten math OCR model: {e}")
            self.model_loaded = False
    
    def image_to_features(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL image to feature tensor using stroke extraction
        """
        try:
            # Import stroke extraction module
            # UPDATED: Use relative import since both are in backend package
            from .stroke_extraction import extract_features_from_image
            
            # Extract strokes and convert to features
            features = extract_features_from_image(image)
            
            print(f"[INFO] Extracted {features.shape[0]} feature points from image")
            return features
            
        except Exception as e:
            print(f"[WARN] Stroke extraction failed: {e}, using fallback")
            # Fallback to simplified approach if stroke extraction fails
            return self._simple_image_to_features(image)
    
    def _simple_image_to_features(self, image: Image.Image) -> torch.Tensor:
        """
        Simplified fallback: Convert image to basic features
        """
        # Convert to grayscale
        img_gray = image.convert('L')
        img_array = np.array(img_gray)
        
        # Sample points from the image
        height, width = img_array.shape
        num_points = min(100, height * width // 100)
        
        features = torch.zeros(num_points, 11)
        
        # Fill with basic image statistics
        for i in range(num_points):
            y = (i * height) // num_points
            x = (i * width) // num_points
            if y < height and x < width:
                pixel_val = img_array[y, x] / 255.0
                features[i, 0] = x / width  # Normalized x
                features[i, 1] = y / height  # Normalized y
                features[i, 2] = pixel_val  # Intensity
        
        return features
    
    def transcribe(self, image: Image.Image) -> Dict:
        """
        Transcribe handwritten math from image to LaTeX
        """
        self.load_model()
        
        if not self.model_loaded:
            return {
                'latex': '',
                'confidence': 0.0,
                'method': 'Handwritten Math OCR (unavailable)',
                'error': 'Model not loaded'
            }
        
        try:
            # Convert image to features
            features = self.image_to_features(image)
            
            if features.size(0) == 0:
                return {
                    'latex': '',
                    'confidence': 0.0,
                    'method': 'Handwritten Math OCR',
                    'error': 'No features extracted'
                }
            
            # Prepare for inference
            src = features.unsqueeze(0).to(self.device)
            lengths = torch.tensor([features.size(0)]).to(self.device)
            
            with torch.no_grad():
                # Encode
                enc_out, (h, c) = self.model.encoder(src, lengths)
                if self.model.encoder.bidirectional:
                    h = h.view(self.model.encoder.num_layers, 2, 1, -1).sum(dim=1)
                    c = c.view(self.model.encoder.num_layers, 2, 1, -1).sum(dim=1)
                mask = self.model.create_mask(src)
                
                # Decode
                token = torch.tensor([LATEX_VOCAB['<sos>']]).to(self.device)
                out_idx = [LATEX_VOCAB['<sos>']]
                max_length = 150
                
                for _ in range(max_length):
                    logits, h, c, _ = self.model.decoder(token, h, c, enc_out, mask)
                    top = logits.argmax(1).item()
                    out_idx.append(top)
                    if top == LATEX_VOCAB['<eos>']:
                        break
                    token = torch.tensor([top]).to(self.device)
            
            # Convert indices to LaTeX
            latex = self._indices_to_latex(out_idx)
            
            # Estimate confidence (simplified)
            confidence = 0.85  # Placeholder - would need proper confidence estimation
            
            return {
                'latex': latex,
                'confidence': confidence,
                'method': 'Handwritten Math OCR (Seq2Seq)'
            }
            
        except Exception as e:
            return {
                'latex': '',
                'confidence': 0.0,
                'method': 'Handwritten Math OCR',
                'error': str(e)
            }
    
    def _indices_to_latex(self, indices):
        """Convert token indices to LaTeX string"""
        if not HMT_AVAILABLE:
            return ""
        
        tokens = []
        for idx in indices:
            if idx in LATEX_VOCAB_REVERSE and idx not in [
                LATEX_VOCAB['<pad>'],
                LATEX_VOCAB['<sos>'],
                LATEX_VOCAB['<eos>']
            ]:
                tokens.append(LATEX_VOCAB_REVERSE[idx])
        return ''.join(tokens)
