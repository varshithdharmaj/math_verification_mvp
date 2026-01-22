"""
MVM² Configuration Module
Centralizes environment variables, paths, and constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Paths
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = DATASETS_DIR / "results"
MODELS_DIR = BASE_DIR / "backend" / "models"

# Service Configuration
HANDWRITTEN_OCR_ENABLED = True
SYMPY_TIMEOUT_SECONDS = 5.0
OCR_CONFIDENCE_THRESHOLD = 0.85

# Weights (MVM² Formula)
WEIGHT_SYMBOLIC = 0.40
WEIGHT_LOGICAL = 0.35
WEIGHT_CLASSIFIER = 0.25
