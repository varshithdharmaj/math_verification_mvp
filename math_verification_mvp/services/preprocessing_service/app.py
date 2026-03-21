from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re
import uvicorn
import cv2
import base64
import numpy as np
from image_enhancing import ImageEnhancer

app = FastAPI(title="Preprocessing Service", description="Enhances images and normalizes text for OCR.")
enhancer = ImageEnhancer(sigma=1.2)

class PreprocessRequest(BaseModel):
    text: str
    source: Optional[str] = "unknown"

class PreprocessResponse(BaseModel):
    status: str
    normalized_text: str
    tokens: List[str]

@app.get("/health")
async def health_check():
    """Returns the health status of the Preprocessing service."""
    return {"status": "healthy", "service": "preprocessing"}

def clean_ocr_errors(text: str) -> str:
    """Fix common OCR misreadings contextually."""
    # Common math OCR mistakes substitution
    replacements = {
        '×': '*', 
        '÷': '/', 
        '−': '-', 
        '–': '-', 
        ' x ': ' * ', # variable vs multiplier
        '**': '^'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    return text.strip()

def tokenize_math(text: str) -> List[str]:
    """Tokenize the mathematical expression into components."""
    # Simple regex to split numbers, variables, and operators while keeping them as tokens
    pattern = re.compile(r'(\d+\.?\d*|[a-zA-Z]+|[+\-*/=()^])')
    tokens = [t.strip() for t in pattern.findall(text) if t.strip()]
    return tokens

@app.post("/enhance_image")
async def enhance_image(file: UploadFile = File(...)):
    """
    Accepts an image file, applies MVM² noise reduction and lighting normalization,
    and returns the enhanced image as a base64 string along with quality metadata.
    """
    try:
        contents = await file.read()
        enhanced_img, metadata = enhancer.enhance(contents)
        
        # Encode back to base64 to send in JSON response
        _, buffer = cv2.imencode('.png', enhanced_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "metadata": metadata,
            "enhanced_image_base64": img_base64
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_text(request: PreprocessRequest):
    """
    Accepts raw text (from OCR or direct input), normalizes it, cleans common OCR errors, 
    and tokenizes it into mathematical components.
    This structured data will subsequently be sent to the Representation Service.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # 1. Clean OCR Errors
    cleaned_text = clean_ocr_errors(request.text)
    
    # 2. Tokenize
    tokens = tokenize_math(cleaned_text)
    
    return {
        "status": "success",
        "normalized_text": cleaned_text,
        "tokens": tokens
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
