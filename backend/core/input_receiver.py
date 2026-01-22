"""
Input Receiver Module
Handles the initial receipt and routing of verification requests.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, Dict
import base64

# Define Request Models here to avoid circular imports if simple, 
# or use a shared schemas module if this grows.
class TextInputRequest(BaseModel):
    text: str
    metadata: Optional[Dict] = {}

class ImageInputMetadata(BaseModel):
    source: str = "upload"
    dpi: int = 300

router = APIRouter()

async def receive_image(file: UploadFile, metadata: Dict) -> bytes:
    """
    Validates and accepts an uploaded image file.
    
    Args:
        file: The uploaded image file.
        metadata: Additional info about the image.
        
    Returns:
        bytes: The raw file content.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format")
    content = await file.read()
    return content

async def receive_text(request: TextInputRequest) -> str:
    """
    Validates text input.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return request.text
