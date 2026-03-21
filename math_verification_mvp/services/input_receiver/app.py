from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn

app = FastAPI(title="Input Receiver Service", description="Accepts and routes math problems in text or image format.")

class TextInputRequest(BaseModel):
    text: str
    metadata: Optional[Dict] = {}

@app.get("/health")
async def health_check():
    """Returns the health status of the Input Receiver service."""
    return {"status": "healthy", "service": "input-receiver"}

@app.post("/receive/text")
async def receive_text(request: TextInputRequest):
    """
    Accepts a math problem in text format and returns it in a standardized JSON payload.
    This payload will subsequently be sent to the Preprocessing Service.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    return {
        "status": "success",
        "input_type": "text",
        "data": request.text,
        "metadata": request.metadata
    }

@app.post("/receive/image")
async def receive_image(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form("{}")
):
    """
    Accepts a math problem as an image upload and returns a confirmation JSON payload.
    The image bytes will subsequently be sent to the OCR Service.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Use JPEG or PNG.")
    
    # In a real microservice, we might save this to an S3 bucket or shared volume and pass the URL/path.
    # For MVP, we return a success format. The actual forwarding will be handled by the API Gateway.
    return {
        "status": "success",
        "input_type": "image",
        "filename": file.filename,
        "content_type": file.content_type,
        "metadata": metadata
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
