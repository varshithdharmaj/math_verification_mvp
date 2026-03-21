from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
import uvicorn
from extractor import extract_math

app = FastAPI(title="OCR Service", description="MVM² Specialized OCR Service utilizing Visual LLM extraction.")

@app.get("/health")
async def health_check():
    """Returns the health status of the OCR service."""
    return {"status": "healthy", "service": "ocr", "backend": "Gemini Vision"}

@app.post("/extract")
async def extract_text(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form("{}")
):
    """
    Accepts an image upload, runs the Gemini Vision extraction and MVM² Weighted Confidence rating, 
    and returns the extracted canonical LaTeX text and confidence in JSON format.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an image.")
    
    try:
        image_bytes = await file.read()
        
        # Dispatch to Gemini 1.5 Flash Vision OCR
        result = extract_math(image_bytes)
        
        return {
            "status": "success",
            "extracted_text": result["text"],
            "confidence": result["confidence"],
            "method": result["method"],
            "metadata": metadata
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR Vision processing failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
