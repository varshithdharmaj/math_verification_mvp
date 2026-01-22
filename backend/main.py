"""
Main Backend Application
Wires together the Input Receiver -> Pipeline -> Reporting flow.
"""
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from backend.core import (
    input_receiver,
    preprocessing_service,
    ocr_service,
    representation_service,
    verification_service,
    classifier_service,
    reporting_service
)
from typing import Dict, Optional
import uvicorn
import json

app = FastAPI(title="MVM² Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev simple access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/solve/image")
async def solve_image(
    file: UploadFile = File(...),
    metadata_json: Optional[str] = Form("{}")
):
    """
    End-to-end verification for Image input.
    """
    # 1. Input Receiver
    metadata = json.loads(metadata_json)
    raw_bytes = await input_receiver.receive_image(file, metadata)
    
    # 2. Preprocessing
    processed_bytes = await preprocessing_service.preprocess_image(raw_bytes)
    
    # 3. OCR
    raw_text = await ocr_service.extract_text(processed_bytes)
    ocr_conf = await ocr_service.get_ocr_confidence(processed_bytes)
    
    # 4. Representation
    structured_data = await representation_service.normalize_input(raw_text)
    
    # 5. Verification
    verdict_details = await verification_service.verify_step_by_step(structured_data)
    
    # 6. Classification & Scoring
    score = await classifier_service.classify_and_score(verdict_details, ocr_confidence=ocr_conf)
    
    # 7. Reporting
    full_report = await reporting_service.generate_full_report(
        input_type="image",
        raw_input="[Image Blob]", 
        scoring_result=score,
        details={
            "ocr_text": raw_text, 
            "ocr_confidence": ocr_conf,
            "verification": verdict_details,
            "structure": structured_data
        }
    )
    
    return full_report

@app.post("/solve/text")
async def solve_text(request: input_receiver.TextInputRequest):
    """
    End-to-end verification for Text/LaTeX input.
    """
    # 1. Input Receiver
    text = await input_receiver.receive_text(request)
    
    # 2. Representation (Skip Preprocessing/OCR)
    structured_data = await representation_service.normalize_input(text)
    
    # 3. Verification
    verdict_details = await verification_service.verify_step_by_step(structured_data)
    
    # 4. Classification & Scoring
    score = await classifier_service.classify_and_score(verdict_details, ocr_confidence=1.0)
    
    # 5. Reporting
    full_report = await reporting_service.generate_full_report(
        input_type="text",
        raw_input=text,
        scoring_result=score,
        details={
            "verification": verdict_details,
            "structure": structured_data
        }
    )
    
    return full_report

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
