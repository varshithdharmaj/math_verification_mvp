from fastapi.testclient import TestClient
from main import app
from PIL import Image
import io

client = TestClient(app)

def test_health_check():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "ocr"}

def test_extract_text():
    """Test the OCR extraction endpoint with a blank dummy image."""
    # Create a dummy image using PIL
    img = Image.new('RGB', (100, 30), color = (255, 255, 255))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    
    files = {"file": ("test.png", img_bytes.getvalue(), "image/png")}
    response = client.post("/extract", files=files)
    
    assert response.status_code == 200
    assert "extracted_text" in response.json()
