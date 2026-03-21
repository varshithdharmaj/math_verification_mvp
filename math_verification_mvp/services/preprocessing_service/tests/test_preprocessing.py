from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "preprocessing"}

def test_preprocess_text():
    """Test the text preprocessing and tokenization logic."""
    payload = {"text": "2 x 4 + y = 10", "source": "ocr"}
    response = client.post("/preprocess", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["normalized_text"] == "2 * 4 + y = 10"
    assert "tokens" in data
    assert data["tokens"] == ["2", "*", "4", "+", "y", "=", "10"]
