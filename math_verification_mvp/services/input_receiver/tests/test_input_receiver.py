from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "input-receiver"}

def test_receive_text():
    """Test the text ingestion endpoint."""
    response = client.post("/receive/text", json={"text": "2x + 4 = 10", "metadata": {"source": "test"}})
    assert response.status_code == 200
    assert response.json()["input_type"] == "text"
    assert response.json()["data"] == "2x + 4 = 10"

def test_receive_image():
    """Test the image ingestion endpoint with a dummy file."""
    # Create a dummy image file bytes
    dummy_image = b"fake_image_bytes"
    files = {"file": ("test.png", dummy_image, "image/png")}
    data = {"metadata": '{"source": "test"}'}
    
    response = client.post("/receive/image", files=files, data=data)
    assert response.status_code == 200
    assert response.json()["input_type"] == "image"
    assert response.json()["filename"] == "test.png"
