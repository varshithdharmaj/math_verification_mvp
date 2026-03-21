from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "representation"}

def test_represent_expression():
    """Test SymPy parsing of a simple expression."""
    payload = {"normalized_text": "2*x + 4"}
    response = client.post("/represent", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "2*x + 4" in data["symbolic_expr"]
    assert data["is_equation"] is False

def test_represent_equation():
    """Test SymPy parsing of an equation."""
    payload = {"normalized_text": "2*x + 4 = 10"}
    response = client.post("/represent", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Eq" in data["symbolic_expr"]
    assert data["is_equation"] is True

def test_represent_fallback():
    """Test fallback logic when SymPy fails to parse garbage."""
    payload = {"normalized_text": "2 * * x + +"}
    response = client.post("/represent", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "partial_success"
    assert data["symbolic_expr"] == "2 * * x + +"
