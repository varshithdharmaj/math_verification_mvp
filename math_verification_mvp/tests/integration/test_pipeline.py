import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from services.input_receiver.app import app as input_app
from services.representation_service.app import app as rep_app

client_input = TestClient(input_app)
client_rep = TestClient(rep_app)

def test_input_receiver_health():
    response = client_input.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_input_receiver_text():
    # In the refactored architecture, Input Receiver returns immediately for routing
    payload = {"text": "2x + 4 = 10\n2x = 6\nx = 3"}
    response = client_input.post("/receive/text", json=payload)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_representation_health():
    response = client_rep.get("/health")
    assert response.status_code == 200

def test_representation_normalization():
    payload = {"normalized_text": "2*x=4"}
    response = client_rep.post("/represent", json=payload)
    
    assert response.status_code == 200
    assert "Eq" in response.json()["symbolic_expr"]
