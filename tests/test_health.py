import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "code-analyzer-backend"


def test_health_check_returns_json():
    """Test that health check returns JSON content type."""
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]
