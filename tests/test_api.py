import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app

client = app.test_client()

SAMPLE_INPUT = {
    "well_count": 10,
    "total_water_rate": 5000.0,
    "total_gas_rate": 2000.0,
    "water_injection_rate": 3000.0,
    "lift_type": "ESP",
    "operating_cost_usd": 150000.0,
    "revenue_per_bbl": 75.0,
    "artificial_lift_power_kw": 200.0,
}


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "field_optimizer" in data
    assert "allocation_model" in data


def test_api_docs():
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.0"


def test_optimize_valid():
    response = client.post("/api/optimize", json=SAMPLE_INPUT)
    assert response.status_code == 200
    data = response.get_json()
    assert "predicted_net_profit" in data
    assert "input" in data


def test_optimize_missing_fields():
    response = client.post("/api/optimize", json={
        "well_count": 5,
    })
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_allocate_valid():
    response = client.post("/api/allocate", json=SAMPLE_INPUT)
    assert response.status_code == 200
    data = response.get_json()
    assert "predicted_production_efficiency" in data
    assert "input" in data
    assert 0.0 <= data["predicted_production_efficiency"] <= 1.0


def test_allocate_missing_fields():
    response = client.post("/api/allocate", json={
        "well_count": 5,
    })
    assert response.status_code == 400


def test_optimize_no_json():
    response = client.post("/api/optimize", content_type="application/json")
    assert response.status_code in [400, 415]
