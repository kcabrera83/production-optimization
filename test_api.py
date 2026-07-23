"""Integration tests for the Production Optimization FastAPI endpoints."""

import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, ".")
from app import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "well_count": 45,
    "total_water_rate": 1200,
    "total_gas_rate": 3500,
    "water_injection_rate": 2000,
    "lift_type": "ESP",
    "operating_cost_usd": 95000,
    "revenue_per_bbl": 80,
    "artificial_lift_power_kw": 150,
}


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"


def test_models():
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert "field_optimizer" in data
    assert "allocation_model" in data
    assert "cv_r2" in data["field_optimizer"]
    assert "cv_r2" in data["allocation_model"]


def test_optimize():
    r = client.post("/api/optimize", json=SAMPLE_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_net_profit" in data
    assert isinstance(data["predicted_net_profit"], float)


def test_allocate():
    r = client.post("/api/allocate", json=SAMPLE_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_production_efficiency" in data
    assert isinstance(data["predicted_production_efficiency"], float)


def test_optimize_missing_field():
    bad = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "well_count"}
    r = client.post("/api/optimize", json=bad)
    assert r.status_code == 422


def test_allocate_missing_field():
    bad = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "lift_type"}
    r = client.post("/api/allocate", json=bad)
    assert r.status_code == 422


def test_optimize_multiple_inputs():
    payloads = [
        {**SAMPLE_PAYLOAD, "well_count": 10, "lift_type": "Rod_Pump"},
        {**SAMPLE_PAYLOAD, "well_count": 80, "lift_type": "Gas_Lift"},
        {**SAMPLE_PAYLOAD, "well_count": 60, "lift_type": "PCP"},
    ]
    for p in payloads:
        r = client.post("/api/optimize", json=p)
        assert r.status_code == 200
        assert "predicted_net_profit" in r.json()


def test_allocate_multiple_inputs():
    payloads = [
        {**SAMPLE_PAYLOAD, "well_count": 10, "lift_type": "Rod_Pump"},
        {**SAMPLE_PAYLOAD, "well_count": 80, "lift_type": "Gas_Lift"},
    ]
    for p in payloads:
        r = client.post("/api/allocate", json=p)
        assert r.status_code == 200
        assert "predicted_production_efficiency" in r.json()
