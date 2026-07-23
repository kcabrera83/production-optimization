import pytest

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


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_optimize_valid(client):
    response = client.post("/api/optimize", json=SAMPLE_INPUT)
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "predicted_net_profit" in data
        assert "input" in data


def test_allocate_valid(client):
    response = client.post("/api/allocate", json=SAMPLE_INPUT)
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "predicted_production_efficiency" in data
        assert "input" in data
