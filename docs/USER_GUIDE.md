# User Guide - Production Optimization

## Overview

The Production Optimization system uses machine learning to predict net profit and production efficiency for oil field operations, helping optimize well allocation and resource management.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd production-optimization
pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

This generates 2,000 synthetic field operation records and trains:
- Field Optimizer (GradientBoostingRegressor) - Predicts net profit
- Allocation Model (RandomForestRegressor) - Predicts production efficiency

### Run the Server

```bash
python app.py
```

Open `http://localhost:5014` in your browser.

## Dashboard Features

- **Profit Optimization Panel** - Input field parameters to predict net profit
- **Allocation Panel** - Predict production efficiency for well allocation
- **Feature Importance Charts** - View which parameters most affect outcomes
- **Model Metrics** - Cross-validation R2 scores and performance data
- **Dark Theme UI** - Modern dark-themed dashboard

## API Usage

### Predict Net Profit (curl)

```bash
curl -X POST http://localhost:5014/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "well_count": 45,
    "total_water_rate": 1200,
    "total_gas_rate": 3500,
    "water_injection_rate": 2000,
    "lift_type": "ESP",
    "operating_cost_usd": 95000,
    "revenue_per_bbl": 80,
    "artificial_lift_power_kw": 150
  }'
```

### Predict Net Profit (Python)

```python
import requests

response = requests.post("http://localhost:5014/api/optimize", json={
    "well_count": 45,
    "total_water_rate": 1200,
    "total_gas_rate": 3500,
    "water_injection_rate": 2000,
    "lift_type": "ESP",
    "operating_cost_usd": 95000,
    "revenue_per_bbl": 80,
    "artificial_lift_power_kw": 150
})
result = response.json()
print(f"Predicted net profit: ${result['predicted_net_profit']:,.2f}")
```

### Predict Production Efficiency (curl)

```bash
curl -X POST http://localhost:5014/api/allocate \
  -H "Content-Type: application/json" \
  -d '{
    "well_count": 45,
    "total_water_rate": 1200,
    "total_gas_rate": 3500,
    "water_injection_rate": 2000,
    "lift_type": "ESP",
    "operating_cost_usd": 95000,
    "revenue_per_bbl": 80,
    "artificial_lift_power_kw": 150
  }'
```

### Predict Production Efficiency (Python)

```python
import requests

response = requests.post("http://localhost:5014/api/allocate", json={
    "well_count": 45,
    "total_water_rate": 1200,
    "total_gas_rate": 3500,
    "water_injection_rate": 2000,
    "lift_type": "ESP",
    "operating_cost_usd": 95000,
    "revenue_per_bbl": 80,
    "artificial_lift_power_kw": 150
})
result = response.json()
print(f"Production efficiency: {result['predicted_production_efficiency']:.2%}")
```

### Check Health

```bash
curl http://localhost:5014/api/health
```

### Get Model Info

```bash
curl http://localhost:5014/api/models
```

## Typical Workflow

1. Enter field parameters (well count, rates, costs, lift type)
2. Call `/api/optimize` to predict net profit
3. Call `/api/allocate` to predict production efficiency
4. Compare scenarios by varying lift_type, costs, or revenue
5. Use feature importances to identify optimization targets

## Running Tests

```bash
pytest test_api.py -v
```

## Troubleshooting

- **Models not loaded**: Run `python train.py` first
- **Missing fields**: All 8 fields are required
- **Invalid lift_type**: Must be one of ESP, rod_pump, gas_lift, progressive_cavity
- **Port in use**: Change port in `app.py`

---

*Elaborado por Ing. Kelvin Cabrera*
