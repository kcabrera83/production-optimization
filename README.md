# Production Optimization ML System

Machine learning system for oil field production optimization and well allocation.

## Overview

This project provides two ML models:

- **Field Optimizer** (GradientBoostingRegressor) - Predicts net profit based on field production parameters
- **Allocation Model** (RandomForestRegressor) - Predicts production efficiency for well allocation decisions

## Directory Structure

```
production-optimization/
  production_optimization/
    __init__.py
    data_generator.py
    models/
      __init__.py
      field_optimizer.py
      allocation_model.py
    utils/
      __init__.py
      preprocessor.py
  outputs/models/
  templates/
  .github/workflows/
  train.py
  app.py
  test_api.py
  requirements.txt
  setup.py
```

## Setup

```bash
pip install -r requirements.txt
python train.py
python app.py
```

The Flask application runs on port 5014.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/api/health` | GET | Health check |
| `/api/models` | GET | Model metadata and feature importances |
| `/api/optimize` | POST | Predict net profit |
| `/api/allocate` | POST | Predict production efficiency |

## Example Request

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

## Tests

```bash
pytest test_api.py -v
```

## License

MIT

Elaborado por Ing. Kelvin Cabrera
