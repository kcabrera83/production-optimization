# API Documentation - Production Optimization

## Base URL

```
http://localhost:5014
```

## Endpoints

### GET /

Serve the main web dashboard UI.

**Response:** HTML page

---

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

---

### GET /api/models

Return model metadata, cross-validation scores, and feature importances.

**Response:**
```json
{
  "field_optimizer": {
    "algorithm": "GradientBoostingRegressor",
    "cv_r2": 0.8856,
    "feature_importances": {
      "revenue_per_bbl": 0.25,
      "total_gas_rate": 0.20,
      "operating_cost_usd": 0.18,
      "well_count": 0.12,
      "..."
    }
  },
  "allocation_model": {
    "algorithm": "RandomForestRegressor",
    "cv_r2": 0.8623,
    "feature_importances": {
      "total_gas_rate": 0.22,
      "water_injection_rate": 0.19,
      "artificial_lift_power_kw": 0.15,
      "..."
    }
  }
}
```

---

### POST /api/optimize

Predict net profit for field operations.

**Request:**
```json
{
  "well_count": 45,
  "total_water_rate": 1200,
  "total_gas_rate": 3500,
  "water_injection_rate": 2000,
  "lift_type": "ESP",
  "operating_cost_usd": 95000,
  "revenue_per_bbl": 80,
  "artificial_lift_power_kw": 150
}
```

**Required Fields:**

| Field | Type | Description |
|-------|------|-------------|
| well_count | int | Number of wells in field |
| total_water_rate | float | Total water production rate (bbl/day) |
| total_gas_rate | float | Total gas production rate (MCF/day) |
| water_injection_rate | float | Water injection rate (bbl/day) |
| lift_type | string | Artificial lift type |
| operating_cost_usd | float | Total operating cost (USD) |
| revenue_per_bbl | float | Revenue per barrel of oil (USD) |
| artificial_lift_power_kw | float | Artificial lift power consumption (kW) |

**lift_type Options:**
- ESP (Electric Submersible Pump)
- rod_pump (Rod Pump / Beam Pump)
- gas_lift (Gas Lift)
- progressive_cavity (Progressive Cavity Pump)

**Response:**
```json
{
  "predicted_net_profit": 2450000.75,
  "input": {
    "well_count": 45,
    "total_water_rate": 1200,
    "total_gas_rate": 3500,
    "water_injection_rate": 2000,
    "lift_type": "ESP",
    "operating_cost_usd": 95000,
    "revenue_per_bbl": 80,
    "artificial_lift_power_kw": 150
  }
}
```

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | No JSON body | `{"error": "No JSON body provided"}` |
| 400 | Missing fields | `{"error": "Missing fields: [...]"}"}` |
| 500 | Models not loaded | `{"error": "Models not loaded"}` |

---

### POST /api/allocate

Predict production efficiency for well allocation decisions.

**Request:**
```json
{
  "well_count": 45,
  "total_water_rate": 1200,
  "total_gas_rate": 3500,
  "water_injection_rate": 2000,
  "lift_type": "ESP",
  "operating_cost_usd": 95000,
  "revenue_per_bbl": 80,
  "artificial_lift_power_kw": 150
}
```

**Response:**
```json
{
  "predicted_production_efficiency": 0.8234,
  "input": {
    "well_count": 45,
    "total_water_rate": 1200,
    "total_gas_rate": 3500,
    "water_injection_rate": 2000,
    "lift_type": "ESP",
    "operating_cost_usd": 95000,
    "revenue_per_bbl": 80,
    "artificial_lift_power_kw": 150
  }
}
```

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | No JSON body | `{"error": "No JSON body provided"}` |
| 400 | Missing fields | `{"error": "Missing fields: [...]"}"}` |
| 500 | Models not loaded | `{"error": "Models not loaded"}` |

---

### GET /api/docs

Return OpenAPI 3.0 specification.

---

## Feature Reference

| Feature | Type | Description |
|---------|------|-------------|
| well_count | int | Number of wells in the field |
| total_water_rate | float | Total water production rate (bbl/day) |
| total_gas_rate | float | Total gas production rate (MCF/day) |
| water_injection_rate | float | Water injection rate (bbl/day) |
| lift_type | string | Artificial lift type (categorical) |
| operating_cost_usd | float | Total operating cost (USD) |
| revenue_per_bbl | float | Revenue per barrel of oil (USD) |
| artificial_lift_power_kw | float | Artificial lift power (kW) |

## Preprocessing

- Numeric features: StandardScaler normalization
- Categorical feature (lift_type): OneHotEncoder encoding
- Separate preprocessors for optimize and allocate models

## Error Codes

- **200**: Success
- **400**: Bad request (missing or invalid parameters)
- **500**: Server error (models not loaded)

---

*Elaborado por Ing. Kelvin Cabrera*
