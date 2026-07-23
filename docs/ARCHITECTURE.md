# Architecture - Production Optimization

## System Overview

```
+------------------+     +-------------------+     +------------------+
|   Data Layer     | --> |   Model Layer     | --> |   API Layer      |
| (Data Generator) |     | (ML Models)       |     | (Flask REST)     |
+------------------+     +-------------------+     +------------------+
                                                          |
                                                          v
                                                 +------------------+
                                                 | Dashboard Layer  |
                                                 | (HTML/CSS/JS)    |
                                                 +------------------+
```

## Components

### Data Layer

- **Source**: Synthetic data generator (`generate_field_data`)
- **Samples**: 2,000 field operation records
- **Features**: 8 parameters (7 numeric + 1 categorical)
- **Targets**: net_profit (regression), production_efficiency (regression)

### Model Layer

#### Field Optimizer
- **Algorithm**: GradientBoostingRegressor
- **Task**: Predict net profit (USD) for field operations
- **Input**: 8 features (7 numeric + 1 categorical encoded)
- **Output**: Net profit (continuous)
- **Serialization**: joblib (`.joblib`)
- **Cross-validation**: CV R2 score reported

#### Allocation Model
- **Algorithm**: RandomForestRegressor
- **Task**: Predict production efficiency (0-1 scale)
- **Input**: Same 8 features
- **Output**: Production efficiency (continuous)
- **Serialization**: joblib (`.joblib`)
- **Cross-validation**: CV R2 score reported

### Preprocessing Pipeline

- **ColumnTransformer** with:
  - StandardScaler for 7 numeric features
  - OneHotEncoder for lift_type (4 categories)
- Separate preprocessors for each model target
- Saved as `preprocessor_opt.joblib` and `preprocessor_alloc.joblib`

### API Layer

- **Framework**: Flask
- **Port**: 5014
- **Format**: JSON request/response
- **Endpoints**: 5 (optimize, allocate, health, models, docs)

### Dashboard Layer

- **Frontend**: HTML/CSS/JS (Jinja2 templates)
- **Charts**: Plotly.js for feature importance visualization
- **Theme**: Dark theme UI

## Data Flow

### Net Profit Optimization Flow
1. User provides field parameters
2. Input wrapped in DataFrame
3. Preprocessor transforms features (scale + encode)
4. GradientBoostingRegressor predicts net profit
5. Response returns prediction + input echo

### Production Efficiency Flow
1. User provides same field parameters
2. Separate preprocessor transforms features
3. RandomForestRegressor predicts efficiency
4. Response returns prediction + input echo

## Feature Processing

| Feature | Type | Transform | Description |
|---------|------|-----------|-------------|
| well_count | numeric | StandardScaler | Number of wells |
| total_water_rate | numeric | StandardScaler | Water production rate |
| total_gas_rate | numeric | StandardScaler | Gas production rate |
| water_injection_rate | numeric | StandardScaler | Injection rate |
| operating_cost_usd | numeric | StandardScaler | Operating expenses |
| revenue_per_bbl | numeric | StandardScaler | Oil revenue per barrel |
| artificial_lift_power_kw | numeric | StandardScaler | Lift power consumption |
| lift_type | categorical | OneHotEncoder | ESP/rod_pump/gas_lift/progressive_cavity |

## Project Structure

```
production-optimization/
├── production_optimization/
│   ├── __init__.py
│   ├── data_generator.py              # Synthetic field data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── field_optimizer.py         # GradientBoosting regressor
│   │   └── allocation_model.py        # RandomForest regressor
│   └── utils/
│       ├── __init__.py
│       └── preprocessor.py            # ColumnTransformer pipeline
├── templates/
│   └── index.html                     # Dashboard UI
├── outputs/
│   └── models/                        # Saved model artifacts
├── train.py                           # Training pipeline
├── app.py                             # Flask API server
├── test_api.py                        # API test suite (pytest)
├── requirements.txt
└── setup.py
```

## Saved Artifacts

| File | Description |
|------|-------------|
| field_optimizer.joblib | GradientBoosting model for net profit |
| allocation_model.joblib | RandomForest model for production efficiency |
| preprocessor_opt.joblib | Preprocessor for profit optimization |
| preprocessor_alloc.joblib | Preprocessor for efficiency allocation |
| training_results.json | CV scores and dataset summary |
| field_data.csv | Generated training data |

## Model Evaluation

### Field Optimizer
- CV R2: ~0.88+
- Feature importances available via `/api/models`

### Allocation Model
- CV R2: ~0.86+
- Feature importances available via `/api/models`

---

*Elaborado por Ing. Kelvin Cabrera*
