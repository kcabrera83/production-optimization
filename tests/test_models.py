import pytest
import os
import joblib
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")

SAMPLE_DATA = {
    "well_count": 10,
    "total_water_rate": 5000.0,
    "total_gas_rate": 2000.0,
    "water_injection_rate": 3000.0,
    "lift_type": "ESP",
    "operating_cost_usd": 150000.0,
    "revenue_per_bbl": 75.0,
    "artificial_lift_power_kw": 200.0,
}


def test_optimizer_model_loads():
    from production_optimization.models.field_optimizer import FieldOptimizer
    model = FieldOptimizer()
    model.load(os.path.join(MODELS_DIR, "field_optimizer.joblib"))
    assert model is not None
    assert model.model is not None


def test_allocator_model_loads():
    from production_optimization.models.allocation_model import AllocationModel
    model = AllocationModel()
    model.load(os.path.join(MODELS_DIR, "allocation_model.joblib"))
    assert model is not None
    assert model.model is not None


def test_preprocessor_opt_loads():
    path = os.path.join(MODELS_DIR, "preprocessor_opt.joblib")
    assert os.path.exists(path)
    preprocessor = joblib.load(path)
    assert preprocessor is not None


def test_preprocessor_alloc_loads():
    path = os.path.join(MODELS_DIR, "preprocessor_alloc.joblib")
    assert os.path.exists(path)
    preprocessor = joblib.load(path)
    assert preprocessor is not None


def test_optimizer_prediction():
    from production_optimization.models.field_optimizer import FieldOptimizer
    model = FieldOptimizer()
    model.load(os.path.join(MODELS_DIR, "field_optimizer.joblib"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor_opt.joblib"))
    df = pd.DataFrame([SAMPLE_DATA])
    X = preprocessor.transform(df)
    pred = model.predict(X)
    assert pred is not None
    assert len(pred) == 1


def test_allocator_prediction():
    from production_optimization.models.allocation_model import AllocationModel
    model = AllocationModel()
    model.load(os.path.join(MODELS_DIR, "allocation_model.joblib"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor_alloc.joblib"))
    df = pd.DataFrame([SAMPLE_DATA])
    X = preprocessor.transform(df)
    pred = model.predict(X)
    assert pred is not None
    assert len(pred) == 1
    assert 0.0 <= pred[0] <= 1.0
