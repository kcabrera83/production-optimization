"""FastAPI for production optimization."""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from production_optimization.data_generator import generate_field_data
from production_optimization.utils.preprocessor import (
    FieldPreprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    OPT_TARGET, ALLOC_TARGET,
)
from production_optimization.models.field_optimizer import FieldOptimizer
from production_optimization.models.allocation_model import AllocationModel

app = FastAPI(
    title="Production Optimization",
    description="Field production optimization and allocation prediction",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

MODEL_DIR = os.path.join("outputs", "models")

optimizer = FieldOptimizer()
allocator = AllocationModel()
models_loaded = False
opt_preprocessor = None
alloc_preprocessor = None


@app.on_event("startup")
async def load_models():
    global models_loaded, opt_preprocessor, alloc_preprocessor
    try:
        optimizer.load(os.path.join(MODEL_DIR, "field_optimizer.joblib"))
        allocator.load(os.path.join(MODEL_DIR, "allocation_model.joblib"))
        opt_preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor_opt.joblib"))
        alloc_preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor_alloc.joblib"))
        models_loaded = True
    except Exception:
        models_loaded = False


class FieldRequest(BaseModel):
    well_count: int
    total_water_rate: float
    total_gas_rate: float
    water_injection_rate: float
    lift_type: str
    operating_cost_usd: float
    revenue_per_bbl: float
    artificial_lift_power_kw: float


class OptimizeResponse(BaseModel):
    predicted_net_profit: float
    input: dict


class AllocateResponse(BaseModel):
    predicted_production_efficiency: float
    input: dict


@app.get("/api/health")
async def health():
    return {"status": "healthy", "models_loaded": models_loaded}


@app.get("/api/models")
async def models_info():
    if not models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")
    return {
        "field_optimizer": {
            "algorithm": "pymoo NSGA2 multi-objective",
            "cv_r2": optimizer.cv_score,
            "feature_importances": optimizer.feature_importances(),
        },
        "allocation_model": {
            "algorithm": "OR-Tools LP + linear regression",
            "cv_r2": allocator.cv_score,
            "feature_importances": allocator.feature_importances(),
        },
    }


@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize(request: FieldRequest):
    if not models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")
    data = request.model_dump()
    df = pd.DataFrame([data])
    X = opt_preprocessor.transform(df)
    pred = optimizer.predict(X)[0]
    return OptimizeResponse(predicted_net_profit=round(float(pred), 2), input=data)


@app.post("/api/allocate", response_model=AllocateResponse)
async def allocate(request: FieldRequest):
    if not models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")
    data = request.model_dump()
    df = pd.DataFrame([data])
    X = alloc_preprocessor.transform(df)
    pred = allocator.predict(X)[0]
    return AllocateResponse(predicted_production_efficiency=round(float(pred), 4), input=data)
