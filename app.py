import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(__file__))

from production_optimization.data_generator import generate_field_data
from production_optimization.utils.preprocessor import (
    FieldPreprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    OPT_TARGET, ALLOC_TARGET,
)
from production_optimization.models.field_optimizer import FieldOptimizer
from production_optimization.models.allocation_model import AllocationModel

app = Flask(__name__)

MODEL_DIR = os.path.join("outputs", "models")

optimizer = FieldOptimizer()
allocator = AllocationModel()

try:
    optimizer.load(os.path.join(MODEL_DIR, "field_optimizer.joblib"))
    allocator.load(os.path.join(MODEL_DIR, "allocation_model.joblib"))
    opt_preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor_opt.joblib"))
    alloc_preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor_alloc.joblib"))
    models_loaded = True
except Exception:
    models_loaded = False
    opt_preprocessor = None
    alloc_preprocessor = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
    })


@app.route("/api/models", methods=["GET"])
def models_info():
    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 500
    return jsonify({
        "field_optimizer": {
            "algorithm": "GradientBoostingRegressor",
            "cv_r2": optimizer.cv_score,
            "feature_importances": optimizer.feature_importances(),
        },
        "allocation_model": {
            "algorithm": "RandomForestRegressor",
            "cv_r2": allocator.cv_score,
            "feature_importances": allocator.feature_importances(),
        },
    })


@app.route("/api/optimize", methods=["POST"])
def optimize():
    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 500

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    required = ["well_count", "total_water_rate", "total_gas_rate",
                 "water_injection_rate", "lift_type", "operating_cost_usd",
                 "revenue_per_bbl", "artificial_lift_power_kw"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    df = pd.DataFrame([data])
    X = opt_preprocessor.transform(df)
    pred = optimizer.predict(X)[0]

    return jsonify({
        "predicted_net_profit": round(float(pred), 2),
        "input": data,
    })


@app.route("/api/allocate", methods=["POST"])
def allocate():
    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 500

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    required = ["well_count", "total_water_rate", "total_gas_rate",
                 "water_injection_rate", "lift_type", "operating_cost_usd",
                 "revenue_per_bbl", "artificial_lift_power_kw"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    df = pd.DataFrame([data])
    X = alloc_preprocessor.transform(df)
    pred = allocator.predict(X)[0]

    return jsonify({
        "predicted_production_efficiency": round(float(pred), 4),
        "input": data,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5014, debug=False)
