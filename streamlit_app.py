import streamlit as st
import pickle
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Production Optimization", layout="wide")
st.title("Production Optimization")
st.markdown("Optimize field production and well allocation to maximize net profit.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("profit", "net_profit_model.pkl"), ("efficiency", "production_efficiency_model.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
well_count = st.sidebar.slider("Well Count", 1, 100, 50)
water_rate_bbl_d = st.sidebar.slider("Water Rate Bbl D", 0, 50000, 25000)
gas_rate_mcf_d = st.sidebar.slider("Gas Rate Mcf D", 0, 100000, 50000)
water_injection_bbl_d = st.sidebar.slider("Water Injection Bbl D", 0, 50000, 25000)
lift_type = st.sidebar.selectbox("Lift Type", ['ESP', 'gas_lift', 'rod_pump', 'pcp'])
operating_cost_usd = st.sidebar.slider("Operating Cost Usd", 1000, 100000, 50500)
revenue_per_bbl = st.sidebar.slider("Revenue Per Bbl", 20, 100, 60)
lift_power_kw = st.sidebar.slider("Lift Power Kw", 0, 10000, 5000)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[well_count, water_rate_bbl_d, gas_rate_mcf_d, water_injection_bbl_d, lift_type, operating_cost_usd, revenue_per_bbl, lift_power_kw]])
        m = models["profit"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Profit", result if isinstance(result, str) else f"{result:.4f}")
        m = models["efficiency"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Efficiency", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")
