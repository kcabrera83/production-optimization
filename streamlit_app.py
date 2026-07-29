import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Production Optimization", layout="wide")
st.title("Production Optimization")
st.markdown("Optimize field production to maximize net profit.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'profit': joblib.load(d / 'net_profit_model.pkl'), 'efficiency': joblib.load(d / 'production_efficiency_model.pkl')}

st.sidebar.header("Input Parameters")
well_count = st.sidebar.slider('Well Count', 1, 100, 50)
water_rate = st.sidebar.slider('Water Rate', 0, 50000, 25000)
gas_rate = st.sidebar.slider('Gas Rate', 0, 100000, 50000)
water_injection = st.sidebar.slider('Water Injection', 0, 50000, 25000)
lift_type = st.sidebar.selectbox('Lift Type', ['ESP','gas_lift','rod_pump','pcp'])
opex = st.sidebar.slider('Opex', 1000, 100000, 50500)
revenue = st.sidebar.slider('Revenue', 20, 100, 60)
lift_power = st.sidebar.slider('Lift Power', 0, 10000, 5000)

if st.sidebar.button("Run"):
    try:
        x = np.array([[well_count, water_rate, gas_rate, water_injection, lift_type, opex, revenue, lift_power]])
        cols = st.columns(2)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))