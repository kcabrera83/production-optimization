import streamlit as st, joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Production Optimizer")
st.title("Production Optimizer")

p = Path(__file__).parent / 'outputs' / 'models'
models = {'profit': joblib.load(p / 'net_profit_model.pkl'), 'efficiency': joblib.load(p / 'production_efficiency_model.pkl')}

step = st.session_state.get('step', 1)

if step == 1:
    st.subheader('Step 1: Basic Parameters')
    wells = st.slider('Wells', 1, 100, 50)
    water = st.slider('Water', 0, 50000, 25000)
    gas = st.slider('Gas', 0, 100000, 50000)
    inj = st.slider('Inj', 0, 50000, 25000)
    if st.button('Next'):
        st.session_state.update({'step': 2})
        st.rerun()

elif step == 2:
    st.subheader('Step 2: Advanced Parameters')
    lift = st.selectbox('Lift', ['ESP','gas','rod','pcp'])
    opex = st.slider('Opex', 1000, 100000, 50500)
    rev = st.slider('Rev', 20, 100, 60)
    power = st.slider('Power', 0, 10000, 5000)
    if st.button('Run'):
        st.session_state['step'] = 1