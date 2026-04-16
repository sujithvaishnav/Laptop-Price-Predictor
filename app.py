import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")
st.title("💻 Laptop Price Predictor")
st.markdown("Fill in the laptop specifications below to get an estimated price.")
st.divider()

# ── Load Artefacts ────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model    = joblib.load('Best_Model.pkl')
    encoders = joblib.load('encoders.pkl')   # dict of fitted LabelEncoders (saved from notebook)
    return model, encoders

try:
    model, encoders = load_artefacts()
except Exception as e:
    st.error(f"❌ Could not load model artefacts: {e}")
    st.info("Make sure `Best_Model.pkl` and `encoders.pkl` are in the same folder as `app.py`.")
    st.stop()

# ── Category Name Lists (from fitted encoders) ────────────────
brands     = list(encoders['Company'].classes_)
types      = list(encoders['TypeName'].classes_)
os_options = list(encoders['OpSys'].classes_)
cpu_brands = list(encoders['Cpu brand'].classes_)
gpu_brands = list(encoders['Gpu brand'].classes_)

# ── UI Layout ─────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🖥️ Brand & Type")
    company  = st.selectbox("Brand",            brands)
    typename = st.selectbox("Laptop Type",      types)
    os       = st.selectbox("Operating System", os_options)

with col2:
    st.subheader("⚙️ Processor & Graphics")
    cpu = st.selectbox("CPU Brand", cpu_brands)
    gpu = st.selectbox("GPU Brand", gpu_brands)

st.divider()
col3, col4 = st.columns(2)

with col3:
    st.subheader("📐 Display")
    inches      = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0,
                                   value=15.6, step=0.1)
    resolution  = st.selectbox("Screen Resolution",
                                ['1920x1080', '2560x1440', '3840x2160',
                                 '1366x768',  '1600x900',  '2880x1800',
                                 '2560x1600', '2304x1440'])
    touchscreen = st.radio("Touchscreen", ["No", "Yes"], horizontal=True)
    hd          = st.radio("HD Display",  ["No", "Yes"], horizontal=True)

with col4:
    st.subheader("🔩 Memory & Build")
    ram    = st.selectbox("RAM (GB)",    [2, 4, 6, 8, 12, 16, 32, 64])
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0,
                              value=2.0, step=0.1)

st.divider()
st.subheader("💾 Storage Configuration")

scol1, scol2 = st.columns(2)
with scol1:
    st.markdown("**Primary Storage**")
    primary_ssd    = st.selectbox("Primary SSD (GB)",           [0, 32, 64, 128, 256, 512, 1000])
    primary_hdd    = st.selectbox("Primary HDD (GB)",           [0, 32, 64, 128, 256, 512, 1000])
    primary_flash  = st.selectbox("Primary Flash Storage (GB)", [0, 32, 64, 128])
    primary_hybrid = st.selectbox("Primary Hybrid (GB)",        [0, 32, 64, 128, 256])

with scol2:
    st.markdown("**Secondary Storage**")
    secondary_ssd    = st.selectbox("Secondary SSD (GB)",    [0, 32, 64, 128, 256, 512, 1000])
    secondary_hdd    = st.selectbox("Secondary HDD (GB)",    [0, 32, 64, 128, 256, 512, 1000])
    secondary_hybrid = st.selectbox("Secondary Hybrid (GB)", [0, 32, 64, 128, 256, 512, 1000])

st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button("🔍 Predict Price", use_container_width=True, type="primary"):

    # Encode categoricals using the saved encoders (category name → integer)
    company_enc  = int(encoders['Company'].transform([company])[0])
    typename_enc = int(encoders['TypeName'].transform([typename])[0])
    os_enc       = int(encoders['OpSys'].transform([os])[0])
    cpu_enc      = int(encoders['Cpu brand'].transform([cpu])[0])
    gpu_enc      = int(encoders['Gpu brand'].transform([gpu])[0])

    # Binary flags
    touch_flag = 1 if touchscreen == "Yes" else 0
    hd_flag    = 1 if hd == "Yes" else 0

    # Compute PPI from resolution + screen size
    width, height = map(int, resolution.split('x'))
    ppi = ((width**2 + height**2) ** 0.5) / inches

    # Assemble feature vector (must match training column order exactly)
    features = np.array([[
        company_enc, typename_enc, ram, inches, os_enc, weight,
        touch_flag, ppi, hd_flag, cpu_enc, gpu_enc,
        primary_flash, primary_hybrid, primary_ssd, primary_hdd,
        secondary_hdd, secondary_hybrid, secondary_ssd
    ]])

    try:
        prediction = model.predict(features)[0]
        st.success(f"### 💰 Estimated Price: ₹{prediction:,.0f}")
        st.caption("Prediction based on a machine learning model trained on real laptop market data.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
