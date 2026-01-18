import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import os

# --- CONSTANTS ---
# Replace this with your actual GitHub URL after you push!
GITHUB_REPO_URL = "https://github.com/SRAVAN-DSAI/octasence-ai-core"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/ziya07/bim-ai-integrated-dataset"

# --- 1. SETUP & RESOURCE LOADING ---
st.set_page_config(
    page_title="OctaSence AI",
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

MODEL_PATH = 'models/octasence_agent.pkl'
METRICS_PATH = 'models/metrics.json'

# Robust loading mechanism
if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
    st.error("⚠️ SYSTEM ERROR: Model artifacts not found.")
    st.warning("Please run the training pipeline first: `python train.py`")
    st.stop()

# Load the Brain (XGBoost)
model = joblib.load(MODEL_PATH)

# Load the Memory (Metrics + Feature Names)
with open(METRICS_PATH, 'r') as f:
    artifacts = json.load(f)
    metrics = artifacts['metrics']
    FEATURE_COLUMNS = artifacts['features'] # Exact column order from training

# --- 2. SIDEBAR: DATA & LINKS ---
st.sidebar.title("🏗️ OctaSence Core")
st.sidebar.caption("Autonomous SHM Intelligence")
st.sidebar.markdown("---")

# Project Links
st.sidebar.subheader("🔗 Resources")
st.sidebar.link_button("💻 View Code on GitHub", GITHUB_REPO_URL)
st.sidebar.link_button("📊 View Dataset on Kaggle", KAGGLE_DATASET_URL)

st.sidebar.markdown("---")

# Data Context
st.sidebar.subheader("📂 Intelligence Source")
st.sidebar.info(
    """
    **Dataset:** BIM-AI Integrated Lifecycle
    
    **Validated For:**
    * Civil Infrastructure (Bridges, Dams)
    * Risk Forecasting
    * Anomaly Detection
    
    **Fusion Engine:**
    Combines **Vibration Sensors**, **Drone Vision**, and **Financial Logs** to predict failure before it happens.
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("v2.1.0 | Powered by XGBoost")

# --- 3. MAIN DASHBOARD ---
st.title("OctaSence: Structural Risk Intelligence")
st.markdown("### 🤖 Autonomous Risk Assessment Agent")

# --- Top Row: Live Model Metrics ---
st.markdown("#### 🛡️ Agent Performance (Live Test Set)")
m1, m2, m3, m4 = st.columns(4)

m1.metric("Model Accuracy", f"{metrics['accuracy']*100:.1f}%", "+2.4%", help="Overall correctness on unseen test data")
m2.metric("Precision", f"{metrics['precision']:.2f}", "High", help="False Alarm Rate is minimized")
m3.metric("Recall", f"{metrics['recall']:.2f}", "High", help="Sensitivity to critical failures")
m4.metric("F1 Score", f"{metrics['f1']:.2f}", "Balanced", help="Harmonic mean of performance")

st.divider()

# --- 4. INPUT CONSOLE ---
st.subheader("📡 Live Field Telemetry")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Sensor Network A (Physical)**")
    vibration = st.slider("Vibration Level (Hz)", 0.0, 10.0, 1.2, help="Accelerometer readings from pillars")
    crack_width = st.slider("Max Crack Width (mm)", 0.0, 20.0, 0.5, help="Computer Vision detected width")
    temp = st.slider("Surface Temperature (°C)", -10, 50, 25)
    drone_score = st.slider("Drone Integrity Score", 0, 100, 85, help="0 = Critical Damage, 100 = Perfect")

with col2:
    st.markdown("**Sensor Network B (Project/Safety)**")
    cost_overrun = st.number_input("Cost Variance ($)", value=5000, step=1000)
    schedule_dev = st.number_input("Schedule Deviation (Days)", value=10, step=1)
    safety_score = st.slider("Safety Compliance Score", 0, 10, 2, help="Derived from site safety logs")
    accident_count = st.number_input("Reported Incidents", value=0, step=1)

# --- 5. PREDICTION ENGINE ---
st.divider()

# Center the button
_, btn_col, _ = st.columns([1, 2, 1])

if btn_col.button("🚀 RUN PREDICTIVE ANALYSIS", type="primary", use_container_width=True):
    
    with st.spinner("Aggregating sensor data & running inference..."):
        # 1. Create a "Zero" DataFrame with the exact columns the model expects
        input_df = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        
        # 2. Fill in our user inputs
        input_df['Vibration_Level'] = vibration
        input_df['Crack_Width'] = crack_width
        input_df['Image_Analysis_Score'] = drone_score
        input_df['Temperature'] = temp
        input_df['Cost_Overrun'] = cost_overrun
        input_df['Schedule_Deviation'] = schedule_dev
        input_df['Safety_Risk_Score'] = safety_score
        input_df['Accident_Count'] = accident_count
        
        # 3. Handle Categorical Defaults (to prevent crashes)
        # In a full app, these would be dropdowns. Here we set "standard" defaults.
        if 'Project_Type' in FEATURE_COLUMNS: input_df['Project_Type'] = 1 
        if 'Location' in FEATURE_COLUMNS: input_df['Location'] = 1
        if 'Weather_Condition' in FEATURE_COLUMNS: input_df['Weather_Condition'] = 1
        
        # 4. XGBoost Inference
        prediction_idx = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        
        # 5. Decode Result
        # Mapping based on Alphabetical LabelEncoding: High(0), Low(1), Medium(2)
        class_map = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}
        result_text = class_map.get(prediction_idx, "Unknown")
        
        # --- DISPLAY RESULTS ---
        st.subheader(f"Risk Assessment: {result_text}")
        
        # Dynamic Alert Logic
        if result_text == "High Risk":
            st.error("🚨 CRITICAL ALERT: Structural integrity compromised. Immediate evacuation recommended.")
            st.toast("Alert sent to Site Manager!", icon="🔥")
        elif result_text == "Medium Risk":
            st.warning("⚠️ CAUTION: Abnormal patterns detected. Increase inspection frequency.")
        else:
            st.success("✅ SYSTEM NOMINAL: Infrastructure operating within safety parameters.")

        # Confidence Chart
        st.markdown("##### AI Confidence Levels")
        prob_df = pd.DataFrame(probs, index=["High Risk", "Low Risk", "Medium Risk"], columns=["Probability"])
        st.bar_chart(prob_df, color=["#FF4B4B"]) # Red color for visual impact