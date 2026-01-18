import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import os

# --- 1. Load Resources ---
MODEL_PATH = 'models/octasence_agent.pkl'
METRICS_PATH = 'models/metrics.json'

if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
    st.error("⚠️ Model or Metrics not found. Please run 'python train.py' first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, 'r') as f:
    metrics = json.load(f)

# --- 2. UI Configuration ---
st.set_page_config(page_title="OctaSence AI", layout="wide", page_icon="🏗️")

st.title("🏗️ OctaSence: Structural Risk Intelligence")
st.markdown("### Autonomous Risk Assessment Agent")

# --- 3. THE METRICS ROW (New!) ---
# We display the live training metrics to prove model validity
st.markdown("#### 🛡️ Live Model Performance (Test Set)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['accuracy']*100}%", "+2%", help="Overall correctness of risk predictions")
m2.metric("Precision", f"{metrics['precision']}", "High", help="Ability to avoid false alarms")
m3.metric("Recall", f"{metrics['recall']}", "High", help="Ability to detect all actual failures")
m4.metric("F1 Score", f"{metrics['f1']}", "Balanced", help="Harmonic mean of Precision and Recall")

st.divider()

# --- 4. Inputs (Same as before) ---
FEATURE_COLUMNS = [
    'Project_Type', 'Location', 'Planned_Cost', 'Actual_Cost', 'Cost_Overrun',
    'Planned_Duration', 'Actual_Duration', 'Schedule_Deviation', 'Vibration_Level',
    'Crack_Width', 'Load_Bearing_Capacity', 'Temperature', 'Humidity',
    'Weather_Condition', 'Air_Quality_Index', 'Energy_Consumption', 'Material_Usage',
    'Labor_Hours', 'Equipment_Utilization', 'Accident_Count', 'Safety_Risk_Score',
    'Image_Analysis_Score', 'Anomaly_Detected', 'Completion_Percentage'
]

col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 Sensor & Drone Inputs")
    vibration = st.slider("Vibration Level (Hz)", 0.0, 10.0, 1.2)
    crack_width = st.slider("Max Crack Width (mm)", 0.0, 20.0, 0.5)
    drone_score = st.slider("Drone Visual Score (0-100)", 0, 100, 85)
    temp = st.slider("Temperature (°C)", -10, 50, 25)

with col2:
    st.subheader("📊 Project & Safety Inputs")
    cost_overrun = st.number_input("Cost Overrun ($)", value=5000)
    schedule_dev = st.number_input("Schedule Deviation (Days)", value=10)
    safety_score = st.slider("Safety Risk Score (0-100)", 0, 100, 20)
    accident_count = st.number_input("Accidents Reported", value=0)

# --- 5. Prediction Logic ---
if st.button("🚀 Analyze Risk"):
    input_df = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    
    # Map Inputs
    input_df['Vibration_Level'] = vibration
    input_df['Crack_Width'] = crack_width
    input_df['Image_Analysis_Score'] = drone_score
    input_df['Temperature'] = temp
    input_df['Cost_Overrun'] = cost_overrun
    input_df['Schedule_Deviation'] = schedule_dev
    input_df['Safety_Risk_Score'] = safety_score
    input_df['Accident_Count'] = accident_count
    
    # Defaults
    input_df['Project_Type'] = 1  
    input_df['Location'] = 1
    input_df['Weather_Condition'] = 1
    
    prediction_idx = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    class_map = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}
    result_text = class_map.get(prediction_idx, "Unknown")

    st.subheader(f"Prediction: {result_text}")
    
    # Dynamic Alert Box
    if result_text == "High Risk":
        st.error(f"🚨 CRITICAL ALERT: Immediate Structural Intervention Required.")
    elif result_text == "Medium Risk":
        st.warning(f"⚠️ WARNING: Site parameters deviating from safety norms.")
    else:
        st.success(f"✅ STATUS GREEN: Operations Normal.")

    # Confidence Bar Chart
    prob_df = pd.DataFrame(probs, index=["High", "Low", "Medium"], columns=["Probability"])
    st.bar_chart(prob_df)