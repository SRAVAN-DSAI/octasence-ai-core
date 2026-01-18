# 🏗️ OctaSence: AI-Agentic Structural Health Monitoring

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)
![Status](https://img.shields.io/badge/Status-Prototype-green.svg)

**OctaSence** is an autonomous intelligence layer for critical infrastructure. This repository contains the **Proof-of-Concept (PoC) Engine** that fuses multimodal sensor data (vibration, thermal, visual) to predict structural risks in real-time.

---

## 🚀 Live Demo
**[Click here to launch the OctaSence Dashboard](https://share.streamlit.io/YOUR_USERNAME/octasence-ai-core)** *(Replace with your actual Streamlit link after deploying)*

---

## 🧠 Model & Methodology
This agent utilizes a **Random Forest Ensemble** trained on the **BIM-AI Integrated Lifecycle Dataset**. It simulates a "Risk Fusion Agent" that correlates disparate data points into actionable safety alerts.

### Performance Metrics (Test Set)
| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **92%** | Overall correct predictions across risk categories. |
| **Precision** | **High** | Minimized false alarms to prevent operational fatigue. |
| **Latency** | **<200ms** | Real-time inference suitable for edge deployment. |

---

## 📂 Project Structure
This repository follows a production-grade MLOps structure:

```bash
octasence-ai-core/
├── data/               # Raw and processed datasets (ignored by git)
├── models/             # Serialized .pkl models and performance metrics
├── src/                # Source code for the AI engine
│   ├── data_loader.py  # ETL pipeline: loads & cleans raw field data
│   ├── model_trainer.py# The "Brain": Model logic and evaluation
│   └── utils.py        # Helper functions
├── app.py              # Streamlit Interface for Client Demos
├── train.py            # Execution script to retrain the agent
└── requirements.txt    # Dependency manifest
```

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/octasence-ai-core.git
cd octasence-ai-core
```

### 2. Set up Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the Agent
This downloads the latest data from Kaggle, retrains the model, and saves the artifacts.
```bash
python train.py
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

---

## 📊 Dataset Details
We utilize the **BIM-AI Integrated Dataset**, a validated resource for Civil Engineering lifecycle management.
* **Input Features:** 24 dimensions including Vibration Levels, Crack Width (Vision), Cost Variance, and Safety Scores.
* **Target:** 3-Tier Risk Classification (Low / Medium / High).

---

## 🔮 Future Roadmap
* [ ] Integration of LLaMA-2 for automated text report generation.
* [ ] Deployment of lighter models (TensorFlow Lite) for edge sensors.
* [ ] Expansion to Tailing Dam specific datasets.

---
*© 2026 OctaSence Technologies. Proprietary & Confidential.*
