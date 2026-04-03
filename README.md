## 🏗️ Architecture
Data Ingestion → Preprocessing → Feature Engineering → Model Training → Prediction → KPI Calculation (OEE)

# 🏭 Industrial AI Platform – Predictive Maintenance

## 📌 Problem Statement
Manufacturing systems face unplanned downtime due to unexpected machine failures.

This project builds an AI-driven predictive maintenance system to:
- Predict Remaining Useful Life (RUL)
- Identify critical failure-driving sensors
- Align predictions with manufacturing KPIs such as OEE

---

## 🚀 Solution Overview
An end-to-end ML pipeline that processes industrial sensor data and predicts machine failure.

**Pipeline:**  
Data → Feature Engineering → ML Model → RUL Prediction → OEE Metrics → Insights

## 🧠 Features
- Predict Remaining Useful Life (RUL)
- Feature importance analysis for sensor insights
- OEE (Overall Equipment Effectiveness) calculation
- Industrial time-series feature engineering

---

## 🛠️ Tech Stack
- Python
- Pandas
- Scikit-learn
- NumPy

---

## 📊 Dataset
NASA Turbofan Engine Dataset (CMAPSS)

---

## 📈 Results
- Model: Random Forest Regressor
- RMSE: ~41 cycles

**Key Sensors Identified:**
- sensor_11 (most important)
- sensor_4
- sensor_9

---

## 🏭 Business Impact
- Predict failures before breakdown
- Reduce unplanned downtime
- Improve maintenance planning
- Align ML outputs with OEE metrics

---

## ⚡ Real-World Integration

In a production environment, this system can be integrated with:
- MES / SCADA systems
- OPC-UA / MQTT data streams
- Real-time monitoring dashboards

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/explore_data.py
python src/train_model.py
