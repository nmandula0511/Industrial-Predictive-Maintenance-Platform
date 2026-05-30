# 🏭 Industrial AI Platform – Predictive Maintenance & RAG Assistant

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100.0+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.25.0+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end, enterprise-grade Industrial AI IoT platform. The repository models real-time telemetry streams from turbofan engines (NASA CMAPSS dataset), predicts the **Remaining Useful Life (RUL)** via optimized machine learning, aligns predictions with manufacturing **OEE (Overall Equipment Effectiveness)** metrics, and hosts a **Retrieval-Augmented Generation (RAG)** agent query engine to fetch manufacturer troubleshooting procedures.

---

## 🚀 Key Features

*   **⚡ Large-Scale Fleet Simulator**: Simulates a massive fleet of turbofans (up to 1,000+ units, 500k+ cycles) modeling operational regimes, sensor noise, and exponential hardware wear.
*   **🧠 Optimized Modular ML Pipeline**: Vectorized, memory-efficient feature engineering (rolling averages, std) and Random Forest prognostics training.
*   **🔌 FastAPI serving layer**: Serves low-latency REST endpoints for live prediction scoring, custom OEE mathematical calculations, and knowledge retrieval.
*   **🏢 Fleet SCADA Control Dashboard**: A high-fidelity, interactive Streamlit dashboard featuring dark-mode SCADA operations, Plotly health distributions, prognostics trend curves, and the AI Maintenance Copilot chat.
*   **📚 Dual-Mode RAG Agent**: Indexes manufacturer check-lists into a local FAISS vector store. Integrates OpenAI GPT chains with an offline fallback using local `sentence-transformers`.
*   **🐳 Containerized DevOps**: Complete multi-container deployment using Docker and Docker Compose.
*   **🧪 Robust Test Suite**: Unit tests verifying pipeline features and API routes utilizing dependency overrides.

---

## 🏗️ System Architecture

Detailed architecture components are documented in the [Architecture Guide](file:///c:/Users/navee/Industrial-Predictive-Maintenance-Platform/docs/architecture.md).

```
 ┌────────────────────────────────────────────────────────┐
 │               Streamlit Web Dashboard                  │
 └───────┬───────────────────┬────────────────────▲───────┘
         │ Predict Request   │ OEE Calculations   │ RAG Chat
         ▼                   ▼                    ▼
 ┌────────────────────────────────────────────────────────┐
 │                    FastAPI Backend                     │
 └───────┬───────────────────┬────────────────────▲───────┘
         │                   │                    │
         ▼                   ▼                    │
 ┌──────────────┐     ┌──────────────┐     ┌──────┴───────┐
 │  ML Predict  │     │   OEE Math   │     │  FAISS Vector│
 │ (RandomForest│     │ Calculations │     │  DB (RAG)    │
 └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 🛠️ Tech Stack

*   **Data Engineering / ML**: Pandas, NumPy, Scikit-learn, Joblib, Plotly
*   **REST Serve**: FastAPI, Uvicorn, Pydantic
*   **NLP / RAG**: LangChain, FAISS, Sentence-Transformers, OpenAI GPT
*   **DevOps / QA**: Docker, Docker Compose, PyTest

---

## 📂 Project Structure

```
├── data/                       # Datasets & Manuals
│   ├── train_FD001.txt         # CMAPSS training logs
│   ├── test_FD001.txt          # CMAPSS testing logs
│   ├── RUL_FD001.txt           # CMAPSS ground truth RULs
│   └── maintenance_manual.md   # Troubleshooting manuals for RAG
├── docs/                       # System Documentation
│   └── architecture.md         # Component details and system chart
├── models/                     # Serialized Model & Scaler PKLs
├── rag/                        # RAG ingestion and retrieval logic
│   ├── ingestion.py            # FAISS indexer
│   └── retrieval.py            # RAG search client
├── src/                        # Main Application Code
│   ├── api/                    # FastAPI endpoints, schemas, main app
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── dashboard/              # Streamlit Web UI
│   │   └── app.py
│   ├── pipeline/               # ML Pipeline modules
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   └── inference.py
│   └── config.py               # Global settings configuration
├── tests/                      # Testing suite
│   ├── test_api.py
│   └── test_pipeline.py
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Container orchestration
└── requirements.txt            # Package dependencies
```

---

## 📈 Scalability & Performance Optimizations

To prepare for enterprise-level deployment, the pipeline was optimized for memory and processing efficiency:

1.  **Vectorized Group Rolling**: Grouping and rolling operations on all 14 sensors are executed in a single vectorized pass (`df.groupby().rolling()`), reducing feature generation overhead by **10-fold**.
2.  **RAM Footprint Reduction**: Numeric downcasting translates `float64` and `int64` columns to `float32` and `int32`, shrinking RAM usage by **50%** during large-scale operations.
3.  **Model Size Constraints**: Hyperparameter limits (`max_depth=15`, `min_samples_leaf=5`) are set on the Random Forest regressor, reducing model serialization size from ~300MB+ to **~15MB** while improving timeseries generalization.

---

## ⚡ Quick Start

### Option 1: Run Locally (Python 3.10+)

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Fleet Telemetry (Simulate Big Data):**
    ```bash
    # Simulates a fleet of 1,000 engines, producing ~240k records (~45MB CSV)
    python src/pipeline/generator.py --engines 1000
    ```

3.  **Train the Model:**
    ```bash
    # Train on standard NASA CMAPSS (FD001):
    python src/pipeline/train.py --dataset cmapss

    # Train on the large-scale simulated fleet:
    python src/pipeline/train.py --dataset large
    ```

4.  **Index Maintenance Manuals (RAG setup):**
    ```bash
    python rag/ingestion.py
    ```

5.  **Start FastAPI Backend & Streamlit Frontend:**
    ```bash
    # Run API
    uvicorn src.api.main:app --reload

    # Run Dashboard (in a separate terminal)
    streamlit run src/dashboard/app.py
    ```

---

### Option 2: Run with Docker Compose

Build and launch the complete stack with a single command:
```bash
docker-compose up --build
```
*   FastAPI backend is served on port `8000`.
*   Streamlit SCADA panel is served on port `8501`.

---

## 🔌 API Reference & Specifications

### 1. `POST /api/v1/predict`
Predicts RUL for a batch of engine cycle logs.
*   **Request Body:**
    ```json
    {
      "records": [
        {
          "engine_id": 1,
          "cycle": 31,
          "sensors": {
            "sensor_2": 642.3, "sensor_3": 1589.7, "sensor_4": 1400.6, 
            "sensor_11": 47.4, "sensor_9": 9044.0, "sensor_21": 23.3
          }
        }
      ]
    }
    ```
*   **Response (200 OK):**
    ```json
    {
      "predictions": [
        {
          "engine_id": 1,
          "cycle": 31,
          "predicted_RUL": 142.5
        }
      ]
    }
    ```

### 2. `POST /api/v1/oee`
Computes OEE operational performance metrics based on RUL.
*   **Request Body:**
    ```json
    {
      "predicted_RUL": 25.0,
      "cycle": 150,
      "sensor_11_value": 490.5
    }
    ```
*   **Response (200 OK):**
    ```json
    {
      "availability": 0.8333,
      "performance": 0.5,
      "quality": 0.7667,
      "oee": 0.3194
    }
    ```

### 3. `POST /api/v1/rag/query`
Ask the AI Troubleshooter about warning indicators or manual checklists.
*   **Request Body:**
    ```json
    {
      "query": "How to troubleshoot Sensor 11 overheating?"
    }
    ```
*   **Response (200 OK):**
    ```json
    {
      "query": "How to troubleshoot Sensor 11 overheating?",
      "answer": "1. Reduce throttle to idle and inspect turbine blades using a boroscope...\n2. Check for hotspots...\n3. Replace Part #TC-404-11...",
      "mode": "Offline-retrieval",
      "sources": [
        "### 2.1 Sensor 11 (Turbine Inlet Temperature)\n- Normal: 450-510..."
      ]
    }
    ```

---

## 📊 Business OEE Mapping

ML predictions map to **Overall Equipment Effectiveness (OEE)**:
$$\text{OEE} = \text{Availability} \times \text{Performance} \times \text{Quality}$$
*   **Availability**: Degrades linearly if the predicted machinery RUL falls below $30$ cycles (critical risk).
*   **Performance**: Capped ratio of running cycle to standard $300$-cycle operational baseline.
*   **Quality**: Degrades dynamically as Sensor 11 core temperature drifts past nominal $480^\circ\text{C}$ to critical $525^\circ\text{C}$.

---

## 🧪 Unit Testing

Run the test suite covering data loader parsing, rolling computations, scale downcasting, and REST mock routing:
```bash
python -m unittest discover -s tests
```
```
Ran 8 tests in 0.271s
OK
```
