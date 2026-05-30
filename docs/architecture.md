# 🏗️ Architecture Design & System Flow

This document details the software architecture, data pipelines, and service components of the **Industrial Predictive Maintenance Platform**.

---

## 📌 High-Level Architecture Diagram

The system consists of three main tiers:
1. **Machine Learning & Data Processing Pipeline**: Preprocesses sensor readings and trains a regressor.
2. **Serving Backend (FastAPI)**: Serves prediction scores, calculates manufacturing KPIs (OEE), and powers the RAG Knowledge Engine.
3. **Application Layer (Streamlit Dashboard)**: Connects to the backend to render dashboards and operator interfaces.

```mermaid
flowchart TD
    %% Dataset
    subgraph DataTier [1. Raw Data & Documents]
        DS[NASA CMAPSS Turbofan Dataset]
        MM[F-100 Maintenance Manual MD]
        SF[Simulated Fleet Telemetry CSV]
    end

    %% Pipeline
    subgraph PipelineTier [2. ML & RAG Processing]
        GS[Telemetry Simulator] -->|Generates| SF
        DL[Data Loader] --> FE[Feature Engineer]
        FE -->|Rolling Features| TR[Model Trainer]
        TR -->|Saves Models| AM[Artifacts: models/predictor.pkl & scaler.pkl]
        
        MM -->|Parses & Splits| IN[Ingestion Pipeline]
        IN -->|Vector Embeddings| VS[FAISS Vector Store]
    end

    %% Backend
    subgraph ApiTier [3. Serve API Backend - FastAPI]
        api_predict[POST /api/v1/predict]
        api_oee[POST /api/v1/oee]
        api_rag[POST /api/v1/rag/query]
    end

    %% UI
    subgraph UiTier [4. UI & Operator Dashboard - Streamlit]
        dash_telemetry[Telemetry Monitor]
        dash_prognostics[RUL Forecaster]
        dash_oee[OEE Dashboard]
        dash_rag[AI Maintenance Copilot]
    end

    %% Flow connections
    DS --> DL
    SF --> DL
    AM -->|Loaded by| api_predict
    VS -->|Indexed & Queried| api_rag
    
    api_predict -->|Calculates Life Score| api_oee
    
    api_predict <-->|REST API| dash_prognostics
    api_oee <-->|REST API| dash_oee
    api_rag <-->|REST API| dash_rag
    DS & SF -->|Interactive selection| dash_telemetry

    classDef orange fill:#FFF8E1,stroke:#FFB300,stroke-width:2px;
    classDef blue fill:#E1F5FE,stroke:#03A9F4,stroke-width:2px;
    classDef green fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px;
    classDef grey fill:#ECEFF1,stroke:#607D8B,stroke-width:2px;
    
    class DataTier orange;
    class PipelineTier blue;
    class ApiTier green;
    class UiTier grey;
```

---

## 🛠️ Component Breakdown

### 1. Data Ingestion & Scaling (`src/pipeline/`)
*   **`data_loader.py`**: Reads NASA's space-delimited text logs. Safely handles trailing blanks and applies core schema mapping.
*   **`feature_engineering.py`**: Translates raw records into a historical timeseries window. Computes rolling averages and rolling standard deviations for the selected core sensor indices. Scales input spaces using `StandardScaler`.
*   **`train.py`**: Fits a Random Forest Regressor to estimate Remaining Useful Life (RUL). Evaluates model with RMSE and MAE before outputting serialization components.
*   **`inference.py`**: Performs predictions. Integrates CMAPSS test logic to evaluate accuracy against final cycle ground truths.

### 2. Retrieval-Augmented Generation (`src/rag/`)
*   **`ingestion.py`**: Reads maintenance instructions, segments sections via a recursive character text splitter, maps them into FAISS indices, and saves them locally.
*   **`retrieval.py`**: Intercepts search queries. Leverages local HuggingFace embeddings or OpenAI embeddings to select relevant manual blocks, compiling structured troubleshooting guides.

### 3. Serving Backend (`src/api/`)
*   **`main.py`**: FastAPI entrypoint mounting middleware and routes.
*   **`routes.py`**: Implements endpoints:
    *   `/predict`: Processes sequence readings to output expected RUL.
    *   `/oee`: Combines cycle metrics, RUL limits, and sensor core heat to evaluate Availability, Performance, Quality, and overall OEE.
    *   `/rag/query`: Connects natural language developer/operator questions to the RAG database.
*   **`schemas.py`**: Strongly-typed request/response models.

### 4. Interactive Operator Interface (`src/dashboard/app.py`)
*   Provides dynamic telemetry plotting, simulated forecasting, automated metrics panels, and a chat interface to the RAG assistant.
