import os
import sys
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config
from src.api.schemas import (
    PredictRequest, PredictResponse, BatchPredictResponse, 
    OeeRequest, OeeResponse, RagQueryRequest, RagQueryResponse
)
from src.pipeline.inference import InferenceEngine
from rag.retrieval import MaintenanceCopilot

router = APIRouter()

# Global engine instances (lazy loaded)
_inference_engine = None
_rag_copilot = None

def get_inference_engine():
    global _inference_engine
    if _inference_engine is None:
        try:
            _inference_engine = InferenceEngine()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503, 
                detail=f"Predictive model is unavailable. Please train the model first. Details: {str(e)}"
            )
    return _inference_engine

def get_rag_copilot():
    global _rag_copilot
    if _rag_copilot is None:
        try:
            _rag_copilot = MaintenanceCopilot()
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"RAG Knowledge Base is unavailable. Details: {str(e)}"
            )
    return _rag_copilot

@router.post("/predict", response_model=BatchPredictResponse)
def predict_rul(request: PredictRequest, engine: InferenceEngine = Depends(get_inference_engine)):
    """Predicts Remaining Useful Life (RUL) for a batch of engine telemetry records.
    
    Historical cycle records per engine should be provided to compute accurate rolling features.
    The prediction is returned for the latest cycle of each engine in the payload.
    """
    # 1. Parse JSON request to flat structure for Pandas
    rows = []
    for record in request.records:
        row = {
            "engine_id": record.engine_id,
            "cycle": record.cycle,
            "op_setting_1": record.op_setting_1,
            "op_setting_2": record.op_setting_2,
            "op_setting_3": record.op_setting_3
        }
        # Add all 21 sensor columns, setting default to 0 if not present in request
        for i in range(1, 22):
            s_name = f"sensor_{i}"
            row[s_name] = record.sensors.get(s_name, 0.0)
            
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # 2. Run inference
    try:
        # Sort values to ensure correct rolling stats calculation
        df = df.sort_values(by=["engine_id", "cycle"])
        preds = engine.predict(df)
        df["predicted_RUL"] = preds
        
        # 3. For each engine, return the prediction for its maximum (latest) cycle in the batch
        latest_records = df.groupby("engine_id").last().reset_index()
        
        responses = []
        for _, rec in latest_records.iterrows():
            responses.append(PredictResponse(
                engine_id=int(rec["engine_id"]),
                cycle=int(rec["cycle"]),
                predicted_RUL=float(rec["predicted_RUL"])
            ))
            
        return BatchPredictResponse(predictions=responses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/oee", response_model=OeeResponse)
def calculate_oee(request: OeeRequest):
    """Computes simulated OEE metrics based on engine telemetry and RUL predictions.
    
    - Availability: 100% if predicted RUL >= 30 cycles; scales down linearly below that.
    - Performance: Capped ratio of current cycle to standard baseline cycle.
    - Quality: Evaluates Sensor 11 thermal stress. Quality drops if temp exceeds 480°C.
    - OEE: Availability * Performance * Quality.
    """
    # 1. Availability Calculation (using RUL threshold)
    if request.predicted_RUL >= Config.RUL_CRITICAL_THRESHOLD:
        availability = 1.0
    else:
        # Scale down availability linearly to represent higher downtime risk
        availability = max(0.0, request.predicted_RUL / Config.RUL_CRITICAL_THRESHOLD)
        
    # 2. Performance Calculation (represented by lifetime efficiency utilization)
    performance = min(1.0, request.cycle / Config.OEE_BASELINE_CYCLE)
    
    # 3. Quality Calculation (based on Sensor 11 thermal drift)
    # Sensor 11 normal max ~480. High stress up to 525.
    if request.sensor_11_value is None:
        quality = 1.0  # Assumed nominal if no sensor reading is passed
    else:
        temp = request.sensor_11_value
        if temp <= 480.0:
            quality = 1.0
        else:
            # Quality degrades as temperature rises above 480 toward the critical limit of 525
            degradation = (temp - 480.0) / (Config.SENSOR_11_STABLE_MAX - 480.0)
            quality = max(0.0, 1.0 - degradation)
            
    # 4. Overall OEE
    oee = availability * performance * quality
    
    return OeeResponse(
        availability=round(availability, 4),
        performance=round(performance, 4),
        quality=round(quality, 4),
        oee=round(oee, 4)
    )

@router.post("/rag/query", response_model=RagQueryResponse)
def query_manual(request: RagQueryRequest, copilot: MaintenanceCopilot = Depends(get_rag_copilot)):
    """Queries the F-100 Turbofan Maintenance Manual using the RAG Troubleshooting Assistant.
    
    Returns standard operating thresholds and step-by-step procedures for repair.
    """
    try:
        result = copilot.query(request.query)
        return RagQueryResponse(
            query=result["query"],
            answer=result["answer"],
            mode=result["mode"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Query error: {str(e)}")
