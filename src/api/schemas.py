from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class TelemetryRecord(BaseModel):
    engine_id: int = Field(..., description="ID of the engine", example=1)
    cycle: int = Field(..., description="Current operational cycle count", example=31)
    # Settings (defaulting to typical F-100 values if not provided)
    op_setting_1: float = Field(0.0, description="Operating setting 1")
    op_setting_2: float = Field(0.0, description="Operating setting 2")
    op_setting_3: float = Field(100.0, description="Operating setting 3")
    # Sensor dictionary: maps e.g. "sensor_1" to float values
    sensors: Dict[str, float] = Field(
        ..., 
        description="Key-value mapping of sensor names to their values. Must contain at least the selected sensors.",
        example={
            "sensor_2": 642.3, "sensor_3": 1589.7, "sensor_4": 1400.6, "sensor_7": 554.3, 
            "sensor_8": 2388.0, "sensor_9": 9044.0, "sensor_11": 47.4, "sensor_12": 521.6, 
            "sensor_13": 2388.0, "sensor_14": 8138.6, "sensor_15": 8.4, "sensor_17": 392.0, 
            "sensor_20": 39.0, "sensor_21": 23.3
        }
    )

class PredictRequest(BaseModel):
    records: List[TelemetryRecord] = Field(..., description="List of telemetry records. For rolling features calculation, historical records of the same engine should be included.")

class PredictResponse(BaseModel):
    engine_id: int
    cycle: int
    predicted_RUL: float = Field(..., description="Predicted Remaining Useful Life in cycles")

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

class OeeRequest(BaseModel):
    predicted_RUL: float = Field(..., description="Predicted RUL for the engine")
    cycle: int = Field(..., description="Current operational cycle")
    sensor_11_value: Optional[float] = Field(None, description="Current reading of Sensor 11 for Quality evaluation")

class OeeResponse(BaseModel):
    availability: float = Field(..., description="Availability rate (0 to 1)")
    performance: float = Field(..., description="Performance rate (0 to 1)")
    quality: float = Field(..., description="Quality rate (0 to 1)")
    oee: float = Field(..., description="Overall Equipment Effectiveness (0 to 1)")

class RagQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question about troubleshooting or operations", example="How to fix Sensor 11 hot temperature?")

class RagQueryResponse(BaseModel):
    query: str
    answer: str
    mode: str = Field(..., description="Response generation mode: 'LLM-enhanced' or 'Offline-retrieval'")
    sources: List[str] = Field(..., description="List of source document chunks retrieved from the manual")
