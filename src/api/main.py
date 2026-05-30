import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.api.routes import router as api_router

app = FastAPI(
    title="Industrial AI Predictive Maintenance API",
    description="Enterprise API providing model inference (Remaining Useful Life), OEE KPIs, and RAG Troubleshooting Copilot.",
    version="1.0.0"
)

# CORS middleware to allow connection from dashboard or other frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Industrial AI Predictive Maintenance Service",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/api/v1/predict (Requires POST)",
        "rag_assistant": "/api/v1/rag/query (Requires POST)"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
