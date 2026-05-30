import os
import sys
from fastapi.testclient import TestClient
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.main import app
from src.api.routes import get_inference_engine, get_rag_copilot

client = TestClient(app)

# 1. Create mock dependencies for testing API in isolation
class MockInferenceEngine:
    def predict(self, raw_df):
        # Return mock predictions of 50 cycles for every row
        return np.array([50.0] * len(raw_df))

class MockRagCopilot:
    def query(self, user_query):
        return {
            "query": user_query,
            "answer": "Mocked RAG response for testing.",
            "mode": "Offline-retrieval",
            "sources": ["Mock source content"]
        }

import unittest

class TestApiEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Override FastAPI dependencies
        app.dependency_overrides[get_inference_engine] = lambda: MockInferenceEngine()
        app.dependency_overrides[get_rag_copilot] = lambda: MockRagCopilot()

    @classmethod
    def tearDownClass(cls):
        # Clear dependencies after test run
        app.dependency_overrides.clear()

    def test_read_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "online")
        self.assertIn("docs_url", json_data)

    def test_oee_calculation_nominal(self):
        payload = {
            "predicted_RUL": 40.0,
            "cycle": 150,
            "sensor_11_value": 470.0
        }
        response = client.post("/api/v1/oee", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # RUL >= 30, so Availability should be 1.0
        self.assertEqual(data["availability"], 1.0)
        # Performance = 150 / 300 = 0.5
        self.assertEqual(data["performance"], 0.5)
        # Sensor 11 <= 480, so Quality should be 1.0
        self.assertEqual(data["quality"], 1.0)
        # OEE = 1.0 * 0.5 * 1.0 = 0.5
        self.assertEqual(data["oee"], 0.5)

    def test_oee_calculation_degraded(self):
        payload = {
            "predicted_RUL": 15.0, # Below 30
            "cycle": 90,
            "sensor_11_value": 502.5 # Between 480 and 525 (degraded)
        }
        response = client.post("/api/v1/oee", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Availability = 15 / 30 = 0.5
        self.assertEqual(data["availability"], 0.5)
        # Performance = 90 / 300 = 0.3
        self.assertEqual(data["performance"], 0.3)
        # Quality = 1.0 - (502.5 - 480) / (525 - 480) = 1.0 - 22.5 / 45 = 0.5
        self.assertEqual(data["quality"], 0.5)
        # OEE = 0.5 * 0.3 * 0.5 = 0.075
        self.assertEqual(data["oee"], 0.075)

    def test_predict_endpoint(self):
        payload = {
            "records": [
                {
                    "engine_id": 1,
                    "cycle": 10,
                    "sensors": {
                        "sensor_2": 642.0,
                        "sensor_3": 1580.0,
                        "sensor_4": 1400.0,
                        "sensor_11": 47.0
                    }
                }
            ]
        }
        response = client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 1)
        self.assertEqual(data["predictions"][0]["engine_id"], 1)
        self.assertEqual(data["predictions"][0]["predicted_RUL"], 50.0) # From MockInferenceEngine

    def test_rag_query_endpoint(self):
        payload = {"query": "How to fix sensor 11?"}
        response = client.post("/api/v1/rag/query", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "How to fix sensor 11?")
        self.assertIn("answer", data)
        self.assertEqual(data["mode"], "Offline-retrieval")
        self.assertTrue(len(data["sources"]) > 0)

