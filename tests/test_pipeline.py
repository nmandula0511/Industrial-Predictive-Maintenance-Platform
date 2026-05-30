import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.pipeline.feature_engineering import FeatureEngineer

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Create a mock dataframe that mirrors CMAPSS columns
        self.mock_data = pd.DataFrame({
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "op_setting_1": [0.0] * 5,
            "op_setting_2": [0.0] * 5,
            "op_setting_3": [100.0] * 5
        })
        # Add values for all sensors
        for i in range(1, 22):
            self.mock_data[f"sensor_{i}"] = np.linspace(10.0, 50.0, 5)
            
        self.fe = FeatureEngineer(window_size=2)

    def test_add_rul(self):
        # Test target RUL calculation
        df_rul = self.fe.add_rul(self.mock_data)
        
        # For engine 1, max cycle is 3, so RULs should be 3-1=2, 3-2=1, 3-3=0
        engine_1_ruls = df_rul[df_rul["engine_id"] == 1]["RUL"].values.tolist()
        self.assertEqual(engine_1_ruls, [2, 1, 0])
        
        # For engine 2, max cycle is 2, so RULs should be 2-1=1, 2-2=0
        engine_2_ruls = df_rul[df_rul["engine_id"] == 2]["RUL"].values.tolist()
        self.assertEqual(engine_2_ruls, [1, 0])

    def test_compute_rolling_features(self):
        df_roll = self.fe.compute_rolling_features(self.mock_data)
        
        # Verify that rolling columns are created
        for sensor in Config.SELECTED_SENSORS:
            self.assertIn(f"{sensor}_roll_mean", df_roll.columns)
            self.assertIn(f"{sensor}_roll_std", df_roll.columns)

        # Check calculation for engine 1 sensor_2 (which has values 10, 20, 30 for cycles 1, 2, 3)
        # We set window_size=2, so cycle 2 rolling mean of sensor_2 should be (10+20)/2 = 15.0
        engine_1_roll = df_roll[df_roll["engine_id"] == 1]
        
        # Values of sensor_2: index 0 -> 10.0, index 1 -> 20.0
        # Check mean at cycle 2
        val_mean = engine_1_roll.iloc[1]["sensor_2_roll_mean"]
        self.assertEqual(val_mean, 15.0)

    def test_fit_transform(self):
        X, y = self.fe.fit_transform(self.mock_data)
        
        # Verify output shapes and properties
        self.assertEqual(len(X), len(self.mock_data))
        self.assertEqual(len(y), len(self.mock_data))
        
        # Scale test: columns should be centered around mean 0 (variance 1)
        # Note: with tiny datasets, std might be zero for constant cols which results in 0.0 after fillna/scaling
        self.assertTrue(np.all(np.isfinite(X.values)))

if __name__ == "__main__":
    unittest.main()
