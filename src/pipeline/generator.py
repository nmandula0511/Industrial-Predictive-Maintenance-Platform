import pandas as pd
import numpy as np
import os
import argparse
import time

def generate_fleet_telemetry(num_engines: int, output_path: str):
    print("=" * 60)
    print(f"STARTING FLEET TELEMETRY SIMULATOR")
    print(f"Target Units: {num_engines} engines")
    print(f"Output File:  {output_path}")
    print("=" * 60)
    
    start_time = time.time()
    np.random.seed(42)
    
    # Base stats for CMAPSS-like sensors (mean and noise std)
    # Mapping sensor index (1 to 21) -> (base_mean, base_std, drift_direction, drift_magnitude)
    # Drift direction: 1 for up, -1 for down, 0 for stable
    sensor_profiles = {
        1: (518.67, 0.0, 0, 0.0),
        2: (641.82, 0.5, 1, 3.5),
        3: (1589.65, 5.0, 1, 25.0),
        4: (1400.60, 9.0, 1, 65.0),
        5: (14.62, 0.0, 0, 0.0),
        6: (21.61, 0.01, 1, 0.05),
        7: (554.30, 0.8, -1, -8.0),
        8: (2388.08, 0.07, 1, 0.25),
        9: (9044.03, 15.0, 1, 80.0),
        10: (1.30, 0.0, 0, 0.0),
        11: (47.41, 0.2, 1, 1.8),
        12: (521.60, 0.7, -1, -6.5),
        13: (2388.09, 0.07, 1, 0.26),
        14: (8138.50, 19.0, 1, 110.0),
        15: (8.41, 0.02, 1, 0.18),
        16: (0.03, 0.0, 0, 0.0),
        17: (392.00, 1.5, 1, 8.0),
        18: (2388.00, 0.0, 0, 0.0),
        19: (100.00, 0.0, 0, 0.0),
        20: (39.06, 0.2, -1, -1.5),
        21: (23.41, 0.1, -1, -0.9)
    }
    
    rows = []
    
    for engine_id in range(1, num_engines + 1):
        # Lifespan: each engine runs until failure, cycles range between 120 and 360
        lifespan = np.random.randint(120, 360)
        
        # Operational settings (flight profiles, throttle positions)
        op_set_1_base = np.random.choice([0.000, 10.00, 20.00, 35.00, 42.00])
        op_set_2_base = np.random.choice([0.000, 0.25, 0.70, 0.84])
        op_set_3_base = np.random.choice([100.0, 60.0, 80.0])
        
        for cycle in range(1, lifespan + 1):
            # Calculate wear index (0 to 1, where 1 is total wear/failure)
            # Wear model: exponential degradation curve
            wear_factor = (np.exp(3.0 * (cycle / lifespan)) - 1.0) / (np.exp(3.0) - 1.0)
            
            row = {
                "engine_id": engine_id,
                "cycle": cycle,
                "op_setting_1": round(op_set_1_base + np.random.normal(0, 0.001), 4),
                "op_setting_2": round(op_set_2_base + np.random.normal(0, 0.001), 4),
                "op_setting_3": round(op_set_3_base, 1)
            }
            
            # Generate sensor readings based on wear profiles
            for s_idx, (mean, std, drift_dir, drift_mag) in sensor_profiles.items():
                sensor_name = f"sensor_{s_idx}"
                noise = np.random.normal(0, std) if std > 0 else 0.0
                
                # Apply wear degradation drift
                drift = drift_dir * drift_mag * wear_factor
                
                # Dynamic sensor value
                s_val = mean + drift + noise
                
                # Cap minimums to avoid negative readings on positive physical indicators
                if s_idx not in [7, 12, 20, 21]:
                    s_val = max(0.0, s_val)
                    
                row[sensor_name] = round(s_val, 4)
                
            rows.append(row)
            
    # Convert to DataFrame
    print("[*] Materializing DataFrame...")
    df = pd.DataFrame(rows)
    
    # Save to CSV
    print(f"[*] Writing to CSV file: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    elapsed_time = time.time() - start_time
    
    print("-" * 60)
    print("GENERATION STATISTICS:")
    print(f"  Total Rows:     {len(df):,}")
    print(f"  Columns:        {len(df.columns)}")
    print(f"  File Size:      {file_size_mb:.2f} MB")
    print(f"  Execution Time: {elapsed_time:.2f} seconds")
    print("=" * 60)
    print("TELEMETRY SIMULATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fleet Telemetry Data Simulator")
    parser.add_argument("--engines", type=int, default=500, help="Number of engine units to simulate")
    parser.add_argument("--output", type=str, default="data/fleet_telemetry_large.csv", help="Path to save output CSV")
    args = parser.parse_args()
    
    # Force default path inside project workspace if relative path is passed
    output_path = args.output
    if not os.path.isabs(output_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_path = os.path.join(base_dir, output_path)
        
    generate_fleet_telemetry(args.engines, output_path)
