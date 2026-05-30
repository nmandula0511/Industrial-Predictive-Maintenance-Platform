import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import sys
import time
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config
from src.pipeline.data_loader import load_dataset

# Setup page layout
st.set_page_config(
    page_title="Industrial AI - Enterprise Fleet Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: #1E293B;
    }
    
    /* Sidebar styling */
    .css-1639ggc {
        background-color: #0F172A;
    }
    
    /* Metrics panel card */
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        border-color: #3B82F6;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748B;
        font-weight: 600;
    }
    
    /* Status indicators */
    .status-badge {
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
        font-size: 0.9rem;
    }
    .status-ok { background-color: #DCFCE7; color: #15803D; }
    .status-warning { background-color: #FEF9C3; color: #A16207; }
    .status-danger { background-color: #FEE2E2; color: #B91C1C; }
</style>
""", unsafe_allow_html=True)

# API endpoint URL
API_URL = "http://localhost:8000/api/v1"

# Helper function to hit API or fallback locally
@st.cache_resource
def get_local_engine():
    try:
        from src.pipeline.inference import InferenceEngine
        return InferenceEngine()
    except Exception:
        return None

@st.cache_resource
def get_local_copilot():
    try:
        from rag.retrieval import MaintenanceCopilot
        return MaintenanceCopilot()
    except Exception:
        return None

def predict_rul_api(records_list):
    """Hits FastAPI endpoint to calculate RUL."""
    try:
        payload = {"records": records_list}
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        if res.status_code == 200:
            return res.json()["predictions"]
    except Exception:
        pass
    
    # Fallback to direct Python invocation if API is offline
    engine = get_local_engine()
    if engine:
        df = pd.DataFrame(records_list)
        # expand sensors dictionary for DataFrame representation
        sensor_rows = []
        for rec in records_list:
            row = {
                "engine_id": rec["engine_id"],
                "cycle": rec["cycle"],
                "op_setting_1": rec.get("op_setting_1", 0.0),
                "op_setting_2": rec.get("op_setting_2", 0.0),
                "op_setting_3": rec.get("op_setting_3", 100.0)
            }
            for k, v in rec["sensors"].items():
                row[k] = v
            # fill rest with 0
            for i in range(1, 22):
                s_name = f"sensor_{i}"
                if s_name not in row:
                    row[s_name] = 0.0
            sensor_rows.append(row)
        df_flat = pd.DataFrame(sensor_rows).sort_values(by=["engine_id", "cycle"])
        preds = engine.predict(df_flat)
        df_flat["predicted_RUL"] = preds
        latest = df_flat.groupby("engine_id").last().reset_index()
        return [
            {"engine_id": int(r["engine_id"]), "cycle": int(r["cycle"]), "predicted_RUL": float(r["predicted_RUL"])}
            for _, r in latest.iterrows()
        ]
    return None

def calculate_oee_api(predicted_rul, cycle, sensor_11_value=None):
    """Hits FastAPI endpoint to compute OEE metrics."""
    try:
        payload = {
            "predicted_RUL": predicted_rul,
            "cycle": cycle,
            "sensor_11_value": sensor_11_value
        }
        res = requests.post(f"{API_URL}/oee", json=payload, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
        
    # Local calculations if API is offline
    availability = 1.0 if predicted_rul >= Config.RUL_CRITICAL_THRESHOLD else max(0.0, predicted_rul / Config.RUL_CRITICAL_THRESHOLD)
    performance = min(1.0, cycle / Config.OEE_BASELINE_CYCLE)
    if sensor_11_value is None or sensor_11_value <= 480.0:
        quality = 1.0
    else:
        degradation = (sensor_11_value - 480.0) / (Config.SENSOR_11_STABLE_MAX - 480.0)
        quality = max(0.0, 1.0 - degradation)
    return {
        "availability": availability,
        "performance": performance,
        "quality": quality,
        "oee": availability * performance * quality
    }

def query_rag_api(query_str):
    """Queries the vector manual using API or local fallback."""
    try:
        payload = {"query": query_str}
        res = requests.post(f"{API_URL}/rag/query", json=payload, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
        
    copilot = get_local_copilot()
    if copilot:
        return copilot.query(query_str)
        
    return {
        "answer": "⚠️ Connection to RAG API and local engine both failed. Ensure manual has been indexed.",
        "mode": "Error",
        "sources": []
    }

# -----------------
# APP HEADER
# -----------------
st.write(
    """
    <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 20px;'>
        <span style='font-size: 3rem;'>🏭</span>
        <div>
            <h1 style='margin: 0; padding: 0; line-height: 1.1;'>Enterprise Fleet Control Center</h1>
            <p style='margin: 0; color: #64748B; font-weight: 500;'>Real-time AI Prognostics & SCADA Control Room</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------
# LOAD DATASET
# -----------------
@st.cache_data
def get_cached_test_data():
    try:
        return load_dataset(Config.TEST_DATA_PATH)
    except Exception:
        return None

test_data = get_cached_test_data()

# Check for large simulated dataset
large_data_path = os.path.join(Config.DATA_DIR, "fleet_telemetry_large.csv")

# Load large data downcasted to save RAM
@st.cache_data
def get_cached_large_data():
    if os.path.exists(large_data_path):
        try:
            df = pd.read_csv(large_data_path)
            # downcast numeric types to keep memory usage low
            for col in df.columns:
                if df[col].dtype == np.float64:
                    df[col] = df[col].astype(np.float32)
                elif df[col].dtype == np.int64:
                    df[col] = df[col].astype(np.int32)
            return df
        except Exception:
            return None
    return None

large_data = get_cached_large_data()

# -----------------
# SIDEBAR CONTROLS
# -----------------
st.sidebar.header("⚙️ Fleet Selection")

if test_data is None:
    st.error("❌ Failed to load standard NASA test dataset. Please verify folder contents.")
    st.stop()

# Determine fleet source: NASA or Simulated Large
fleet_source = st.sidebar.radio(
    "Active Data Feed:",
    options=["NASA CMAPSS Dataset", "Large Simulated Fleet (100MB+)"],
    index=1 if large_data is not None else 0
)

# Active dataset mapping
if fleet_source == "Large Simulated Fleet (100MB+)":
    if large_data is None:
        st.sidebar.warning("Large telemetry dataset not found on disk.")
        active_df = test_data
    else:
        active_df = large_data
else:
    active_df = test_data

# Engine ID dropdown selector
engine_ids = sorted(active_df["engine_id"].unique())
selected_engine = st.sidebar.selectbox("Inspect Engine Unit:", engine_ids, index=0)

# Filter telemetry for selected engine
engine_df = active_df[active_df["engine_id"] == selected_engine].sort_values(by="cycle")
max_cycles = int(engine_df["cycle"].max())

# Selected cycle slider
selected_cycle = st.sidebar.slider(
    "Inspect Cycle Timeline:", 
    min_value=1, 
    max_value=max_cycles, 
    value=max_cycles
)

# Current engine slice up to the selected cycle
history_df = engine_df[engine_df["cycle"] <= selected_cycle]
current_state = history_df.iloc[-1]

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Inspector Stats")
st.sidebar.write(f"**Engine Unit:** #{selected_engine}")
st.sidebar.write(f"**Active Feed:** {'Simulated Large' if active_df is large_data else 'NASA CMAPSS'}")
st.sidebar.write(f"**Total Data Rows:** {len(active_df):,}")

# Create tabs
tab_fleet, tab1, tab2, tab3, tab4 = st.tabs([
    "🏢 Fleet Operations (SCADA)",
    "📈 Real-time Telemetry", 
    "🔮 Predictive RUL Analytics", 
    "📊 OEE Operational Metrics", 
    "🤖 AI Troubleshooting Assistant"
])

# ----------------------------------------------------
# TAB FLEET: FLEET OPERATIONS (SCADA)
# ----------------------------------------------------
with tab_fleet:
    st.subheader("🏢 Fleet Operations Center & Analytics")
    st.write("Aggregates prognostics across all running engines in the fleet to identify wear distributions and units at risk.")
    
    # Check if we need to prompt user to generate large dataset
    if large_data is None and fleet_source == "Large Simulated Fleet (100MB+)":
        st.info("💡 You have selected the Large Simulated Fleet feed, but `fleet_telemetry_large.csv` is not generated yet.")
        c_gen1, c_gen2 = st.columns([1, 3])
        with c_gen1:
            gen_size = st.selectbox("Fleet size (Engines):", [100, 500, 1000], index=1)
            if st.button("Generate Large Telemetry File", type="primary"):
                with st.spinner("Simulating engine telemetry, noise, and failures (this can take 5-10s)..."):
                    from src.pipeline.generator import generate_fleet_telemetry
                    generate_fleet_telemetry(num_engines=gen_size, output_path=large_data_path)
                    st.success("Telemetry generated successfully! Refreshing dashboard...")
                    st.cache_data.clear()
                    time.sleep(2)
                    st.rerun()
        with c_gen2:
            st.markdown(
                """
                ### 🏗️ Why Generate Simulated Fleet Data?
                1. **Scale**: Generates **500,000+ rows** and **100MB+** of data to showcase big data handling capability to recruiters.
                2. **Analytics**: Populates this SCADA screen with realistic operational distributions and failure risks across hundreds of units.
                3. **Prognostics**: Simulates complex drift models, showcasing standard scaling and RandomForest accuracy at scale.
                """
            )
    else:
        # Load and run fleet summary calculations
        with st.spinner("Aggregates fleet status predictions..."):
            # For fleet summary, we grab the latest cycle for all engines
            fleet_latest = active_df.groupby("engine_id").last().reset_index()
            total_fleet_units = len(fleet_latest)
            
            # Predict RUL for all latest states using local engine (fast batch run)
            engine = get_local_engine()
            
            if engine is None:
                st.warning("⚠️ ML Model not found. Run model training (`python src/pipeline/train.py --dataset large`) to see live prognostics.")
                # fallback mock RULs
                fleet_latest["predicted_RUL"] = 120.0
            else:
                # Engineering feature set for all engines' latest cycle
                # To do this safely and quickly, we engineer features on the active dataframe, then group by engine and grab the last row
                engineered_df = engine.fe.transform(active_df)
                # Keep index matching active_df to merge engine_id
                engineered_df["engine_id"] = active_df["engine_id"]
                engineered_df["cycle"] = active_df["cycle"]
                for s in Config.SELECTED_SENSORS:
                    engineered_df[s] = active_df[s] # keep raw sensors for quality
                    
                # Last records
                latest_engineered = engineered_df.groupby("engine_id").last().reset_index()
                
                # Run batch predictions
                preds = engine.model.predict(latest_engineered[engine.fe.feature_cols])
                latest_engineered["predicted_RUL"] = np.clip(preds, 0, None)
                fleet_latest = latest_engineered
            
            # Calculate OEE for each engine latest cycle
            oee_list = []
            for _, r in fleet_latest.iterrows():
                oee_m = calculate_oee_api(
                    predicted_rul=r["predicted_RUL"],
                    cycle=int(r["cycle"]),
                    sensor_11_value=float(r["sensor_11"]) if "sensor_11" in r else None
                )
                oee_list.append(oee_m)
            
            oee_df = pd.DataFrame(oee_list)
            fleet_latest["OEE"] = oee_df["oee"].values
            fleet_latest["Availability"] = oee_df["availability"].values
            fleet_latest["Performance"] = oee_df["performance"].values
            fleet_latest["Quality"] = oee_df["quality"].values
            
            # Categorize health states
            def get_health(rul):
                if rul >= 50: return "Nominal"
                elif rul >= 30: return "Warning"
                else: return "Critical"
            
            fleet_latest["Health"] = fleet_latest["predicted_RUL"].apply(get_health)
            
            # Metric aggregations
            crit_count = sum(fleet_latest["Health"] == "Critical")
            warn_count = sum(fleet_latest["Health"] == "Warning")
            nominal_count = sum(fleet_latest["Health"] == "Nominal")
            avg_oee = fleet_latest["OEE"].mean() * 100
            
        # Top KPI Rows
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Fleet Active Units</div>
                    <div class='metric-value' style='color: #1E293B;'>{total_fleet_units} Engines</div>
                    <p style='margin: 0; font-size: 0.9rem; color: #64748B;'>Monitoring total active components</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with col_f2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Average Fleet OEE</div>
                    <div class='metric-value' style='color: #2563EB;'>{avg_oee:.1f}%</div>
                    <p style='margin: 0; font-size: 0.9rem; color: #64748B;'>Combined plant effectiveness index</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with col_f3:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Critical Alarms (RUL < 30)</div>
                    <div class='metric-value' style='color: #B91C1C;'>{crit_count} Units</div>
                    <p style='margin: 0; font-size: 0.9rem; color: #64748B;'>Require immediate maintenance action</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with col_f4:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Warning State (RUL 30-50)</div>
                    <div class='metric-value' style='color: #A16207;'>{warn_count} Units</div>
                    <p style='margin: 0; font-size: 0.9rem; color: #64748B;'>Overhaul inspections scheduled</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        # Fleet Analytics Visualizations
        st.write(" ")
        st.markdown("### 📊 Fleet Diagnostics Analytics")
        col_vis1, col_vis2 = st.columns(2)
        
        with col_vis1:
            # RUL distribution histogram
            fig_hist = px.histogram(
                fleet_latest, x="predicted_RUL", color="Health",
                color_discrete_map={"Nominal": "#22C55E", "Warning": "#F59E0B", "Critical": "#EF4444"},
                title="Prognostic Wear Distribution (Remaining Useful Life)",
                labels={"predicted_RUL": "Predicted Remaining Useful Life (Cycles)"},
                template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_vis2:
            # Sensor scatter plotting relationships: Sensor 9 (HPC Speed) vs Sensor 11 (Heat Temp)
            fig_scat = px.scatter(
                fleet_latest, x="sensor_11", y="sensor_9", color="Health",
                color_discrete_map={"Nominal": "#22C55E", "Warning": "#F59E0B", "Critical": "#EF4444"},
                size="cycle", hover_data=["engine_id", "predicted_RUL", "OEE"],
                title="Fleet Thermal Speed Anomalies Map",
                labels={"sensor_11": "Sensor 11 Core Temp", "sensor_9": "Sensor 9 HPC Speed"},
                template="plotly_white"
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            
        # Searchable table
        st.markdown("### 🏢 Fleet Status Registry")
        st.write("Search and sort through engine diagnostics logs. Click on the sidebar selector to drill down into anomalous components.")
        
        # Search inputs
        search_col1, search_col2 = st.columns([1, 2])
        with search_col1:
            health_filter = st.selectbox("Filter by Health:", ["All", "Nominal", "Warning", "Critical"], index=0)
        with search_col2:
            search_query = st.text_input("Search Engine Unit ID:", value="", placeholder="Type Engine ID...")
            
        # Filter table
        table_df = fleet_latest[["engine_id", "cycle", "predicted_RUL", "OEE", "Availability", "Performance", "Quality", "Health"]].copy()
        
        # Format columns for display
        table_df["OEE"] = (table_df["OEE"] * 100).map("{:.1f}%".format)
        table_df["Availability"] = (table_df["Availability"] * 100).map("{:.1f}%".format)
        table_df["Performance"] = (table_df["Performance"] * 100).map("{:.1f}%".format)
        table_df["Quality"] = (table_df["Quality"] * 100).map("{:.1f}%".format)
        table_df["predicted_RUL"] = table_df["predicted_RUL"].map("{:.1f} cycles".format)
        
        if health_filter != "All":
            table_df = table_df[table_df["Health"] == health_filter]
        if search_query.strip():
            try:
                e_id = int(search_query.strip())
                table_df = table_df[table_df["engine_id"] == e_id]
            except ValueError:
                st.warning("Please type a valid numerical Engine ID.")
                
        st.dataframe(table_df, use_container_width=True, hide_index=True)

# ----------------------------------------------------
# TAB 1: REAL-TIME TELEMETRY
# ----------------------------------------------------
with tab1:
    st.subheader("📊 Live Sensor Telemetry Charts")
    st.write("Track multi-sensor signals over historical operational cycles. Anomalies are highlighted.")
    
    # Selection of sensors to chart
    chart_sensors = st.multiselect(
        "Choose Sensors to Graph:", 
        options=Config.SELECTED_SENSORS,
        default=["sensor_11", "sensor_4", "sensor_9"]
    )
    
    if not chart_sensors:
        st.info("Please select at least one sensor to visualize.")
    else:
        for sensor in chart_sensors:
            fig = px.line(
                history_df, 
                x="cycle", 
                y=sensor, 
                title=f"{sensor.upper()} Readings over Cycles",
                labels={"cycle": "Cycle Count", sensor: "Sensor Value"},
                template="plotly_white"
            )
            fig.update_traces(line_color="#2563EB", line_width=2)
            fig.add_shape(
                type="line", line_dash="dash", line_color="red",
                x0=selected_cycle, y0=history_df[sensor].min(),
                x1=selected_cycle, y1=history_df[sensor].max(),
                name="Current Cycle"
            )
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# TAB 2: PREDICTIVE RUL ANALYTICS
# ----------------------------------------------------
with tab2:
    st.subheader("🔮 Predictive Analytics (Remaining Useful Life)")
    st.write("Calculates health trends and forecasts remaining operational lifespan before critical failure.")
    
    # 1. Format payload for API
    records_payload = []
    # Include history up to selected cycle so rolling statistics can be calculated properly
    for _, row in history_df.iterrows():
        sensors_dict = {s: float(row[s]) for s in Config.ALL_COLUMNS if "sensor" in s}
        records_payload.append({
            "engine_id": int(row["engine_id"]),
            "cycle": int(row["cycle"]),
            "op_setting_1": float(row["op_setting_1"]),
            "op_setting_2": float(row["op_setting_2"]),
            "op_setting_3": float(row["op_setting_3"]),
            "sensors": sensors_dict
        })
        
    with st.spinner("Calculating RUL using machine learning model..."):
        preds_list = predict_rul_api(records_payload)
        
    if preds_list is None:
        st.error("⚠️ Predictive model not available. Make sure to train the model first by running `python src/pipeline/train.py`.")
    else:
        pred_rul = preds_list[0]["predicted_RUL"]
        
        # Determine status color
        if pred_rul >= 50:
            status_text = "NOMINAL HEALTH"
            status_class = "status-ok"
            status_color = "#15803D"
        elif pred_rul >= 30:
            status_text = "MAINTENANCE REQUIRED"
            status_class = "status-warning"
            status_color = "#A16207"
        else:
            status_text = "CRITICAL FAILURE RISK"
            status_class = "status-danger"
            status_color = "#B91C1C"
            
        # Top Cards Row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Predicted Remaining Useful Life</div>
                    <div class='metric-value' style='color: {status_color};'>{pred_rul:.1f} Cycles</div>
                    <p style='margin: 0; font-size: 0.9rem; color: #64748B;'>Estimated cycles remaining before overhaul</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>System Status</div>
                    <div class='metric-value' style='color: {status_color}; font-size: 1.6rem; padding: 10px 0;'>
                        <span class='status-badge {status_class}'>{status_text}</span>
                    </div>
                    <p style='margin: 0; font-size: 0.9rem; color: #64748B;'>Based on predictive wear calculations</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Total Cycle Count</div>
                    <div class='metric-value' style='color: #1E293B;'>{selected_cycle} Cycles</div>
                    <p style='margin: 0; font-size: 0.9rem; color: #64748B;'>Accumulated flight/operation cycles</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        # RUL Degradation Chart
        st.markdown("### 📉 Prognostics Trend")
        # Build RUL degradation timeline for simulated history
        simulated_ruls = []
        # To avoid lagging, we predict at steps of 10 cycles (faster for UI responsiveness)
        cycles_to_eval = sorted(list(set(list(range(10, len(history_df) + 1, 10)) + [len(history_df)])))
        
        with st.spinner("Compiling degradation curve..."):
            for cyc in cycles_to_eval:
                sub_history = history_df.iloc[:cyc]
                sub_payload = []
                for _, row in sub_history.iterrows():
                    sensors_dict = {s: float(row[s]) for s in Config.ALL_COLUMNS if "sensor" in s}
                    sub_payload.append({
                        "engine_id": int(row["engine_id"]),
                        "cycle": int(row["cycle"]),
                        "op_setting_1": float(row["op_setting_1"]),
                        "op_setting_2": float(row["op_setting_2"]),
                        "op_setting_3": float(row["op_setting_3"]),
                        "sensors": sensors_dict
                    })
                sub_preds = predict_rul_api(sub_payload)
                if sub_preds:
                    simulated_ruls.append({"cycle": cyc, "predicted_RUL": sub_preds[0]["predicted_RUL"]})
                    
        if simulated_ruls:
            rul_trend_df = pd.DataFrame(simulated_ruls)
            fig_trend = px.line(
                rul_trend_df, x="cycle", y="predicted_RUL", 
                title="Predicted RUL Trend Over Time",
                labels={"cycle": "Cycle Count", "predicted_RUL": "Predicted RUL"},
                template="plotly_white"
            )
            fig_trend.update_traces(line_color=status_color, line_width=3)
            # Add critical line
            fig_trend.add_shape(
                type="line", line_dash="dot", line_color="orange",
                x0=0, y0=Config.RUL_CRITICAL_THRESHOLD,
                x1=max(cycles_to_eval), y1=Config.RUL_CRITICAL_THRESHOLD,
                name="Maintenance Threshold"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

# ----------------------------------------------------
# TAB 3: OEE OPERATIONAL METRICS
# ----------------------------------------------------
with tab3:
    st.subheader("📊 Overall Equipment Effectiveness (OEE) Analysis")
    st.write("Calculates performance, quality, and availability factors mapping AI predictions directly to manufacturing metrics.")
    
    # We fetch predicted RUL from the previous tab (if calculated)
    # If not, we set it to nominal
    current_rul = 150.0
    if 'pred_rul' in locals():
        current_rul = pred_rul
        
    s11_val = float(current_state["sensor_11"])
    
    # Pull metrics
    oee_metrics = calculate_oee_api(
        predicted_rul=current_rul,
        cycle=int(selected_cycle),
        sensor_11_value=s11_val
    )
    
    avail = oee_metrics["availability"]
    perf = oee_metrics["performance"]
    qual = oee_metrics["quality"]
    oee = oee_metrics["oee"]
    
    # Multi columns gauge layout
    col1, col2, col3, col4 = st.columns(4)
    
    def create_gauge(title, val, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 18, 'family': 'Outfit'}},
            number={'suffix': "%", 'font': {'size': 26, 'family': 'Space Grotesk'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.1)'},
                    {'range': [60, 85], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [85, 100], 'color': 'rgba(34, 197, 94, 0.1)'}
                ],
            }
        ))
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
        return fig
        
    with col1:
        st.plotly_chart(create_gauge("Availability", avail, "#E11D48"), use_container_width=True)
        st.write(f"<p style='text-align: center; color: #64748B; font-size: 0.85rem;'>Downtime Risk: {'CRITICAL' if avail < 1.0 else 'NOMINAL'}</p>", unsafe_allow_html=True)
    with col2:
        st.plotly_chart(create_gauge("Performance", perf, "#F59E0B"), use_container_width=True)
        st.write(f"<p style='text-align: center; color: #64748B; font-size: 0.85rem;'>Lifetime Used: {perf*100:.1f}%</p>", unsafe_allow_html=True)
    with col3:
        st.plotly_chart(create_gauge("Quality Index", qual, "#10B981"), use_container_width=True)
        st.write(f"<p style='text-align: center; color: #64748B; font-size: 0.85rem;'>Core Heat: {s11_val:.2f}°C</p>", unsafe_allow_html=True)
    with col4:
        st.plotly_chart(create_gauge("Overall OEE", oee, "#2563EB"), use_container_width=True)
        st.write(f"<p style='text-align: center; color: #64748B; font-size: 0.85rem;'>Combined Performance Indicator</p>", unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 4: AI TROUBLESHOOTING ASSISTANT
# ----------------------------------------------------
with tab4:
    st.subheader("🤖 AI Maintenance Troubleshooting Copilot (RAG)")
    st.write("Query standard operating manuals and retrieve targeted troubleshooting checklists based on current telemetry warnings.")
    
    # Quick query buttons
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    q_str = ""
    with c_btn1:
        if st.button("🔍 Check Sensor 11 Warning Protocol"):
            q_str = "How to troubleshoot Sensor 11 warning?"
    with c_btn2:
        if st.button("🔍 Check LPC Temperature Alert (Sensor 4)"):
            q_str = "What is the troubleshooting protocol for high Sensor 4 temperature?"
    with c_btn3:
        if st.button("🔍 Check HPC Rotor Instability (Sensor 9)"):
            q_str = "Troubleshoot Sensor 9 HPC Speed spikes."
            
    user_query = st.text_input(
        "Ask a specific question about engine components or sensors:", 
        value=q_str if q_str else "How to handle high turbine temperature?",
        placeholder="Type query here (e.g. Parts needed for Sensor 11 thermocouple overhaul...)"
    )
    
    if st.button("Get Maintenance Checklists", type="primary"):
        with st.spinner("Searching document indexes..."):
            rag_res = query_rag_api(user_query)
            
        st.markdown("### 🤖 System Response")
        st.markdown(rag_res["answer"])
        
        with st.expander("📚 Reference Sources (Manual Chunks)"):
            for i, src in enumerate(rag_res["sources"], 1):
                st.markdown(f"**Chunk #{i}:**")
                st.code(src, language="markdown")
