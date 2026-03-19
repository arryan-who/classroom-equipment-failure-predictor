import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv("data/equipment_data.csv")

st.set_page_config(layout="wide")

# Title
st.title("📦 Classroom Equipment Risk Intelligence System")

# Sidebar - Equipment Selection
equipment = st.sidebar.selectbox(
    "Select Equipment",
    ["Projector", "Smartboard", "Lighting", "AC"]
)

# Load model
model = pd.read_pickle(f"models/{equipment.lower()}_model.pkl")

# KPI Section
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Failure Rate", f"{df['failure_within_30_days'].mean()*100:.2f}%")

with col3:
    st.metric("Equipment Types", df['equipment_type'].nunique())

st.divider()

# Layout
left, right = st.columns([1, 2])

# =========================
# LEFT: INPUTS
# =========================
with left:
    st.subheader("⚙️ Input Parameters")

    age = st.slider("Equipment Age (years)", 0, 10)
    usage = st.slider("Daily Usage Hours", 0, 12)
    maintenance = st.slider("Maintenance Gap (days)", 0, 60)
    power = st.slider("Power Fluctuations", 0, 20)

    input_data = {
        "equipment_age_years": age,
        "daily_usage_hours": usage,
        "maintenance_gap_days": maintenance,
        "power_fluctuation_events": power
    }

    if equipment == "Projector":
        temp = st.slider("Room Temperature", 20, 40)
        hours = st.slider("Operating Hours", 0, 5000)
        filter_gap = st.slider("Filter Cleaning Gap", 0, 60)

        input_data.update({
            "room_temperature": temp,
            "projector_operating_hours": hours,
            "filter_cleaning_gap_days": filter_gap
        })

    elif equipment == "Smartboard":
        error = st.slider("Touch Error Rate", 0.0, 1.0)
        firmware = st.slider("Firmware Gap", 0, 365)

        input_data.update({
            "touch_error_rate": error,
            "firmware_update_gap_days": firmware
        })

    elif equipment == "Lighting":
        cycles = st.slider("Switch Cycles", 0, 50)
        voltage = st.slider("Voltage Variation", 0, 20)

        input_data.update({
            "switch_cycles_per_day": cycles,
            "voltage_variation_events": voltage
        })

    elif equipment == "AC":
        temp = st.slider("Room Temperature", 20, 40)
        diff = st.slider("Temperature Difference", 0, 20)
        occ = st.slider("Room Occupancy", 0, 100)
        filter_gap = st.slider("Filter Cleaning Gap", 0, 60)

        input_data.update({
            "room_temperature": temp,
            "temperature_difference": diff,
            "room_occupancy": occ,
            "filter_cleaning_gap_days": filter_gap
        })

    input_df = pd.DataFrame([input_data])

    # Prediction
    prob = model.predict_proba(input_df)[0][1]
    risk = prob * 100

    confidence = abs(prob - 0.5) * 2 * 100

# =========================
# RIGHT: OUTPUTS
# =========================
with right:
    st.subheader("📊 Risk Analysis")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={"text": "Failure Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "color": "green"},
                {"range": [40, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Decision Message
    if risk < 40:
        st.success("🟢 Low Risk — Safe")
    elif risk < 70:
        st.warning("🟡 Medium Risk — Monitor")
    else:
        st.error("🔴 High Risk — Immediate Maintenance")

    st.info(f"Model Confidence: {confidence:.2f}%")

    # What-if Simulation
    st.subheader("🔁 What-if Analysis")

    new_gap = st.slider("Adjust Maintenance Gap", 0, 60)

    input_df_sim = input_df.copy()
    input_df_sim["maintenance_gap_days"] = new_gap

    new_prob = model.predict_proba(input_df_sim)[0][1]
    new_risk = new_prob * 100

    st.write(f"Old Risk: {risk:.2f}%")
    st.write(f"New Risk: {new_risk:.2f}%")

# =========================
# ANALYTICS
# =========================
st.subheader("📈 Data Insights")

fig1 = px.scatter(
    df,
    x="daily_usage_hours",
    y="maintenance_gap_days",
    color="equipment_type"
)

st.plotly_chart(fig1, use_container_width=True)

fig2 = px.histogram(df, x="failure_within_30_days", color="equipment_type")
st.plotly_chart(fig2, use_container_width=True)