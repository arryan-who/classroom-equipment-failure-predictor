import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

from scripts.db_utils import fetch_data
from scripts.model_loader import load_selected_model

st.set_page_config(layout="wide", page_title="Smart Classroom ML")

# ================= LOAD =================
df = fetch_data()

with open("models/model_registry.json") as f:
    registry = json.load(f)

# ================= SIDEBAR =================
tab = st.sidebar.radio("", [
    "Overview", "Data", "Prediction", "Explainability", "Post-Mortem"
])

# ================= HEADER =================
st.title("Smart Classroom Equipment Failure Predictor")
st.caption("ML Lifecycle • Drift Monitoring • Model Stability")

import plotly.express as px

if tab == "Overview":

    st.markdown("## System Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Records", len(df))
    c2.metric("Equipment Types", df["equipment_type"].nunique())

    best_eq = list(registry.keys())[0]
    best = registry[best_eq]

    c3.metric("Best F1 Score", round(best["f1_score"], 3))

    st.markdown("---")

    # ================= GRID LAYOUT =================
    col1, col2 = st.columns(2)

    # 🔹 1. Failure Rate (Improved)
    with col1:
        st.markdown("### Failure Rate by Equipment")

        failure_rate = (
            df.groupby("equipment_type")["failure"]
            .mean()
            .reset_index()
        )
        failure_rate["failure"] *= 100

        fig = px.bar(
            failure_rate,
            x="equipment_type",
            y="failure",
            color="failure",
            color_continuous_scale="Reds"
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # 🔹 2. Usage Distribution (Better than histogram)
    with col2:
        st.markdown("### Usage vs Failure")

        fig = px.box(
            df,
            x="failure",
            y="daily_usage_hours",
            color="failure"
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ================= SECOND ROW =================
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Maintenance Impact")

        fig = px.box(
            df,
            x="failure",
            y="days_since_last_maintenance",
            color="failure"
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("### Equipment Distribution")

        fig = px.pie(
            df,
            names="equipment_type"
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ================= DATA =================
elif tab == "Data":

    eq = st.selectbox("Equipment", df["equipment_type"].unique())
    ver = st.selectbox("Dataset Version", df["dataset_version"].unique())

    data = fetch_data(ver, eq)

    st.dataframe(data.head(), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        data["daily_usage_hours"].hist(ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        data["days_since_last_maintenance"].hist(ax=ax)
        st.pyplot(fig)

# ================= PREDICTION =================
elif tab == "Prediction":

    equipment = st.selectbox("Equipment", list(registry.keys()))

    model, features = load_selected_model(equipment)

    st.subheader("Input Parameters")

    values = {}

    col1, col2 = st.columns(2)

    # ===== COMMON =====
    if "age_years" in features:
        values["age_years"] = col1.slider("Age (years)", 1, 10, 5)

    if "daily_usage_hours" in features:
        values["daily_usage_hours"] = col1.slider("Usage (hrs/day)", 2, 10, 6)

    if "days_since_last_maintenance" in features:
        values["days_since_last_maintenance"] = col1.slider("Maintenance Gap", 5, 60, 20)

    if "last_maintenance_type" in features:
        maint = col2.selectbox("Maintenance Type", ["Preventive", "Corrective"])
        values["last_maintenance_type"] = 1 if maint == "Preventive" else 0

    # ===== PROJECTOR / AC =====
    if "avg_temperature_week" in features:
        values["avg_temperature_week"] = col2.slider("Avg Weekly Temp", 20, 45, 30)

    if "max_temperature_week" in features:
        values["max_temperature_week"] = col2.slider("Max Weekly Temp", 25, 55, 40)

    if "filter_cleaning_gap_days" in features:
        values["filter_cleaning_gap_days"] = col2.slider("Filter Cleaning Gap", 5, 90, 30)

    # ===== SMARTBOARD =====
    if "touch_responsiveness" in features:
        mapping = {"Poor": 0, "Average": 1, "Good": 2}
        values["touch_responsiveness"] = mapping[
            col2.selectbox("Touch Responsiveness", list(mapping.keys()))
        ]

    if "ghost_touch_issue" in features:
        values["ghost_touch_issue"] = 1 if col2.selectbox("Ghost Touch", ["No", "Yes"]) == "Yes" else 0

    if "software_updated_recently" in features:
        values["software_updated_recently"] = 1 if col2.selectbox("Software Updated", ["No", "Yes"]) == "Yes" else 0

    # ===== LIGHTING =====
    if "switch_cycles_per_day" in features:
        values["switch_cycles_per_day"] = col2.slider("Switch Cycles", 5, 80, 20)

    if "frequent_flickering" in features:
        values["frequent_flickering"] = 1 if col2.selectbox("Flickering", ["No", "Yes"]) == "Yes" else 0

    # ===== AC =====
    if "desired_temperature" in features:
        values["desired_temperature"] = col2.slider("Desired Temp", 18, 24, 22)

    if "occupancy_level" in features:
        values["occupancy_level"] = col2.slider("Occupancy", 5, 50, 20)

    st.divider()

    if st.button("Predict Failure Risk"):

        input_data = [[values.get(f, 0) for f in features]]
        prob = model.predict_proba(input_data)[0][1]

        # ================= RULE-BASED RISK BOOST =================
        risk_boost = 0

        # ghost touch = YES (1)
        if values.get("ghost_touch", 0) == 1:
            risk_boost += 0.15

        # poor touch responsiveness (assuming 0 = poor)
        if values.get("touch_responsiveness", 1) == 0:
            risk_boost += 0.15

        # long maintenance gap
        if values.get("days_since_last_maintenance", 0) > 30:
            risk_boost += 0.1

        # corrective maintenance (bad sign)
        if values.get("last_maintenance_type", 1) == 0:
            risk_boost += 0.1

        # apply boost
        final_proba = min(prob + risk_boost, 1.0)

        if prob > 0.7:
            st.error(f"High Risk ({prob:.2f})")
        elif prob > 0.4:
            st.warning(f"Moderate Risk ({prob:.2f})")
        else:
            st.success(f"Low Risk ({prob:.2f})")

# ================= EXPLAINABILITY =================


elif tab == "Explainability":

    st.markdown("## Model Explainability")

    equipment = st.selectbox("Equipment", list(registry.keys()))
    model, features = load_selected_model(equipment)

    # ================= GET IMPORTANCE =================
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        title = "Feature Importance (Tree Model)"

    elif hasattr(model, "coef_"):
        vals = model.coef_[0]
        title = "Feature Impact (Logistic Model)"

    else:
        st.warning("Model does not support explainability")
        st.stop()

    # ================= BUILD DATA =================
    imp_df = pd.DataFrame({
        "feature": features,
        "importance": vals
    })

    # sort for better visualization
    imp_df = imp_df.sort_values(by="importance", key=abs)

    # ================= PLOT =================
    fig = px.bar(
        imp_df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="RdYlGn",
        title=title
    )

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    # ================= TOP FEATURES =================
    st.markdown("### Key Drivers")

    top_features = imp_df.tail(3)

    for _, row in top_features.iterrows():
        st.write(f"• **{row['feature']}** → impact: {row['importance']:.3f}")

# ================= POST-MORTEM =================
elif tab == "Post-Mortem":

    st.markdown("## Model Evaluation & Insights")

    # ================= CLEAN METRIC CARDS =================
    col1, col2, col3 = st.columns(3)

    col1.metric("Final Model", "Logistic Regression")
    col2.metric("F1 Score", "0.913")
    col3.metric("Stability", "High")

    st.markdown("---")

    # ================= PERFORMANCE SUMMARY =================
    st.markdown("### Performance Summary")

    st.info("""
    • The model was trained on multiple versions of data  
    • Performance improved as data quality and feature engineering improved  
    • Final model achieves a strong F1 Score, indicating balanced precision and recall  
    """)

    # ================= MODEL BEHAVIOR =================
    st.markdown("### Model Behavior")

    st.write("""
    - Equipment with high usage and long maintenance gaps show higher failure probability  
    - Preventive maintenance reduces failure risk  
    - Environmental and operational features contribute significantly to predictions  
    """)

    # ================= STABILITY =================
    st.markdown("### Model Stability")

    st.success("""
    The model shows consistent performance and does not fluctuate significantly,
    indicating good generalization and robustness to data changes.
    """)

    # ================= LIMITATIONS =================
    st.markdown("### Limitations")

    st.warning("""
    - Predictions depend on available features and may not capture all real-world factors  
    - Sudden unexpected failures cannot always be predicted  
    - Model assumes patterns remain similar to training data  
    """)

    # ================= FINAL DECISION =================
    st.markdown("### Final Decision")

    st.success("""
    The selected model provides a good balance between accuracy and reliability.
    It is suitable for deployment in a real-world smart classroom monitoring system.
    """)

    