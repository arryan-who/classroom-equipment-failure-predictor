import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/equipment_data.csv")

# Normalize column name (safety)
df.columns = df.columns.str.strip()

if "equipment_type" not in df.columns:
    raise ValueError("❌ 'equipment_type' column not found in dataset")

# Normalize values (CRITICAL FIX)
df["equipment_type"] = df["equipment_type"].str.strip().str.lower()

print("\n📊 Available equipment types in dataset:")
print(df["equipment_type"].value_counts())

# =========================
# FEATURE DEFINITIONS
# =========================
FEATURES = {
    "projector": [
        "equipment_age_years", "daily_usage_hours", "maintenance_gap_days",
        "power_fluctuation_events", "room_temperature",
        "projector_operating_hours", "filter_cleaning_gap_days"
    ],
    "smartboard": [
        "equipment_age_years", "daily_usage_hours", "maintenance_gap_days",
        "power_fluctuation_events", "touch_error_rate",
        "firmware_update_gap_days"
    ],
    "lighting": [
        "equipment_age_years", "daily_usage_hours", "maintenance_gap_days",
        "power_fluctuation_events", "switch_cycles_per_day",
        "voltage_variation_events"
    ],
    "ac": [
        "equipment_age_years", "daily_usage_hours", "maintenance_gap_days",
        "power_fluctuation_events", "room_temperature",
        "temperature_difference", "room_occupancy",
        "filter_cleaning_gap_days"
    ]
}

TARGET = "failure_within_30_days"

# =========================
# SETUP
# =========================
os.makedirs("models", exist_ok=True)
history = []

# =========================
# TRAIN MODELS
# =========================
for equipment, features in FEATURES.items():
    print(f"\n🔧 Training model for: {equipment.upper()}")

    data = df[df["equipment_type"] == equipment].copy()

    # SAFETY CHECK
    if len(data) == 0:
        print(f"⚠️ No data found for '{equipment}', skipping...")
        continue

    # Ensure required features exist
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"⚠️ Missing features for {equipment}: {missing_features}, skipping...")
        continue

    X = data[features]
    y = data[TARGET]

    # Another safety check
    if len(X) < 5:
        print(f"⚠️ Not enough data for {equipment}, skipping...")
        continue

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"✅ Accuracy: {acc:.3f}")
    print(f"✅ Precision: {prec:.3f}")
    print(f"✅ Recall: {rec:.3f}")
    print(f"✅ F1 Score: {f1:.3f}")

    # Save model
    model_path = f"models/{equipment}_model.pkl"
    pd.to_pickle(model, model_path)

    # Save history
    history.append({
        "equipment": equipment,
        "features_used": features,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# =========================
# SAVE HISTORY
# =========================
history_path = "models/model_history.json"

if os.path.exists(history_path):
    with open(history_path, "r") as f:
        existing = json.load(f)
else:
    existing = []

existing.extend(history)

with open(history_path, "w") as f:
    json.dump(existing, f, indent=4)

print("\n🎯 Training complete.")