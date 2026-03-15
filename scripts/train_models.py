import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

# Load dataset
df = pd.read_csv("data/equipment_data.csv")

# Encode maintenance type
df["last_maintenance_type"] = df["last_maintenance_type"].map({
    "preventive": 0,
    "corrective": 1
})

# Feature sets per equipment
feature_sets = {

    "projector": [
        "equipment_age_years",
        "daily_usage_hours",
        "maintenance_gap_days",
        "last_maintenance_type",
        "room_temperature",
        "power_fluctuation_events",
        "projector_operating_hours",
        "filter_cleaning_gap_days"
    ],

    "smartboard": [
        "equipment_age_years",
        "daily_usage_hours",
        "maintenance_gap_days",
        "last_maintenance_type",
        "power_fluctuation_events",
        "touch_error_rate",
        "firmware_update_gap_days"
    ],

    "lighting": [
        "equipment_age_years",
        "daily_usage_hours",
        "maintenance_gap_days",
        "last_maintenance_type",
        "power_fluctuation_events",
        "switch_cycles_per_day",
        "voltage_variation_events"
    ],

    "ac": [
        "equipment_age_years",
        "daily_usage_hours",
        "maintenance_gap_days",
        "last_maintenance_type",
        "room_temperature",
        "temperature_difference",
        "room_occupancy",
        "filter_cleaning_gap_days"
    ]
}

target = "failure_within_30_days"

history = []

for equipment, features in feature_sets.items():

    subset = df[df["equipment_type"] == equipment]

    X = subset[features]
    y = subset[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Save model
    joblib.dump(model, f"models/{equipment}_model.pkl")

    history.append({
        "equipment": equipment,
        "features_used": features,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "date": str(datetime.now().date())
    })

# Save model history
with open("models/model_history.json", "w") as f:
    json.dump(history, f, indent=4)

print("All equipment models trained and saved successfully.")