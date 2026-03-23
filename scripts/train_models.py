import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from db_utils import fetch_data

# MODEL CONFIG
MODELS = {
    "logistic": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=50, max_depth=5),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=50)
}

VERSIONS = ["v1", "v2", "v3"]
EQUIPMENT_TYPES = ["projector", "smartboard", "lighting", "ac"]

os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)


# FEATURE MAPPING PER EQUIPMENT
FEATURES = {
    "projector": [
        "age_years", "daily_usage_hours", "days_since_last_maintenance",
        "last_maintenance_type",
        "avg_temperature_week", "max_temperature_week",
        "filter_cleaning_gap_days"
    ],

    "smartboard": [
        "age_years", "daily_usage_hours", "days_since_last_maintenance",
        "last_maintenance_type",
        "touch_responsiveness", "ghost_touch_issue",
        "software_updated_recently"
    ],

    "lighting": [
        "age_years", "daily_usage_hours", "days_since_last_maintenance",
        "last_maintenance_type",
        "switch_cycles_per_day", "frequent_flickering"
    ],

    "ac": [
        "age_years", "daily_usage_hours", "days_since_last_maintenance",
        "last_maintenance_type",
        "avg_temperature_week", "max_temperature_week",
        "desired_temperature", "occupancy_level",
        "filter_cleaning_gap_days"
    ]
}


def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }


def train():

    all_results = []

    for equipment in EQUIPMENT_TYPES:
        for version in VERSIONS:

            print(f"\nTraining {equipment} | {version}")

            df = fetch_data(version, equipment)

            if df.empty:
                print("No data found, skipping...")
                continue

            X = df[FEATURES[equipment]]
            y = df["failure"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            for model_name, model in MODELS.items():

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                metrics = evaluate_model(y_test, preds)

                model_path = f"models/{equipment}_{model_name}_{version}.pkl"
                joblib.dump(model, model_path)

                result = {
                    "equipment": equipment,
                    "model": model_name,
                    "version": version,
                    **metrics
                }

                all_results.append(result)

                print(f"{model_name} | F1: {metrics['f1_score']:.3f}")

    with open("metrics/model_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    train()