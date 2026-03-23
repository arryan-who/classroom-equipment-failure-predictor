import json
import os

METRICS_FILE = "metrics/model_metrics.json"
OUTPUT_FILE = "models/model_registry.json"


def build_registry():

    with open(METRICS_FILE, "r") as f:
        data = json.load(f)

    # store all results per equipment
    equipment_models = {}

    for entry in data:
        eq = entry["equipment"]

        if eq not in equipment_models:
            equipment_models[eq] = []

        equipment_models[eq].append(entry)

    registry = {}

    for eq, models in equipment_models.items():

        # sort models by F1 score
        sorted_models = sorted(models, key=lambda x: x["f1_score"], reverse=True)

        best_model = sorted_models[0]

        registry[eq] = {
            "selected_model": best_model["model"],
            "selected_version": best_model["version"],
            "f1_score": best_model["f1_score"],
            "all_models": sorted_models
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(registry, f, indent=4)

    print("\nModel Registry Built Successfully:\n")

    for eq, details in registry.items():
        print(f"{eq.upper()} → {details['selected_model']} ({details['selected_version']}) | F1: {details['f1_score']:.3f}")


if __name__ == "__main__":
    build_registry()