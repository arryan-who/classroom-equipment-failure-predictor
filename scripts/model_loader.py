import json
import joblib
import os

REGISTRY_PATH = "models/model_registry.json"
MODELS_DIR = "models"


def load_selected_model(equipment: str):
    """
    Loads selected model and returns:
    (model, feature_list)
    """

    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    if equipment not in registry:
        raise ValueError(f"No model found for equipment: {equipment}")

    info = registry[equipment]

    model_name = info["selected_model"]
    version = info["selected_version"]

    model_path = os.path.join(
        MODELS_DIR,
        f"{equipment}_{model_name}_{version}.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    # extract features
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        raise ValueError("Model does not contain feature names")

    return model, features