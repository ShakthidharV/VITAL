# replace the existing predict_risk(...) in ml_nhanes_module/predictor.py with this

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json

MODEL_DIR = Path(__file__).parent / "model_files"
SCHEMA_PATH = MODEL_DIR / "schema.json"

DIABETES_KEY = "Diabetes"
LIVER_KEY = "Liver Condition"
KIDNEY_KEY = "Weak/Failing Kidney"
CVD_KEY = "CVD"

def _load_schema():
    if not SCHEMA_PATH.exists():
        return {}
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

def list_models():
    schema = _load_schema()
    return [k for k in schema.keys() if k != "cvd_components"]

def get_expected_features(disease_key):
    schema = _load_schema()
    return schema.get(disease_key)

def _load_artifacts_for(disease_key):
    base = MODEL_DIR / disease_key.replace(" ", "_")
    preproc_path = base.with_suffix(".preproc.joblib")
    model_path = base.with_suffix(".model.joblib")
    if not preproc_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Artifacts for '{disease_key}' not found in {MODEL_DIR}")
    preproc = joblib.load(preproc_path)
    model = joblib.load(model_path)
    return preproc, model

def predict_risk(disease_key, features_dict):
    """
    Predict probability for disease_key.
    - features_dict: {feature_name: value, ...} . All expected features must be present.
    - Returns float in [0,1].
    """
    schema = _load_schema()
    if disease_key != CVD_KEY:
        preproc, model = _load_artifacts_for(disease_key)
        expected = schema.get(disease_key)
        if expected is None:
            raise KeyError(f"No schema for disease '{disease_key}'")

        # Check all required features present
        missing = [f for f in expected if f not in features_dict]
        if missing:
            raise KeyError(f"Missing features for {disease_key}: {missing}")

        # Build a single-row DataFrame with the exact column names used during fit
        X_df = pd.DataFrame([{k: features_dict[k] for k in expected}], columns=expected)

        # Transform and predict
        X_t = preproc.transform(X_df)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_t)
            if probs.shape[1] >= 2:
                return float(np.clip(probs[0, 1], 0.0, 1.0))
            else:
                return float(np.clip(probs[0, 0], 0.0, 1.0))
        elif hasattr(model, "decision_function"):
            df_val = model.decision_function(X_t)
            p = 1.0 / (1.0 + np.exp(-df_val[0]))
            return float(np.clip(p, 0.0, 1.0))
        else:
            pred = model.predict(X_t)[0]
            return float(pred)

    else:
        # CVD multilabel case
        preproc, model = _load_artifacts_for(CVD_KEY)
        expected = schema.get(CVD_KEY)
        components = schema.get("cvd_components", [])
        if expected is None or not components:
            raise RuntimeError("CVD schema or components missing")

        missing = [f for f in expected if f not in features_dict]
        if missing:
            raise KeyError(f"Missing features for CVD: {missing}")

        X_df = pd.DataFrame([{k: features_dict[k] for k in expected}], columns=expected)
        X_t = preproc.transform(X_df)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_t)[0]  # shape (n_components,)
            # Aggregation: use max probability among components (conservative).
            return float(np.clip(float(np.max(probs)), 0.0, 1.0))
        else:
            preds = model.predict(X_t)[0]
            return float(np.max(preds))
