import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json

# optional: used only to detect sparse matrices
try:
    from scipy import sparse as _sparse
except Exception:
    _sparse = None

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

# Helper utilities for robust prediction across versions/wrappers
def _to_dense_if_needed(X):
    # Convert sparse matrix to dense only if necessary / safe
    if _sparse is not None and _sparse.issparse(X):
        try:
            return X.toarray()
        except Exception:
            return X
    return X

def _try_predict_proba(model, X):
    """
    Try to obtain probabilities from the model. Return a numpy array of shape (n_samples, n_classes)
    or raise the last exception encountered.
    """
    X_try = _to_dense_if_needed(X)
    last_exc = None

    # 1) try predict_proba
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_try)
            return np.asarray(probs)
        except Exception as e:
            last_exc = e

    # 2) try decision_function -> convert to probabilities
    if hasattr(model, "decision_function"):
        try:
            df_val = model.decision_function(X_try)
            df_val = np.asarray(df_val)
            # binary case: 1D
            if df_val.ndim == 1 or (df_val.ndim == 2 and df_val.shape[1] == 1):
                df_flat = df_val.ravel()
                p = 1.0 / (1.0 + np.exp(-df_flat))
                return np.vstack([1 - p, p]).T
            # multiclass: apply softmax row-wise
            exp = np.exp(df_val - np.max(df_val, axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            return probs
        except Exception as e:
            last_exc = e

    # 3) try predict -> convert to 0/1 probabilities if possible
    if hasattr(model, "predict"):
        try:
            preds = model.predict(X_try)
            preds = np.asarray(preds)
            # multilabel / multiclass predicted probabilities - return as-is if already float
            if preds.dtype.kind in ("f",):
                # already probabilities
                if preds.ndim == 1:
                    return np.vstack([1 - preds, preds]).T
                return preds
            # integer labels
            if preds.ndim == 1:
                # binary case -> build [1-p, p] with p in {0,1}
                p = preds.astype(float)
                return np.vstack([1 - p, p]).T
            # multilabel (n_samples, n_classes) with 0/1 entries
            return preds.astype(float)
        except Exception as e:
            last_exc = e

    # If nothing worked, raise the last exception for visibility
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No suitable prediction method found on model object.")

def predict_risk(disease_key, features_dict):
    """
    Predict probability for disease_key.
    - features_dict: {feature_name: value, ...} . All expected features must be present.
    - Returns float in [0,1] (for single-label models) or a float summary for CVD (max over components).
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

        # Transform and_predict probabilities robustly
        X_t = preproc.transform(X_df)

        try:
            probs = _try_predict_proba(model, X_t)
            # probs shape handling
            probs = np.asarray(probs)
            if probs.ndim == 1:
                # treat as binary probability of positive class
                p = float(np.clip(probs[0], 0.0, 1.0))
                return p
            if probs.shape[1] == 2:
                return float(np.clip(probs[0, 1], 0.0, 1.0))
            # multiclass -> pick class-1 probability? We'll return max-prob for "positive"
            # but keep original behavior: if only one column, use that; else, pick second column if exists
            if probs.shape[1] >= 2:
                return float(np.clip(probs[0, 1], 0.0, 1.0))
            # fallback: take max
            return float(np.clip(np.max(probs[0]), 0.0, 1.0))
        except Exception as e:
            # bubble up as RuntimeError with context
            raise RuntimeError(f"Prediction failed for '{disease_key}': {e}") from e

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

        try:
            probs = _try_predict_proba(model, X_t)
            probs = np.asarray(probs)
            # If probs is 1D -> treat as single score (rare for multilabel)
            if probs.ndim == 1:
                return float(np.clip(probs[0], 0.0, 1.0))
            # If model.predict_proba returned shape (n_samples, n_classes) for multilabel:
            # ensure we have one row, then return maximum component probability (conservative).
            if probs.ndim == 2:
                # If shape is (1, n_components) -> ok
                row = probs[0]
                # If number of probs matches number of CVD components, use max
                return float(np.clip(float(np.max(row)), 0.0, 1.0))
            # fallback: try model.predict and aggregate
        except Exception:
            # fallback to direct predictions
            try:
                preds = model.predict(_to_dense_if_needed(X_t))
                preds = np.asarray(preds)
                if preds.ndim == 1:
                    return float(np.clip(preds[0], 0.0, 1.0))
                return float(np.clip(float(np.max(preds[0])), 0.0, 1.0))
            except Exception as e:
                raise RuntimeError(f"CVD prediction failed: {e}") from e

