# trainer.py (patched for XGBoost version compatibility, sparse OHE, and compressed joblib dumps)
import os
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from inspect import signature

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier

# Constants + target keys
MODEL_DIR = Path(__file__).parent / "model_files"
MODEL_DIR.mkdir(exist_ok=True)
SCHEMA_PATH = MODEL_DIR / "schema.json"

# Keys exactly matching your notebook mapping
DIABETES_KEY = "Diabetes"
LIVER_KEY = "Liver Condition"
KIDNEY_KEY = "Weak/Failing Kidney"
CVD_KEY = "CVD"  # aggregated key for the multilabel CVD model

# original target column names used in your notebook
TARGET_COLS = {
    DIABETES_KEY: "Doctor told you have diabetes",
    LIVER_KEY: "Ever told you had any liver condition",
    KIDNEY_KEY: "Ever told you had weak/failing kidneys",
}
CVD_COMPONENTS = [
    "Ever told you had coronary heart disease",
    "Ever told you had angina/angina pectoris",
    "Ever told you had heart attack",
    "Ever told you had a stroke",
]


def _save_schema(schema):
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def _onehot_encoder_sparse():
    """
    Return an OneHotEncoder configured to produce sparse output where supported.
    Handles scikit-learn API differences between versions.
    """
    encoder_kwargs = {"handle_unknown": "ignore"}
    sig = signature(OneHotEncoder)
    if "sparse_output" in sig.parameters:
        encoder_kwargs["sparse_output"] = True
    elif "sparse" in sig.parameters:
        # older versions use 'sparse' (True means sparse matrix)
        encoder_kwargs["sparse"] = True
    # else: rely on default behavior (usually sparse)
    return OneHotEncoder(**encoder_kwargs)


def _xgb_classifier_version_safe(**extra_params):
    """
    Return an XGBClassifier built in a way that is compatible with both XGBoost 1.x and 3.x:
    - Only add use_label_encoder if the parameter exists in that version.
    """
    xgb_params = {"eval_metric": "logloss", "random_state": 42}
    xgb_params.update(extra_params or {})
    try:
        # instantiate a dummy classifier to inspect supported params
        dummy_params = xgb.XGBClassifier().get_params()
        if "use_label_encoder" in dummy_params:
            xgb_params["use_label_encoder"] = False
    except Exception:
        # if inspection fails, just omit use_label_encoder
        pass
    return xgb.XGBClassifier(**xgb_params)


def _fit_and_save_single(df, disease_key, feature_list, target_col, estimator=None):
    """
    Fit preprocessing & model for a single binary label and save artifacts.
    """
    df_local = df.copy()
    # keep rows where target is 1 or 2 then map 1->1,2->0 (per notebook)
    df_local = df_local[df_local[target_col].isin([1, 2])].copy()
    df_local[target_col] = df_local[target_col].map({1: 1, 2: 0})

    # drop columns with >50% missing (except the required features)
    miss_frac = df_local.isna().mean()
    drop_cols = [c for c in df_local.columns if (miss_frac[c] > 0.5 and c not in feature_list)]
    if drop_cols:
        df_local = df_local.drop(columns=drop_cols)

    # ensure feature_list present
    feature_list = [f for f in feature_list if f in df_local.columns]
    if not feature_list:
        raise RuntimeError(f"No features found for disease {disease_key} in dataframe columns")

    X = df_local[feature_list]
    y = df_local[target_col].astype(int)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # identify categorical vs numeric for ColumnTransformer
    categorical_cols = [
        c for c in X_train.columns if X_train[c].dtype == "object" or X_train[c].nunique() <= 10
    ]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    numeric_transformer = ("num", SimpleImputer(strategy="median"), numeric_cols)
    # Use sparse OneHotEncoder when available to reduce memory/disk footprint
    cat_transformer = ("cat", _onehot_encoder_sparse(), categorical_cols)

    transformers = []
    if numeric_cols:
        transformers.append(numeric_transformer)
    if categorical_cols:
        transformers.append(cat_transformer)

    preproc = ColumnTransformer(transformers=transformers, remainder="drop")

    # fit preprocessing on train
    preproc.fit(X_train)

    X_train_t = preproc.transform(X_train)
    X_test_t = preproc.transform(X_test)

    # choose estimator (XGBoost by default) - version-safe
    if estimator is None:
        estimator = _xgb_classifier_version_safe()

    estimator.fit(X_train_t, y_train)

    # Save artifacts (compressed)
    base = MODEL_DIR / disease_key.replace(" ", "_")
    base.parent.mkdir(parents=True, exist_ok=True)
    preproc_path = base.with_suffix(".preproc.joblib")
    model_path = base.with_suffix(".model.joblib")

    # compress artifacts to reduce size on disk / slug
    joblib.dump(preproc, preproc_path, compress=("lzma", 3), protocol=4)
    joblib.dump(estimator, model_path, compress=("lzma", 3), protocol=4)

    return {
        "preproc": str(preproc_path),
        "model": str(model_path),
        "features": feature_list,
        "categorical": categorical_cols,
        "numeric": numeric_cols,
    }


def _fit_and_save_cvd_multilabel(df, predefined_features, top_n=7, estimator=None):
    """
    Simplified multilabel training: combine the 4 CVD targets, impute/encode features and
    train OneVsRestClassifier with XGBoost as base estimator (default).
    Saves a single preproc + multilabel model artifact.
    """
    df_local = df.copy()
    # mask rows where all cvd targets are in [1,2] per notebook
    mask = df_local[CVD_COMPONENTS].apply(lambda col: col.isin([1, 2])).all(axis=1)
    df_cvd = df_local.loc[mask].copy()
    for t in CVD_COMPONENTS:
        df_cvd[t] = df_cvd[t].map({1: 1, 2: 0})

    # drop high missing columns (except predefined_features)
    miss_frac = df_cvd.isna().mean()
    drop_cols = [c for c in df_cvd.columns if (miss_frac[c] > 0.5 and c not in predefined_features)]
    if drop_cols:
        df_cvd = df_cvd.drop(columns=drop_cols)

    # select features: we'll use predefined_features ∩ columns
    feature_list = [f for f in predefined_features if f in df_cvd.columns]
    if not feature_list:
        raise RuntimeError("No predefined CVD features found in dataframe.")

    X = df_cvd[feature_list]
    y = df_cvd[CVD_COMPONENTS].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_cols = [
        c for c in X_train.columns if X_train[c].dtype == "object" or X_train[c].nunique() <= 10
    ]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", _onehot_encoder_sparse(), categorical_cols))

    preproc = ColumnTransformer(transformers=transformers, remainder="drop")
    preproc.fit(X_train)

    X_train_t = preproc.transform(X_train)

    if estimator is None:
        base = _xgb_classifier_version_safe()
        estimator = OneVsRestClassifier(base)

    estimator.fit(X_train_t, y_train)

    base_name = MODEL_DIR / CVD_KEY.replace(" ", "_")
    preproc_path = base_name.with_suffix(".preproc.joblib")
    model_path = base_name.with_suffix(".model.joblib")
    joblib.dump(preproc, preproc_path, compress=("lzma", 3), protocol=4)
    joblib.dump(estimator, model_path, compress=("lzma", 3), protocol=4)

    return {
        "preproc": str(preproc_path),
        "model": str(model_path),
        "features": feature_list,
        "categorical": categorical_cols,
        "numeric": numeric_cols,
        "cvd_components": CVD_COMPONENTS,
    }


def train_and_save_all_models(csv_path="merged_nhanes_readable.csv"):
    """
    Main entrypoint:
     - loads CSV (Windows-1252 encoding as in your notebook)
     - trains the three single-label models with the predefined features from your notebook
     - trains the multilabel CVD model with predefined features
     - saves artifacts to model_files/
     - writes schema.json describing expected features for runtime
    """
    df = pd.read_csv(csv_path, encoding="Windows-1252")
    schema = {}

    # predefined feature lists from your notebook (only keep those present in df)
    predefined_feature_map = {
        "Doctor told you have diabetes": [
            "Fasting Glucose (mg/dL)",
            "Glycohemoglobin (%)",
            "Triglyceride (mg/dL)",
            "Direct HDL-Cholesterol (mg/dL)",
            "Waist Circumference (cm)",
            "Body Mass Index (kg/m2)",
            "Systolic: Blood pressure (2nd reading) (mm Hg)",
        ],
        "Ever told you had any liver condition": [
            "Alanine aminotransferase (ALT) (U/L)",
            "Aspartate aminotransferase (AST) (U/L)",
            "Alkaline phosphatase (U/L)",
            "Gamma-glutamyl transferase (GGT) (U/L)",
            "Total bilirubin (mg/dL)",
            "Body Mass Index (kg/m2)",
            "Waist Circumference (cm)",
            "Triglyceride (mg/dL)",
        ],
        "Ever told you had weak/failing kidneys": [
            "Creatinine, serum (mg/dL)",
            "Blood urea nitrogen (mg/dL)",
            "Albumin, urine (µg/mL)",
            "Creatinine, urine (mg/dL)",
        ],
    }

    # Train single-label models
    for target_col, disease_key in [
        (TARGET_COLS[DIABETES_KEY], DIABETES_KEY),
        (TARGET_COLS[LIVER_KEY], LIVER_KEY),
        (TARGET_COLS[KIDNEY_KEY], KIDNEY_KEY),
    ]:
        # filter features present in df
        predefined = [f for f in predefined_feature_map[target_col] if f in df.columns]
        print(f"Training {disease_key} using features: {predefined}")
        info = _fit_and_save_single(df, disease_key, predefined, target_col)
        schema[disease_key] = info["features"]

    # Train multilabel CVD
    cvd_predefined = [
        "Age at Screening (Adjudicated - Recode)",
        "Gender",
        "Systolic: Blood pressure (2nd reading) (mm Hg)",
        "Diastolic: Blood pressure (2nd reading) (mm Hg)",
        "Total Cholesterol (mg/dL)",
        "Direct HDL-Cholesterol (mg/dL)",
        "LDL-cholesterol (mg/dL)",
        "Body Mass Index (kg/m2)",
    ]
    print(f"Training multi-label CVD using features: {cvd_predefined}")
    info = _fit_and_save_cvd_multilabel(df, cvd_predefined)
    schema[CVD_KEY] = info["features"]
    # Also store the component mapping
    schema["cvd_components"] = info.get("cvd_components", [])

    _save_schema(schema)
    print("Training complete. Artifacts and schema saved to:", MODEL_DIR)
    return schema
