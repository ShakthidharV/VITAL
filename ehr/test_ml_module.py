# test_nhanes_pipeline.py
from ml_nhanes_module import train_and_save_all_models, predict_risk, get_expected_features, list_models
import json

# 1) Train and save artifacts (this will read merged_nhanes_readable.csv)
print("Training models (this may take minutes depending on CPU)...")
schema = train_and_save_all_models("ehr/merged_nhanes_readable.csv")
print("Schema saved:", json.dumps(schema, indent=2))

# 2) Quick runtime check: list saved models and predict using mean values from CSV
print("Available models:", list_models())

# load CSV to build a sample input (mean values)
import pandas as pd
df = pd.read_csv("ehr/merged_nhanes_readable.csv", encoding="Windows-1252")

for m in list_models():
    feats = get_expected_features(m)
    print(f"\nModel {m} expects {len(feats)} features: {feats}")
    # build sample input: use column mean if present, else 0
    sample = {}
    for f in feats:
        if f in df.columns:
            sample[f] = float(df[f].dropna().mean())
        else:
            sample[f] = 0.0
    p = predict_risk(m, sample)
    print(f"Predicted probability for {m} (sample): {p:.4f}")


from ml_nhanes_module import predict_risk, get_expected_features
import pandas as pd

df = pd.read_csv("ehr/merged_nhanes_readable.csv", encoding="Windows-1252")
# 1) Predict for a real patient row (index 0)
d = "Diabetes"
feats = get_expected_features(d)
sample = {f: (df.loc[0, f] if f in df.columns else 0.0) for f in feats}
print("sample row ->", sample)
print("prob:", predict_risk(d, sample))

# 2) Repeat for other models (Liver Condition, Weak/Failing Kidney, CVD)
