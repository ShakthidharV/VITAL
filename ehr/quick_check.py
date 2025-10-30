# quick_check_nhanes.py
import pandas as pd
from ml_nhanes_module import train_and_save_all_models  # only if you need to retrain
from ml_nhanes_module import predict_risk, get_expected_features, list_models
import json, math

df = pd.read_csv("ehr/merged_nhanes_readable.csv", encoding="Windows-1252")

models = list_models()
print("Models:", models)

def make_sample_from_row(df, row_idx, features):
    sample = {}
    for f in features:
        if f in df.columns:
            val = df.loc[row_idx, f]
            # leave NaN as-is to let imputer handle it
            sample[f] = (None if pd.isna(val) else val)
        else:
            sample[f] = 0.0
    return sample

for m in models:
    feats = get_expected_features(m)
    print(f"\n--- Testing model: {m} (expects {len(feats)} features) ---")
    for i in range(min(10, len(df))):
        sample = make_sample_from_row(df, i, feats)
        # convert None -> numpy.nan to be explicit
        for k,v in sample.items():
            if v is None:
                sample[k] = float("nan")
        p = predict_risk(m, sample)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0, f"Probability out of range for model {m}, row {i}: {p}"
        print(f"row {i}: prob={p:.6f}")
print("\nAll smoke tests passed.")


# distribution_check.py
import pandas as pd, numpy as np
from ml_nhanes_module import list_models, get_expected_features, predict_risk

df = pd.read_csv("ehr/merged_nhanes_readable.csv", encoding="Windows-1252")
for m in list_models():
    feats = get_expected_features(m)
    probs = []
    for idx in range(len(df)):
        sample = {f: (float("nan") if pd.isna(df.loc[idx,f]) else df.loc[idx,f]) if f in df.columns else 0.0 for f in feats}
        probs.append(predict_risk(m, sample))
    probs = np.array(probs)
    print(m, "count:", len(probs), "mean:", probs.mean(), "std:", probs.std(), "min:", probs.min(), "max:", probs.max())


