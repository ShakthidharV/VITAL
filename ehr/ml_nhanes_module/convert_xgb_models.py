# convert_xgb_models.py
import joblib
from pathlib import Path
import xgboost as xgb
import tempfile
import shutil
import sys

MODEL_DIR = Path(__file__).parent / "model_files"
if not MODEL_DIR.exists():
    print("model_files/ not found", file=sys.stderr)
    raise SystemExit(1)

for model_joblib in sorted(MODEL_DIR.glob("*.model.joblib")):
    print("Processing", model_joblib.name)
    obj = joblib.load(model_joblib)

    # If it is an sklearn wrapper XGBClassifier or XGBRegressor
    if isinstance(obj, (xgb.sklearn.XGBClassifier, xgb.sklearn.XGBRegressor)):
        try:
            # Try to obtain booster (works if the wrapper has underlying booster)
            booster = obj.get_booster()
            # Save booster to a temporary file in XGBoost native format (JSON)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            booster.save_model(tmp.name)
            tmp.close()
            print("  saved booster to", tmp.name)

            # Create a fresh XGBClassifier/Regressor and load the native model
            if isinstance(obj, xgb.sklearn.XGBClassifier):
                clean = xgb.sklearn.XGBClassifier()
            else:
                clean = xgb.sklearn.XGBRegressor()
            # load_model supports native json/binary formats
            clean.load_model(tmp.name)

            # Replace joblib artifact with the freshly loaded wrapper
            backup = model_joblib.with_suffix(model_joblib.suffix + ".bak")
            shutil.copy2(model_joblib, backup)
            joblib.dump(clean, model_joblib, compress=('lzma', 3), protocol=4)
            print("  replaced joblib with reloaded model (backup at {})".format(backup.name))
            try:
                Path(tmp.name).unlink()
            except Exception:
                pass
        except Exception as e:
            print("  failed to convert sklearn XGB wrapper:", e)
            # Fallback: try saving booster via xgb.Booster if obj is booster-like
    else:
        # Maybe this is a raw Booster saved inside joblib (rare), handle gracefully
        if isinstance(obj, xgb.core.Booster):
            print("  Found raw Booster - saving native model and creating wrapper")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            obj.save_model(tmp.name)
            tmp.close()
            wrapper = xgb.sklearn.XGBClassifier()
            wrapper.load_model(tmp.name)
            backup = model_joblib.with_suffix(model_joblib.suffix + ".bak")
            shutil.copy2(model_joblib, backup)
            joblib.dump(wrapper, model_joblib, compress=('lzma', 3), protocol=4)
            print("  replaced booster joblib with wrapper (backup at {})".format(backup.name))
            try:
                Path(tmp.name).unlink()
            except Exception:
                pass
        else:
            print("  Skipping (not an xgb wrapper/Booster):", type(obj))
