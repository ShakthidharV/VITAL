# convert_xgb_models_full.py
import joblib
from pathlib import Path
import xgboost as xgb
import tempfile
import shutil
import sys
import traceback

MODEL_DIR = Path(__file__).parent / "model_files"

if not MODEL_DIR.exists():
    print("model_files/ not found", file=sys.stderr)
    raise SystemExit(1)

def save_booster_to_native(booster, out_path):
    # save as JSON (newer friendly) if available
    try:
        booster.save_model(str(out_path))
        return True
    except Exception as e:
        print("Failed to save booster natively:", e)
        return False

def convert_wrapper(wrapper, dst_path):
    """
    Given an sklearn-wrapper XGBClassifier/XGBRegressor (fitted),
    extract booster, save native, then create a fresh wrapper and load native model.
    """
    try:
        booster = wrapper.get_booster()
    except Exception as e:
        print("  cannot get_booster() from wrapper:", e)
        return False

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.close()
    if not save_booster_to_native(booster, tmp.name):
        return False

    # create fresh wrapper of same class
    cls = wrapper.__class__
    fresh = cls()
    try:
        fresh.load_model(tmp.name)
    except Exception as e:
        print("  failed to load native model into fresh wrapper:", e)
        return False

    # replace joblib on disk
    backup = dst_path.with_suffix(dst_path.suffix + ".bak")
    shutil.copy2(dst_path, backup)
    joblib.dump(fresh, dst_path, compress=('lzma', 3), protocol=4)
    print(f"  converted and replaced {dst_path.name} (backup {backup.name})")
    try:
        Path(tmp.name).unlink()
    except Exception:
        pass
    return True

def convert_booster_file(dst_path):
    """
    If joblib file contains raw Booster, convert to wrapper form.
    """
    obj = joblib.load(dst_path)
    if isinstance(obj, xgb.core.Booster):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        if not save_booster_to_native(obj, tmp.name):
            return False
        wrapper = xgb.sklearn.XGBClassifier()
        try:
            wrapper.load_model(tmp.name)
        except Exception as e:
            print("  failed to load booster into wrapper:", e)
            return False
        backup = dst_path.with_suffix(dst_path.suffix + ".bak")
        shutil.copy2(dst_path, backup)
        joblib.dump(wrapper, dst_path, compress=('lzma', 3), protocol=4)
        print(f"  converted raw Booster {dst_path.name} -> wrapper (backup {backup.name})")
        try:
            Path(tmp.name).unlink()
        except Exception:
            pass
        return True
    return False

for p in sorted(MODEL_DIR.glob("*.model.joblib")):
    print("Processing:", p.name)
    try:
        obj = joblib.load(p)
    except Exception as e:
        print("  failed to load joblib:", e)
        traceback.print_exc()
        continue

    # Case A: sklearn OneVsRestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    if isinstance(obj, OneVsRestClassifier):
        print("  Found OneVsRestClassifier — converting inner estimators if needed.")
        # ensure we have fitted estimators_
        if not hasattr(obj, "estimators_") or not obj.estimators_:
            print("  No estimators_ found on this OneVsRestClassifier; cannot convert.")
            continue
        success_all = True
        for i, sub in enumerate(obj.estimators_):
            if isinstance(sub, (xgb.sklearn.XGBClassifier, xgb.sklearn.XGBRegressor)):
                # save booster from sub
                try:
                    booster = sub.get_booster()
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_sub{i}.json")
                    tmp.close()
                    if not save_booster_to_native(booster, tmp.name):
                        success_all = False
                        continue
                    cls = sub.__class__
                    fresh = cls()
                    try:
                        fresh.load_model(tmp.name)
                    except Exception as e:
                        print("    failed to load into fresh sub wrapper:", e)
                        success_all = False
                        continue
                    obj.estimators_[i] = fresh
                    try:
                        Path(tmp.name).unlink()
                    except Exception:
                        pass
                except Exception as e:
                    print("    error converting sub estimator:", e)
                    success_all = False
            else:
                print("    sub-estimator is not xgboost wrapper; skipping conversion for this sub.")
        if success_all:
            backup = p.with_suffix(p.suffix + ".bak")
            shutil.copy2(p, backup)
            joblib.dump(obj, p, compress=('lzma', 3), protocol=4)
            print(f"  replaced OneVsRestClassifier joblib {p.name} (backup {backup.name})")
        else:
            print("  some subs failed to convert; left original file as-is (backup not created).")
        continue

    # Case B: sklearn xgboost wrapper
    if isinstance(obj, (xgb.sklearn.XGBClassifier, xgb.sklearn.XGBRegressor)):
        ok = convert_wrapper(obj, p)
        if not ok:
            print("  conversion failed for wrapper:", p.name)
        continue

    # Case C: raw Booster saved in joblib
    if convert_booster_file(p):
        continue

    print("  Skipping (not an XGBoost wrapper or Booster):", type(obj))
