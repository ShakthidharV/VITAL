# ml_module/loader.py
import joblib
from pathlib import Path
import threading
import json

_ROOT = Path(__file__).parent
_MODEL_DIR = _ROOT / "model_files"
_SCHEMA_PATH = _MODEL_DIR / "schema.json"

_lock = threading.Lock()
_cache = {}

def _ensure_dirs():
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

def list_available_models():
    """Return list of disease keys for which artifacts exist (schema + model file)."""
    if not _MODEL_DIR.exists():
        return []
    if not _MODEL_DIR.exists():
        return []
    if _SCHEMA := load_schema():
        return list(_SCHEMA.keys())
    return []

def load_schema():
    """Load schema.json mapping disease_key -> ordered feature list (if present)."""
    if not _SCHEMA_PATH.exists():
        return {}
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

def save_schema(schema: dict):
    """Save schema.json. schema: {disease_key: [feat1, feat2, ...], ...}"""
    _ensure_dirs()
    _SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")

def model_artifact_paths(disease_key: str):
    """Return dict with 'pipeline' and 'model' file paths for given disease."""
    _ensure_dirs()
    base = _MODEL_DIR / disease_key
    return {
        "pipeline": base.with_suffix(".pipeline.joblib"),
        "model": base.with_suffix(".model.joblib"),
    }

def load_model_artifact(disease_key: str):
    """
    Load and return (pipeline, model) for disease_key.
    pipeline usually contains preprocessing (imputer/scaler/encoder).
    Raises FileNotFoundError if artifacts not found.
    """
    with _lock:
        if disease_key in _cache:
            return _cache[disease_key]
        paths = model_artifact_paths(disease_key)
        ppath = paths["pipeline"]
        mpath = paths["model"]
        if not ppath.exists() or not mpath.exists():
            raise FileNotFoundError(f"Missing model artifacts for '{disease_key}'. "
                                    f"Expected: {ppath} and {mpath}")
        pipeline = joblib.load(ppath)
        model = joblib.load(mpath)
        schema = load_schema()
        features = schema.get(disease_key)
        _cache[disease_key] = (pipeline, model, features)
        return pipeline, model, features
