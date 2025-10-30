# ml_module — quick guide

Two-step workflow:

1) Training & saving models (use your notebook's DataFrame)
```python
from ml_module import fit_and_save_models

# df: pandas DataFrame with features and targets
disease_map = {
  "disease_a": ["feat1","feat2","feat3"],
  "disease_b": ["f1","f2","f3","f4"],
  "disease_c": ["x1","x2"],
  "disease_d": ["a","b","c","d","e"],
}
target_map = {
  "disease_a": "target_a_column_name",
  "disease_b": "target_b_column_name",
  "disease_c": "target_c_column_name",
  "disease_d": "target_d_column_name",
}
fit_and_save_models(df, disease_map, target_map)
# This creates ml_module/model_files/<disease_key>.pipeline.joblib and <...>.model.joblib
