"""
model_predict.py
Loads the trained FraudDetectionModel and wraps predict_fraud()
"""

from pathlib import Path
import pandas as pd
from ml_model.model_class import FraudDetectionModel
from ml_model.feature_engineering import create_features

MODEL_PATH = Path(__file__).parent / "fraud_detection_model.pkl"
_loaded_model: FraudDetectionModel | None = None

# ------------------------------------------------------
# Internal loader (singleton style)
# ------------------------------------------------------
def _get_model() -> FraudDetectionModel:
    global _loaded_model
    if _loaded_model is None:
        _loaded_model = FraudDetectionModel.load_model(MODEL_PATH)
    return _loaded_model

# ------------------------------------------------------
# Public API
# ------------------------------------------------------
def predict_fraud(df: pd.DataFrame) -> pd.DataFrame:
    model = _get_model()
    df_fe = create_features(df.copy())
    df_fe["prediction"] = model.predict(df_fe)
    return df_fe
