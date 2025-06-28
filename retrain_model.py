"""
retrain_model.py
Run this once whenever you change feature_engineering or model_class.
"""

from pathlib import Path
import pandas as pd
from ml_model.model_class import FraudDetectionModel

TRAIN_CSV = Path("data/output_part_1.csv")             
MODEL_OUT = Path("ml_model/fraud_detection_model.pkl")

def main() -> None:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)
    model = FraudDetectionModel()
    model.train(df)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_OUT)

if __name__ == "__main__":
    main()
