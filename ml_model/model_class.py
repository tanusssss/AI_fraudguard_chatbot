# app/ml_model/model_class.py

"""
FraudDetectionModel
- Pipeline: MinMaxScaler ➜ SelectPercentile ➜ KNN
- Safety checks: is_trained flag & guard in predict()
"""

import pickle
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.neighbors import KNeighborsClassifier

from ml_model.feature_engineering import create_features

class FraudDetectionModel:
    def __init__(self) -> None:
        self.pipeline = None
        self.feature_names = None
        self.selected_features = None
        self.is_trained = False

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------
    def train(self, train_df: pd.DataFrame) -> None:
        train_df = create_features(train_df)

        X_train = train_df.drop(
            columns=[
                "isFraud",
                "nameOrig",
                "nameDest",
                "step",
                "type",
                "riskNote",
                "riskLevel",
            ]
        )
        y_train = train_df["isFraud"]

        self.feature_names = X_train.columns.tolist()

        self.pipeline = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("selector", SelectPercentile(score_func=f_classif, percentile=30)),
                ("knn", KNeighborsClassifier(n_neighbors=5, p=2, weights="distance")),
            ]
        )
        self.pipeline.fit(X_train, y_train)

        selector = self.pipeline.named_steps["selector"]
        self.selected_features = X_train.columns[
            selector.get_support(indices=True)
        ].tolist()

        self.is_trained = True
        print(" Model trained. Selected features:", self.selected_features)

    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------
    def predict(self, new_df: pd.DataFrame):
        if not self.is_trained or self.pipeline is None:
            raise ValueError(
                "Model not trained. Call train() or load a trained model first."
            )

        new_df = create_features(new_df)
        X = new_df[self.feature_names]
        return self.pipeline.predict(X)

    # ------------------------------------------------------
    # Save / Load helpers
    # ------------------------------------------------------
    def save_model(self, filename: str | Path) -> None:
        if not self.is_trained:
            raise ValueError("Cannot save an un‑trained model. Train first.")
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename: str | Path):
        with open(filename, "rb") as f:
            model = pickle.load(f)
        # Just in case
        model.is_trained = True
        return model
