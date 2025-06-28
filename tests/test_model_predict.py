# tests/test_model_predict.py
#python -m unittest tests/test_model_predict.py

import unittest
import pandas as pd
from ml_model.model_predict import predict_fraud

class TestPredictFraud(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "step": [1],
                "type": ["TRANSFER"],
                "amount": [1000],
                "nameOrig": ["A1"],
                "oldbalanceOrg": [5000],
                "newbalanceOrig": [4000],
                "nameDest": ["B1"],
                "oldbalanceDest": [0],
                "newbalanceDest": [1000],
                "isFraud": [0],
                "isFlaggedFraud": [0]  
            }
        )

    def test_has_prediction_column(self):
        out = predict_fraud(self.df)
        self.assertIn("prediction", out.columns)
        self.assertEqual(out.shape[0], 1)

    def test_prediction_is_binary(self):
        out = predict_fraud(self.df)
        self.assertIn(out.iloc[0]["prediction"], [0, 1])

if __name__ == "__main__":
    unittest.main()
