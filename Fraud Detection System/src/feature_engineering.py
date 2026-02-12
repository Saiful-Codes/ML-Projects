import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing to learn

    def transform(self, X):
        X = X.copy()

        required = ["Time", "Amount"]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X["log_amount"] = np.log1p(X["Amount"])
        X["hour"] = (X["Time"] // 3600).astype(int)
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)

        return X
