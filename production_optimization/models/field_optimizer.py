import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import joblib
import os


class FieldOptimizer:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )
        self.feature_names = None
        self.cv_score = None

    def train(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.cv_score = cross_val_score(self.model, X, y, cv=5, scoring="r2").mean()
        self.model.fit(X, y)
        return self.cv_score

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        if self.feature_names is None:
            return {}
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importances.tolist()),
            key=lambda kv: kv[1], reverse=True,
        ))

    def save(self, path):
        joblib.dump({"model": self.model, "features": self.feature_names,
                      "cv_score": self.cv_score}, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.cv_score = data["cv_score"]
