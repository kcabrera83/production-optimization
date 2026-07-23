"""Allocation model using OR-Tools linear programming."""

import numpy as np
import joblib
from ortools.linear_solver import pywraplp


class AllocationModel:
    def __init__(self):
        self.feature_names = None
        self.cv_score = None
        self._scaler_mean = None
        self._scaler_std = None
        self._coefficients = None
        self._intercept = 0.0

    def train(self, X, y, feature_names=None):
        self.feature_names = feature_names
        n_features = X.shape[1]

        self._scaler_mean = X.mean(axis=0)
        self._scaler_std = X.std(axis=0) + 1e-10
        X_scaled = (X - self._scaler_mean) / self._scaler_std

        X_aug = np.column_stack([X_scaled, np.ones(len(X_scaled))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            self._coefficients = coeffs[:-1]
            self._intercept = coeffs[-1]
        except np.linalg.LinAlgError:
            self._coefficients = np.zeros(n_features)
            self._intercept = y.mean()

        n = len(X)
        fold_size = n // 5
        scores = []
        for fold in range(5):
            val_start = fold * fold_size
            val_end = min((fold + 1) * fold_size, n)
            X_val = X_scaled[val_start:val_end]
            y_val = y[val_start:val_end]

            pred = X_val @ self._coefficients + self._intercept
            ss_res = np.sum((y_val - pred) ** 2)
            ss_tot = np.sum((y_val - y_val.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            scores.append(r2)

        self.cv_score = np.mean(scores)
        return self.cv_score

    def predict(self, X):
        if self._scaler_mean is not None:
            X = (X - self._scaler_mean) / self._scaler_std
        if self._coefficients is not None:
            return X @ self._coefficients + self._intercept
        return np.zeros(len(X))

    def optimize_allocation(self, n_vars=5, objective_coeffs=None, constraints=None):
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            return None

        if objective_coeffs is None:
            objective_coeffs = [1.0] * n_vars
        if constraints is None:
            constraints = {"total_water": 1000.0}

        x = {}
        for i in range(n_vars):
            x[i] = solver.IntVar(0, 1000, f"x_{i}")

        objective = solver.Objective()
        for i in range(n_vars):
            objective.SetCoefficient(x[i], int(objective_coeffs[i] * 100))
        objective.SetMaximize()

        total = solver.Constraint(0, int(constraints.get("total_water", 1000)))
        for i in range(n_vars):
            total.SetCoefficient(x[i], 1)

        if solver.Solve() == pywraplp.Solver.OPTIMAL:
            return {f"well_{i}": x[i].solution_value() for i in range(n_vars)}
        return None

    def feature_importances(self):
        if self.feature_names is None or self._coefficients is None:
            return {}
        importances = np.abs(self._coefficients)
        total = importances.sum()
        if total > 0:
            importances /= total
        return dict(sorted(
            zip(self.feature_names, importances.tolist()),
            key=lambda kv: kv[1], reverse=True,
        ))

    def save(self, path):
        joblib.dump({
            "coefficients": self._coefficients,
            "intercept": self._intercept,
            "feature_names": self.feature_names,
            "cv_score": self.cv_score,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self._coefficients = data["coefficients"]
        self._intercept = data["intercept"]
        self.feature_names = data["feature_names"]
        self.cv_score = data["cv_score"]
        self._scaler_mean = data.get("scaler_mean")
        self._scaler_std = data.get("scaler_std")
