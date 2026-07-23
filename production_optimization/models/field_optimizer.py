"""Field optimizer using pymoo NSGA2 multi-objective optimization."""

import numpy as np
import joblib
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination


class ProductionAllocationProblem(Problem):
    def __init__(self, n_features, feature_names=None):
        super().__init__(n_var=n_features, n_obj=2, n_constr=0,
                         xl=-3 * np.ones(n_features), xu=3 * np.ones(n_features))
        self.n_features = n_features
        self.feature_names = feature_names
        self._coef_profit = None
        self._coef_efficiency = None
        self._intercept_profit = 0.0
        self._intercept_efficiency = 0.0

    def fit_linear_model(self, X, y):
        X_aug = np.column_stack([X, np.ones(len(X))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            if self._coef_profit is None:
                self._coef_profit = coeffs[:-1]
                self._intercept_profit = coeffs[-1]
            else:
                self._coef_efficiency = coeffs[:-1]
                self._intercept_efficiency = coeffs[-1]
        except np.linalg.LinAlgError:
            pass

    def _evaluate(self, X, out, *args, **kwargs):
        if self._coef_profit is not None:
            profit = -(X @ self._coef_profit + self._intercept_profit)
        else:
            profit = -np.sum(X * np.random.uniform(0.5, 1.5, self.n_features), axis=1)

        if self._coef_efficiency is not None:
            efficiency = -(X @ self._coef_efficiency + self._intercept_efficiency)
        else:
            efficiency = np.sum(X * 0.01, axis=1)

        out["F"] = np.column_stack([profit, efficiency])


class FieldOptimizer:
    def __init__(self):
        self.problem = None
        self.algorithm = None
        self.feature_names = None
        self.cv_score = None
        self._scaler_mean = None
        self._scaler_std = None
        self._result = None

    def train(self, X, y, feature_names=None):
        self.feature_names = feature_names
        n_features = X.shape[1]

        self._scaler_mean = X.mean(axis=0)
        self._scaler_std = X.std(axis=0) + 1e-10
        X_scaled = (X - self._scaler_mean) / self._scaler_std

        n = len(X)
        fold_size = n // 5
        scores = []
        for fold in range(5):
            val_start = fold * fold_size
            val_end = min((fold + 1) * fold_size, n)
            X_val = X_scaled[val_start:val_end]
            y_val = y[val_start:val_end]
            X_train_fold = np.concatenate([X_scaled[:val_start], X_scaled[val_end:]])
            y_train_fold = np.concatenate([y[:val_start], y[val_end:]])

            problem = ProductionAllocationProblem(n_features, feature_names)
            problem.fit_linear_model(X_train_fold, y_train_fold)

            pred = X_val @ problem._coef_profit + problem._intercept_profit
            ss_res = np.sum((y_val - pred) ** 2)
            ss_tot = np.sum((y_val - y_val.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            scores.append(r2)

        self.cv_score = np.mean(scores)

        self.problem = ProductionAllocationProblem(n_features, feature_names)
        self.problem.fit_linear_model(X_scaled, y)

        self.algorithm = NSGA2(pop_size=50)
        termination = get_termination("n_gen", 50)
        self._result = minimize(
            self.problem, self.algorithm, termination, seed=42, verbose=False
        )
        return self.cv_score

    def predict(self, X):
        if self._scaler_mean is not None:
            X = (X - self._scaler_mean) / self._scaler_std
        if self.problem._coef_profit is not None:
            return X @ self.problem._coef_profit + self.problem._intercept_profit
        return np.zeros(len(X))

    def feature_importances(self):
        if self.feature_names is None or self.problem is None:
            return {}
        if self.problem._coef_profit is not None:
            importances = np.abs(self.problem._coef_profit)
            total = importances.sum()
            if total > 0:
                importances /= total
            return dict(sorted(
                zip(self.feature_names, importances.tolist()),
                key=lambda kv: kv[1], reverse=True,
            ))
        return {}

    def save(self, path):
        joblib.dump({
            "problem": self.problem,
            "feature_names": self.feature_names,
            "cv_score": self.cv_score,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.problem = data["problem"]
        self.feature_names = data["feature_names"]
        self.cv_score = data["cv_score"]
        self._scaler_mean = data.get("scaler_mean")
        self._scaler_std = data.get("scaler_std")
