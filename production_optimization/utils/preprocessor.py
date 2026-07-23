import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "well_count", "total_water_rate", "total_gas_rate",
    "water_injection_rate", "operating_cost_usd", "revenue_per_bbl",
    "artificial_lift_power_kw",
]

CATEGORICAL_FEATURES = ["lift_type"]

OPT_TARGET = "net_profit"
ALLOC_TARGET = "production_efficiency"


class FieldPreprocessor:
    def __init__(self):
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 CATEGORICAL_FEATURES),
            ]
        )
        self._fitted = False

    def fit_transform(self, df, target):
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = df[target]
        X_t = self.transformer.fit_transform(X)
        self._fitted = True
        return X_t, y

    def transform(self, df):
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        return self.transformer.transform(X)

    def get_feature_names(self):
        num_names = NUMERIC_FEATURES
        cat_names = list(
            self.transformer.named_transformers_["cat"].get_feature_names_out(
                CATEGORICAL_FEATURES
            )
        )
        return num_names + cat_names
