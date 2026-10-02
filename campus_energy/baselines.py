"""Reference models the structural Ridge is compared against."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import DAYS, NUMERIC, TARGET, ContextImputer


class Persistence:
    """Predict the previous hour's usage (imputed from context when it is missing)."""

    def fit(self, frame: pd.DataFrame) -> "Persistence":
        self.imputer_ = ContextImputer().fit(frame)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.imputer_.transform(frame)[0]["previous_usage"].to_numpy()


class GradientBoosting:
    """Histogram gradient boosting on the raw columns, missing values routed natively by the trees."""

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def _x(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            frame["building_id"].map(self.buildings_).fillna(-1),
            frame["building_type"].map(self.types_).fillna(-1),
            frame["day_of_week"].map({d: i for i, d in enumerate(DAYS)}).fillna(-1),
            frame["hour"], frame["month"], frame[NUMERIC],
        ]).astype(float)

    def fit(self, frame: pd.DataFrame) -> "GradientBoosting":
        self.buildings_ = {k: i for i, k in enumerate(sorted(frame["building_id"].unique()))}
        self.types_ = {k: i for i, k in enumerate(sorted(frame["building_type"].unique()))}
        self.model_ = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=25,
            l2_regularization=10.0, categorical_features=[0, 1, 2], random_state=self.random_state,
        ).fit(self._x(frame), frame[TARGET].to_numpy(float))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._x(frame))
