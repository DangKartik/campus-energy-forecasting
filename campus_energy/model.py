"""The model: a structural Ridge regression on context-imputed features.

The Ridge carries the structure that is known to matter: building baselines,
daily and yearly cycles, and a per-building response to occupancy, weather
and the previous hour's usage. An optional boosted-tree stage can be fitted to
the Ridge residuals; cross-validation shows it does not help on this data, so
it is off by default and kept only so the comparison can be reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .features import TARGET, ContextImputer, design_matrix, tree_features


@dataclass
class EnergyModel:
    """Structural Ridge regression; ``residual=True`` adds a boosted-tree correction on its residuals."""

    alpha: float = 3.0
    hour_harmonics: int = 3
    imputation: str = "model"
    residual: bool = False
    residual_params: dict = field(default_factory=lambda: dict(
        max_iter=150, learning_rate=0.04, max_leaf_nodes=3,
        min_samples_leaf=60, l2_regularization=50.0,
    ))
    random_state: int = 0

    def fit(self, frame: pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> "EnergyModel":
        y = np.asarray(frame[TARGET] if y is None else y, dtype=float)
        self.buildings_ = sorted(frame["building_id"].unique())
        self.types_ = sorted(frame["building_type"].unique())
        self.imputer_ = ContextImputer(method=self.imputation, random_state=self.random_state).fit(frame)
        filled, flags = self.imputer_.transform(frame)
        z = design_matrix(frame, filled, flags, self.buildings_, self.types_, self.hour_harmonics)
        self.ridge_ = Ridge(alpha=self.alpha).fit(z, y)
        if self.residual:
            r = y - self.ridge_.predict(z)
            x = tree_features(frame, filled, flags, self.buildings_, self.types_)
            self.booster_ = HistGradientBoostingRegressor(
                random_state=self.random_state, **self.residual_params).fit(x, r)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        filled, flags = self.imputer_.transform(frame)
        z = design_matrix(frame, filled, flags, self.buildings_, self.types_, self.hour_harmonics)
        pred = self.ridge_.predict(z)
        if self.residual:
            pred = pred + self.booster_.predict(tree_features(frame, filled, flags, self.buildings_, self.types_))
        return pred
