"""Feature construction for hourly campus energy usage.

Two pieces:

* ``ContextImputer`` fills each of the four numeric measurements from the
  other three and from building and calendar context, and keeps a 0/1 flag for
  every value it filled, so the model can still see that it was missing.
* ``design_matrix`` turns a frame into the structural basis used by the
  linear part of the model: building effects, smooth daily and yearly cycles,
  and per-building slopes on the numeric measurements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["temperature", "humidity", "occupancy", "previous_usage"]
CATEGORICAL = ["building_id", "building_type", "day_of_week"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TARGET = "energy_usage"


class ContextImputer:
    """Fill each numeric field from the other fields and its building and time context.

    For every field, a small gradient-boosting regressor is trained on the
    rows where that field is observed, using building, type, day, hour, month
    and the other three measurements (left as NaN when they are missing too;
    the trees route missing values natively). Fitted on training rows only.
    Fields the regressor cannot fill fall back to the training median.

    ``method="median"`` instead fills with the training median for the same
    building and four-hour band; it is kept as the comparison baseline.
    """

    def __init__(self, method: str = "model", random_state: int = 0):
        if method not in ("model", "median"):
            raise ValueError("method must be 'model' or 'median'")
        self.method = method
        self.random_state = random_state

    def _context(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "building": frame["building_id"].map(self.building_codes_).fillna(-1).to_numpy(float),
            "type": frame["building_type"].map(self.type_codes_).fillna(-1).to_numpy(float),
            "day": frame["day_of_week"].map({d: i for i, d in enumerate(DAYS)}).fillna(-1).to_numpy(float),
            "hour": frame["hour"].to_numpy(float),
            "month": frame["month"].to_numpy(float),
        }, index=frame.index)

    def fit(self, frame: pd.DataFrame) -> "ContextImputer":
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.building_codes_ = {k: i for i, k in enumerate(sorted(frame["building_id"].unique()))}
        self.type_codes_ = {k: i for i, k in enumerate(sorted(frame["building_type"].unique()))}
        self.median_ = {c: float(frame[c].median()) for c in NUMERIC}
        context = self._context(frame)
        self.models_ = {}
        if self.method == "median":
            band = frame["building_id"].astype(str) + "/" + (frame["hour"] // 4).astype(str)
            self.band_median_ = {c: frame.groupby(band)[c].median() for c in NUMERIC}
            return self
        for c in NUMERIC:
            observed = frame[c].notna().to_numpy()
            x = pd.concat([context, frame[[o for o in NUMERIC if o != c]]], axis=1).to_numpy(float)
            self.models_[c] = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                l2_regularization=10.0, categorical_features=[0, 1, 2], random_state=self.random_state,
            ).fit(x[observed], frame.loc[observed, c].to_numpy(float))
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (filled numeric values, missing-value flags), aligned to ``frame``."""
        context = self._context(frame)
        filled = pd.DataFrame(index=frame.index)
        flags = pd.DataFrame(index=frame.index)
        for c in NUMERIC:
            value = frame[c].astype(float)
            missing = value.isna().to_numpy()
            out = value.to_numpy().copy()
            if missing.any() and self.method == "median":
                band = frame["building_id"].astype(str) + "/" + (frame["hour"] // 4).astype(str)
                out[missing] = band[missing].map(self.band_median_[c]).to_numpy(float)
            elif missing.any():
                x = pd.concat([context, frame[[o for o in NUMERIC if o != c]]], axis=1).to_numpy(float)[missing]
                out[missing] = self.models_[c].predict(x)
            filled[c] = np.where(np.isnan(out), self.median_[c], out)
            flags[f"{c}_missing"] = missing.astype(float)
        return filled, flags


def _cycles(values: np.ndarray, period: float, harmonics: int) -> np.ndarray:
    angle = 2 * np.pi * values[:, None] * np.arange(1, harmonics + 1)[None, :] / period
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _one_hot(series: pd.Series, levels: list[str]) -> np.ndarray:
    return (series.to_numpy()[:, None] == np.asarray(levels)[None, :]).astype(float)


def design_matrix(
    frame: pd.DataFrame,
    filled: pd.DataFrame,
    flags: pd.DataFrame,
    buildings: list[str],
    types: list[str],
    hour_harmonics: int = 3,
) -> np.ndarray:
    """Structural basis for the linear model.

    Columns: building and building-type indicators, day of week, daily
    (``hour_harmonics`` Fourier pairs) and yearly (2 pairs) cycles, a weekend
    flag, scaled numeric measurements, per-building copies of the daily cycle
    and of the numeric slopes, hinge terms for hot and humid hours, and the
    missing-value flags.
    """
    b = _one_hot(frame["building_id"], buildings)
    t = _one_hot(frame["building_type"], types)
    d = _one_hot(frame["day_of_week"], DAYS)
    weekend = frame["day_of_week"].isin(["Saturday", "Sunday"]).to_numpy(float)[:, None]
    daily = _cycles(frame["hour"].to_numpy(float), 24, hour_harmonics)
    yearly = _cycles(frame["month"].to_numpy(float) - 1, 12, 2)
    temp, hum, occ, prev = (filled[c].to_numpy() for c in NUMERIC)
    numeric = np.column_stack([temp - 28, (hum - 78) / 10, occ / 100, prev / 20])
    hinge = np.column_stack([np.maximum(temp - 30, 0), np.maximum(26 - temp, 0), np.maximum(hum - 85, 0) / 10])
    calendar = np.column_stack([daily, yearly, weekend])

    def per_building(block: np.ndarray) -> np.ndarray:
        return (b[:, :, None] * block[:, None, :]).reshape(len(frame), -1)

    return np.column_stack([
        b, t, d, calendar, numeric, hinge,
        per_building(calendar),
        per_building(numeric),
        (t[:, :, None] * numeric[:, None, :]).reshape(len(frame), -1),
        flags.to_numpy(),
    ])


def tree_features(frame: pd.DataFrame, filled: pd.DataFrame, flags: pd.DataFrame,
                  buildings: list[str], types: list[str]) -> np.ndarray:
    """Compact, unscaled features for the boosted residual model (trees need no basis expansion)."""
    weekend = frame["day_of_week"].isin(["Saturday", "Sunday"]).to_numpy(float)
    return np.column_stack([
        frame["building_id"].map({k: i for i, k in enumerate(buildings)}).fillna(-1).to_numpy(float),
        frame["building_type"].map({k: i for i, k in enumerate(types)}).fillna(-1).to_numpy(float),
        frame["day_of_week"].map({k: i for i, k in enumerate(DAYS)}).fillna(-1).to_numpy(float),
        frame["hour"].to_numpy(float),
        frame["month"].to_numpy(float),
        weekend,
        filled[NUMERIC].to_numpy(),
        flags.to_numpy(),
    ])
