"""Cross-validation that reproduces the test set's missing values.

The test rows are missing each numeric field about 7% of the time, against
about 1.5% in training, so plain cross-validation is optimistic. ``test_like``
copies a validation fold and blanks fields using missingness patterns drawn
from the test features (only which fields are missing; no test values or
labels are used).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .features import NUMERIC, TARGET

SEED = 20261003


def test_like(frame: pd.DataFrame, patterns: np.ndarray, seed: int) -> pd.DataFrame:
    """Copy ``frame`` and blank numeric fields with patterns sampled (with replacement) from ``patterns``."""
    rng = np.random.default_rng(seed)
    drawn = patterns[rng.integers(0, len(patterns), len(frame))]
    out = frame.copy()
    for j, c in enumerate(NUMERIC):
        out.loc[drawn[:, j], c] = np.nan
    return out


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def cross_validate(make: Callable[[], object], train: pd.DataFrame, patterns: np.ndarray,
                   folds: int = 5, masks: tuple[int, ...] = (1, 2, 3)) -> dict:
    """Out-of-fold predictions on the native folds and on ``len(masks)`` test-like copies of each fold."""
    native = np.empty(len(train))
    masked = np.empty((len(masks), len(train)))
    pattern_id = np.empty((len(masks), len(train)), dtype=int)
    for fit_idx, val_idx in KFold(folds, shuffle=True, random_state=SEED).split(train):
        model = make().fit(train.iloc[fit_idx])
        fold = train.iloc[val_idx]
        native[val_idx] = model.predict(fold)
        for k, seed in enumerate(masks):
            blanked = test_like(fold, patterns, seed + 100 * int(val_idx[0]))
            masked[k, val_idx] = model.predict(blanked)
            pattern_id[k, val_idx] = blanked[NUMERIC].isna().to_numpy() @ (1 << np.arange(len(NUMERIC)))
    y = train[TARGET].to_numpy(float)
    return {
        "native_rmse": rmse(y, native),
        "test_like_rmse": rmse(np.tile(y, len(masks)), masked.ravel()),
        "native_pred": native, "masked_pred": masked, "pattern_id": pattern_id,
    }
