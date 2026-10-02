"""Loading the competition files from ``data/``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import CATEGORICAL, NUMERIC, TARGET

DATA = Path(__file__).resolve().parents[1] / "data"
FEATURES = ["id", *CATEGORICAL[:2], "hour", CATEGORICAL[2], "month", *NUMERIC]


def load(directory: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test). Expects ``train.csv`` and ``test.csv`` in ``directory``."""
    train_path, test_path = directory / "train.csv", directory / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Put the competition's train.csv and test.csv in {directory} (see data/README.md).")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    missing = set(FEATURES + [TARGET]) - set(train.columns) | set(FEATURES) - set(test.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return train, test
