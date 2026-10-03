import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campus_energy.features import DAYS, NUMERIC  # noqa: E402


def synthetic(n: int, seed: int, missing: float = 0.05) -> pd.DataFrame:
    """A small campus with a known daily cycle, building baselines and an occupancy effect."""
    rng = np.random.default_rng(seed)
    buildings = {"ENG_A": ("Engineering", 55), "SCI_A": ("Science", 75), "LIB_A": ("Library", 45), "RES_A": ("Residential", 47)}
    ids = rng.choice(list(buildings), n)
    hour = rng.integers(0, 24, n)
    occ = rng.gamma(2.0, 30.0, n)
    temp = 28 + 2 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.5, n)
    base = np.array([buildings[b][1] for b in ids])
    usage = base + 6 * np.sin(2 * np.pi * (hour - 8) / 24) + 0.05 * occ + 0.8 * (temp - 28) + rng.normal(0, 1.5, n)
    frame = pd.DataFrame({
        "id": [f"EC{i:06d}" for i in range(n)],
        "building_id": ids,
        "building_type": [buildings[b][0] for b in ids],
        "hour": hour,
        "day_of_week": rng.choice(DAYS, n),
        "month": rng.integers(1, 13, n),
        "temperature": temp.round(1),
        "humidity": (78 + rng.normal(0, 5, n)).round(1),
        "occupancy": occ.round(),
        "previous_usage": (usage + rng.normal(0, 2, n)).round(1),
        "energy_usage": usage.round(1),
    })
    for c in NUMERIC:
        frame.loc[rng.random(n) < missing, c] = np.nan
    return frame


@pytest.fixture
def train():
    return synthetic(1500, seed=1)


@pytest.fixture
def test_rows():
    return synthetic(400, seed=2, missing=0.15)
