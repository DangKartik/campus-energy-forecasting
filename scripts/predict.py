"""Fit the final model on all training rows and write predictions for the test rows.

    python scripts/predict.py             # writes outputs/predictions.csv and outputs/model.joblib
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_energy.data import load  # noqa: E402
from campus_energy.model import EnergyModel  # noqa: E402


def main() -> None:
    train, test = load()
    model = EnergyModel().fit(train)
    pred = model.predict(test)
    assert len(pred) == len(test) and np.isfinite(pred).all()
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    pd.DataFrame({"id": test["id"], "energy_usage": pred}).to_csv(out / "predictions.csv", index=False)
    joblib.dump(model, out / "model.joblib")
    print(f"wrote {len(pred)} predictions to {out / 'predictions.csv'} (mean {pred.mean():.2f})")


if __name__ == "__main__":
    main()
