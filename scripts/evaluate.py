"""Cross-validate every model and write results/ (tables, report and figures).

    python scripts/evaluate.py            # about 2 minutes on a laptop
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_energy.baselines import GradientBoosting, Persistence  # noqa: E402
from campus_energy.data import load  # noqa: E402
from campus_energy.features import NUMERIC, TARGET  # noqa: E402
from campus_energy.model import EnergyModel  # noqa: E402
from campus_energy.validation import cross_validate, rmse  # noqa: E402

OUT = ROOT / "results"
INK, ACCENT, MUTED, GRID = "#1f2937", "#2563eb", "#9ca3af", "#e5e7eb"

MODELS = {
    "Persistence (previous hour)": Persistence,
    "Gradient boosting, raw columns": GradientBoosting,
    "Structural Ridge, median imputation": lambda: EnergyModel(imputation="median"),
    "Structural Ridge + boosted residual": lambda: EnergyModel(residual=True),
    "Structural Ridge (final)": EnergyModel,
}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train, test = load()
    patterns = test[NUMERIC].isna().to_numpy()
    y = train[TARGET].to_numpy(float)

    results = {}
    for name, make in MODELS.items():
        results[name] = cross_validate(make, train, patterns)
        print(f"{name:40s} native {results[name]['native_rmse']:.4f}  test-like {results[name]['test_like_rmse']:.4f}", flush=True)

    alphas = [0.3, 1, 3, 10, 30]
    sweep = {a: cross_validate(lambda a=a: EnergyModel(alpha=a), train, patterns) for a in alphas}

    table = pd.DataFrame([{"model": k, "cv_rmse": v["native_rmse"], "test_like_cv_rmse": v["test_like_rmse"]}
                          for k, v in results.items()])
    table.to_csv(OUT / "cv_results.csv", index=False, float_format="%.4f")

    final = results["Structural Ridge (final)"]
    gb = results["Gradient boosting, raw columns"]
    yt = np.tile(y, final["masked_pred"].shape[0])
    pid = final["pattern_id"].ravel()
    groups = {"all fields present": pid == 0}
    for j, c in enumerate(NUMERIC):
        groups[f"{c.replace('_', ' ')} missing"] = (pid & (1 << j)) > 0
    by_pattern = pd.DataFrame([{
        "rows": name, "share": float(m.mean()),
        "ridge_rmse": rmse(yt[m], final["masked_pred"].ravel()[m]),
        "boosting_rmse": rmse(yt[m], gb["masked_pred"].ravel()[m]),
    } for name, m in groups.items()])
    by_pattern.to_csv(OUT / "rmse_by_missing_field.csv", index=False, float_format="%.4f")

    by_type = (pd.DataFrame({"type": train["building_type"], "err": (final["native_pred"] - y) ** 2, "y": y})
               .groupby("type").agg(rmse=("err", lambda e: float(np.sqrt(e.mean()))), mean_usage=("y", "mean"), rows=("y", "size"))
               .sort_values("rmse"))
    by_type.to_csv(OUT / "rmse_by_building_type.csv", float_format="%.4f")

    summary = {
        "folds": 5, "test_like_masks_per_fold": 3, "training_rows": len(train),
        "models": {k: {"cv_rmse": round(v["native_rmse"], 4), "test_like_cv_rmse": round(v["test_like_rmse"], 4)} for k, v in results.items()},
        "ridge_alpha_sweep": {str(a): {"cv_rmse": round(v["native_rmse"], 4), "test_like_cv_rmse": round(v["test_like_rmse"], 4)} for a, v in sweep.items()},
        "target_mean": round(float(y.mean()), 2), "target_std": round(float(y.std()), 2),
        "train_missing_rate": {c: round(float(train[c].isna().mean()), 4) for c in NUMERIC},
        "test_missing_rate": {c: round(float(test[c].isna().mean()), 4) for c in NUMERIC},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Figure 1: model comparison.
    fig, ax = plt.subplots(figsize=(8, 4.0))
    names = list(results)[::-1]
    pos = np.arange(len(names))
    ax.barh(pos + 0.18, [results[n]["native_rmse"] for n in names], 0.36, color=ACCENT, label="5-fold CV")
    ax.barh(pos - 0.18, [results[n]["test_like_rmse"] for n in names], 0.36, color=MUTED, label="test-like missing values")
    for p, n in zip(pos, names):
        ax.text(results[n]["native_rmse"] + 0.05, p + 0.18, f"{results[n]['native_rmse']:.2f}", va="center", fontsize=8, color=INK)
        ax.text(results[n]["test_like_rmse"] + 0.05, p - 0.18, f"{results[n]['test_like_rmse']:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(pos, names)
    ax.set_xlabel("RMSE (energy units, lower is better)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_model_comparison.png", dpi=160)
    plt.close(fig)

    # Figure 2: where missing values hurt.
    fig, ax = plt.subplots(figsize=(8, 3.2))
    order = by_pattern.iloc[::-1]
    pos = np.arange(len(order))
    ax.barh(pos + 0.18, order["ridge_rmse"], 0.36, color=ACCENT, label="structural Ridge")
    ax.barh(pos - 0.18, order["boosting_rmse"], 0.36, color=MUTED, label="gradient boosting")
    ax.set_yticks(pos, [f"{r} ({s:.0%})" for r, s in zip(order["rows"], order["share"])])
    ax.set_xlabel("RMSE on test-like validation rows")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_rmse_by_missing_field.png", dpi=160)
    plt.close(fig)

    # Figure 3: predicted against actual, out of fold.
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(y, final["native_pred"], s=4, alpha=0.35, color=ACCENT, linewidths=0)
    lo, hi = float(min(y.min(), final["native_pred"].min())), float(max(y.max(), final["native_pred"].max()))
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8)
    ax.set_xlabel("actual energy usage")
    ax.set_ylabel("out-of-fold prediction")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_predicted_vs_actual.png", dpi=160)
    plt.close(fig)

    print(table.to_string(index=False))
    print(by_pattern.to_string(index=False))
    print(by_type.to_string())
    print(json.dumps(summary["ridge_alpha_sweep"]))


if __name__ == "__main__":
    main()
