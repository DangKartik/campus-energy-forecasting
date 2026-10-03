import numpy as np

from campus_energy.baselines import GradientBoosting, Persistence
from campus_energy.features import NUMERIC
from campus_energy.model import EnergyModel
from campus_energy import validation
from campus_energy.validation import cross_validate, rmse


def test_model_recovers_the_signal(train, test_rows):
    pred = EnergyModel().fit(train).predict(test_rows)
    # Noise in the synthetic target has standard deviation 1.5.
    assert rmse(test_rows.energy_usage, pred) < 2.5
    assert rmse(test_rows.energy_usage, pred) < rmse(test_rows.energy_usage, Persistence().fit(train).predict(test_rows))


def test_predictions_are_deterministic_and_do_not_depend_on_row_order(train, test_rows):
    a = EnergyModel(residual=True).fit(train).predict(test_rows)
    shuffled = test_rows.sample(frac=1.0, random_state=3)
    b = EnergyModel(residual=True).fit(train).predict(shuffled)
    assert np.allclose(a, b[np.argsort(shuffled.index.to_numpy())])


def test_unseen_building_still_gets_a_finite_prediction(train, test_rows):
    rows = test_rows.head(4).copy()
    rows["building_id"] = "NEW_A"
    assert np.isfinite(EnergyModel().fit(train).predict(rows)).all()


def test_test_like_masking_only_removes_values(train):
    patterns = np.array([[True, False, False, False], [False, False, False, True], [False] * 4])
    masked = validation.test_like(train, patterns, seed=0)
    assert masked[NUMERIC].isna().sum().sum() > train[NUMERIC].isna().sum().sum()
    kept = masked[NUMERIC].notna().to_numpy()
    assert np.array_equal(masked[NUMERIC].to_numpy()[kept], train[NUMERIC].to_numpy()[kept])
    assert masked["humidity"].isna().sum() == train["humidity"].isna().sum()


def test_cross_validation_reports_both_scores(train):
    patterns = train[NUMERIC].isna().to_numpy()
    out = cross_validate(GradientBoosting, train.head(600), patterns, folds=3, masks=(1,))
    assert out["native_rmse"] > 0 and out["test_like_rmse"] > 0
    assert out["masked_pred"].shape == (1, 600)
