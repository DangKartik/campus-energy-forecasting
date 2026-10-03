import numpy as np

from campus_energy.features import NUMERIC, ContextImputer, design_matrix


def test_imputer_fills_every_value_and_flags_exactly_the_missing_ones(train, test_rows):
    filled, flags = ContextImputer().fit(train).transform(test_rows)
    assert not filled.isna().any().any()
    for c in NUMERIC:
        assert (flags[f"{c}_missing"].to_numpy() == test_rows[c].isna().to_numpy()).all()
        observed = test_rows[c].notna()
        assert np.allclose(filled.loc[observed, c], test_rows.loc[observed, c])


def test_imputer_survives_a_row_with_every_measurement_missing(train, test_rows):
    rows = test_rows.head(3).copy()
    rows[NUMERIC] = np.nan
    filled, flags = ContextImputer().fit(train).transform(rows)
    assert np.isfinite(filled.to_numpy()).all()
    assert (flags.to_numpy() == 1).all()


def test_imputation_beats_the_median(train, test_rows):
    hidden = test_rows.dropna(subset=["temperature"]).copy()
    truth = hidden["temperature"].to_numpy()
    hidden["temperature"] = np.nan
    filled, _ = ContextImputer().fit(train).transform(hidden)
    model_err = np.abs(filled["temperature"].to_numpy() - truth).mean()
    median_err = np.abs(train["temperature"].median() - truth).mean()
    assert model_err < 0.6 * median_err


def test_design_matrix_shape_does_not_depend_on_the_rows(train, test_rows):
    imp = ContextImputer().fit(train)
    b, t = sorted(train.building_id.unique()), sorted(train.building_type.unique())
    a = design_matrix(train, *imp.transform(train), b, t)
    z = design_matrix(test_rows.head(5), *imp.transform(test_rows.head(5)), b, t)
    assert a.shape[1] == z.shape[1]
    assert np.isfinite(a).all() and np.isfinite(z).all()


def test_median_imputation_is_available_as_a_baseline(train, test_rows):
    filled, flags = ContextImputer(method="median").fit(train).transform(test_rows)
    assert not filled.isna().any().any()
    assert flags.to_numpy().sum() == test_rows[NUMERIC].isna().to_numpy().sum()
