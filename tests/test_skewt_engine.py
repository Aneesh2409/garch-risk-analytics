"""Tests for the skew-t innovation engine and generalised shape extraction.

Covers the pass-1 additions only; the existing normal/t VaR tests remain the
regression guard for the untouched draw paths. Synthetic data only -- no
network, ASCII-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from garch_risk import var_es as ve
from garch_risk import volatility as vol


# --- skew-t innovations -------------------------------------------------------

def test_skewt_innovations_are_standardised():
    rng = np.random.default_rng(0)
    z = ve._standardised_innovations(2_000_000, "skewt", 6.0, rng, skew=-0.3)
    assert abs(z.mean()) < 5e-3
    assert abs(z.var() - 1.0) < 5e-3


def test_skewt_negative_lambda_is_left_skewed():
    rng = np.random.default_rng(0)
    z = ve._standardised_innovations(2_000_000, "skewt", 6.0, rng, skew=-0.3)
    sample_skew = ((z - z.mean()) ** 3).mean() / z.std() ** 3
    assert sample_skew < -0.1


def test_left_skew_pushes_var_and_es_more_negative():
    var_left, es_left = ve.monte_carlo_var_es(0.02, 0.05, 500_000, "skewt",
                                              6.0, 42, skew=-0.3)
    var_sym, es_sym = ve.monte_carlo_var_es(0.02, 0.05, 500_000, "skewt",
                                            6.0, 42, skew=0.0)
    assert var_left < var_sym
    assert es_left < es_sym


def test_symmetric_skewt_matches_student_t():
    var_sym, _ = ve.monte_carlo_var_es(0.02, 0.05, 500_000, "skewt", 6.0, 42,
                                       skew=0.0)
    var_t, _ = ve.monte_carlo_var_es(0.02, 0.05, 500_000, "t", 6.0, 42)
    assert abs(var_sym - var_t) / abs(var_t) < 0.02


def test_skewt_is_deterministic_under_seed():
    a = ve.monte_carlo_var_es(0.02, 0.05, 50_000, "skewt", 6.0, 123, skew=-0.4)
    b = ve.monte_carlo_var_es(0.02, 0.05, 50_000, "skewt", 6.0, 123, skew=-0.4)
    assert a == b


@pytest.mark.parametrize("bad_skew", [-1.0, 1.0, 1.5, -2.0])
def test_lambda_out_of_range_raises(bad_skew):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ve._standardised_innovations(100, "skewt", 6.0, rng, skew=bad_skew)


def test_skewt_low_eta_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ve._standardised_innovations(100, "skewt", 2.0, rng, skew=0.0)


# --- per-day shape path (rolling) ---------------------------------------------

def _sigma_series(n=200):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    s = np.abs(np.random.default_rng(7).normal(0.01, 0.003, n)) + 1e-4
    return pd.Series(s, index=idx)


def test_rolling_skewt_per_day_runs_and_is_deterministic():
    sig = _sigma_series()
    dof = pd.Series(np.full(len(sig), 6.0), index=sig.index)
    skew = pd.Series(np.linspace(-0.4, -0.1, len(sig)), index=sig.index)
    r1 = ve.rolling_var_es(sig, 0.05, 20_000, "skewt", dof, 42, skew)
    r2 = ve.rolling_var_es(sig, 0.05, 20_000, "skewt", dof, 42, skew)
    assert np.array_equal(r1.to_numpy(), r2.to_numpy())
    assert (r1["VaR"] < 0).all()
    assert (r1["ES"] <= r1["VaR"]).all()


# --- volatility shape extraction ----------------------------------------------

@pytest.fixture(scope="module")
def synth_returns():
    g = np.random.default_rng(11)
    raw = g.standard_t(5, 1000) * 0.01
    raw = raw - 0.3 * np.maximum(-raw, 0) * np.abs(raw)
    idx = pd.date_range("2018-01-01", periods=1000, freq="B")
    return pd.Series(raw, index=idx, name="TEST")


def test_t_fit_leaves_skew_column_nan(synth_returns):
    out = vol.rolling_garch_forecasts(synth_returns, window=365,
                                      refit_every=63, dist="t")
    # Schema (the column set) is pinned in test_volatility.py; here we assert
    # only the skew-t *behaviour*: Student-t carries dof but no asymmetry.
    assert np.isfinite(out["nu"]).all()
    assert out["skew"].isna().all()


def test_skewt_fit_carries_eta_and_lambda(synth_returns):
    out = vol.rolling_garch_forecasts(synth_returns, window=365,
                                      refit_every=63, dist="skewt")
    assert np.isfinite(out["nu"]).all() and (out["nu"] > 2).all()
    assert np.isfinite(out["skew"]).all()
    assert out["skew"].between(-1, 1).all()


def test_p2_spec_is_fittable(synth_returns):
    out = vol.rolling_garch_forecasts(synth_returns, window=365,
                                      refit_every=63, dist="skewt", p=2)
    assert np.isfinite(out["sigma"]).all() and (out["sigma"] > 0).all()
