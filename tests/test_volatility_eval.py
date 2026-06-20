"""Tests for the volatility-forecast evaluation module."""

import numpy as np
import pandas as pd
import pytest

from garch_risk.volatility_eval import (
    evaluate_volatility_forecast,
    mincer_zarnowitz,
    qlike,
)


def _sim_varying_variance(n=4000, seed=0):
    """Returns with a known, time-varying conditional variance."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", periods=n)
    true_sigma = 0.01 * (1 + 0.5 * np.sin(np.linspace(0, 40, n)))
    r = true_sigma * rng.standard_normal(n)
    return (pd.Series(r, index=idx, name="R"),
            pd.Series(true_sigma, index=idx, name="S"))


def test_mz_recovers_unit_slope_for_good_forecast():
    """A forecast equal to the true variance gives slope ~1, intercept ~0."""
    r, true_sigma = _sim_varying_variance()
    mz = mincer_zarnowitz(r ** 2, true_sigma ** 2)
    assert 0.6 < mz.slope < 1.4
    assert abs(mz.intercept) < 1e-4
    assert mz.is_unbiased            # cannot reject (a, b) = (0, 1)


def test_mz_perfect_identity():
    """Regressing a series on itself gives slope 1, intercept 0, R^2 = 1."""
    idx = pd.bdate_range("2020-01-01", periods=500)
    fv = pd.Series(np.linspace(0.0001, 0.0009, 500), index=idx)
    mz = mincer_zarnowitz(fv, fv)
    assert mz.slope == pytest.approx(1.0, abs=1e-6)
    assert mz.intercept == pytest.approx(0.0, abs=1e-10)
    assert mz.r_squared == pytest.approx(1.0, abs=1e-9)


def test_mz_biased_forecast_is_flagged():
    """A forecast that is half the true variance should reject unbiasedness."""
    r, true_sigma = _sim_varying_variance()
    biased_var = (true_sigma ** 2) * 0.5
    mz = mincer_zarnowitz(r ** 2, biased_var)
    assert not mz.is_unbiased


def test_qlike_hand_computed():
    rv = pd.Series([0.0004, 0.0001])
    fv = pd.Series([0.0002, 0.0002])
    expected = np.mean(np.log(fv.to_numpy()) + rv.to_numpy() / fv.to_numpy())
    assert qlike(rv, fv) == pytest.approx(expected)


def test_qlike_prefers_better_forecast():
    """The true-variance forecast must score below a constant forecast."""
    r, true_sigma = _sim_varying_variance()
    rv = r ** 2
    good = qlike(rv, true_sigma ** 2)
    const = qlike(rv, pd.Series(np.full(len(r), (r ** 2).mean()), index=r.index))
    assert good < const


def test_evaluate_bundles_and_compares():
    r, true_sigma = _sim_varying_variance()
    # Benchmark: a deliberately worse (constant) vol forecast.
    bench = pd.Series(np.full(len(r), r.std()), index=r.index)
    ev = evaluate_volatility_forecast(r, true_sigma, benchmark_sigma=bench)
    assert ev.qlike_benchmark is not None
    assert ev.beats_benchmark is True
    assert 0.6 < ev.mz.slope < 1.4


def test_evaluate_without_benchmark():
    r, true_sigma = _sim_varying_variance(n=1000)
    ev = evaluate_volatility_forecast(r, true_sigma)
    assert ev.qlike_benchmark is None
    assert ev.beats_benchmark is None


def test_positive_forecast_required():
    idx = pd.bdate_range("2020-01-01", periods=10)
    rv = pd.Series(np.full(10, 0.0001), index=idx)
    bad_fv = pd.Series(np.zeros(10), index=idx)
    with pytest.raises(ValueError):
        mincer_zarnowitz(rv, bad_fv)
