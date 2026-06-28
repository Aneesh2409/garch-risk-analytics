"""Tests for the volatility module.

Key things pinned down here:
  * the vectorised realised-vol matches a naive reference implementation,
  * the GARCH forecaster doesn't trigger arch's DataScaleWarning,
  * forecasts are aligned with no look-ahead, positive, and finite.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from garch_risk.volatility import (
    garch_volatility_by_asset,
    realised_volatility,
    rolling_garch_volatility,
)


def _synthetic_returns(n=500, daily_vol=0.013, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.standard_t(6, n) * daily_vol, index=idx, name="TEST")


def _naive_historical_sigma(returns: pd.Series, window: int) -> pd.Series:
    """A straightforward O(n^2) reference implementation, for cross-checking."""
    out = []
    for i in range(len(returns)):
        if i < window:
            out.append(returns.iloc[:i + 1].std())
        else:
            out.append(returns.iloc[i - window + 1:i + 1].std())
    return pd.Series(out, index=returns.index, name=returns.name)


def test_realised_vol_matches_naive_loop():
    """The vectorised version must equal the naive reference implementation."""
    r = _synthetic_returns(n=200)
    window = 20
    fast = realised_volatility(r, window)
    slow = _naive_historical_sigma(r, window)
    pd.testing.assert_series_equal(fast, slow, check_names=True)


def test_realised_vol_hand_computed():
    r = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02], name="X")
    rv = realised_volatility(r, window=3)
    # Last point: sample std of the last 3 returns.
    expected = np.std([0.015, -0.005, 0.02], ddof=1)
    assert rv.iloc[-1] == pytest.approx(expected)


def test_garch_no_scale_warning():
    """Fitting must not trigger arch's DataScaleWarning (poorly-scaled input)."""
    r = _synthetic_returns(n=350)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rolling_garch_volatility(r, window=250, refit_every=21)
    names = [w.category.__name__ for w in caught]
    assert "DataScaleWarning" not in names


def test_garch_output_alignment_and_validity():
    r = _synthetic_returns(n=400)
    window = 250
    sigma = rolling_garch_volatility(r, window=window, refit_every=21)

    # One forecast per day from `window` onward, indexed by the day it is FOR.
    assert len(sigma) == len(r) - window
    pd.testing.assert_index_equal(sigma.index, r.index[window:])
    # Volatility is strictly positive and finite everywhere.
    assert (sigma > 0).all()
    assert np.isfinite(sigma).all()


def test_garch_recovers_approximate_vol_level():
    """On data with ~1.3% daily vol, forecasts should sit in that ballpark."""
    r = _synthetic_returns(n=600, daily_vol=0.013)
    sigma = rolling_garch_volatility(r, window=250, refit_every=21)
    assert 0.007 < sigma.mean() < 0.022


def test_garch_is_deterministic():
    r = _synthetic_returns(n=400, seed=7)
    s1 = rolling_garch_volatility(r, window=250, refit_every=21)
    s2 = rolling_garch_volatility(r, window=250, refit_every=21)
    pd.testing.assert_series_equal(s1, s2)


def test_daily_refit_runs():
    """refit_every=1 (full daily refit) is a valid, if slow, configuration."""
    r = _synthetic_returns(n=290)
    sigma = rolling_garch_volatility(r, window=250, refit_every=1)
    assert len(sigma) == len(r) - 250
    assert (sigma > 0).all()


def test_by_asset_wrapper():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=350)
    df = pd.DataFrame(
        {"A": rng.standard_t(6, 350) * 0.012,
         "B": rng.standard_t(6, 350) * 0.02},
        index=idx,
    )
    out = garch_volatility_by_asset(df, window=250, refit_every=21)
    assert set(out) == {"A", "B"}
    assert all((s > 0).all() for s in out.values())


def test_garch_forecasts_returns_sigma_and_nu():
    """rolling_garch_forecasts exposes volatility, fitted dof, and skew."""
    from garch_risk.volatility import rolling_garch_forecasts
    r = _synthetic_returns(n=400)
    fc = rolling_garch_forecasts(r, window=250, refit_every=21)
    assert list(fc.columns) == ["sigma", "nu", "skew"]
    assert (fc["sigma"] > 0).all()
    # Fitted Student-t dof must exceed 2 (finite variance) and be finite.
    assert (fc["nu"] > 2).all()
    assert np.isfinite(fc["nu"]).all()
    # Student-t has no asymmetry parameter -> the skew column is all-NaN.
    assert fc["skew"].isna().all()
    # The sigma column matches the dedicated volatility function.
    sig = rolling_garch_volatility(r, window=250, refit_every=21)
    pd.testing.assert_series_equal(fc["sigma"].rename(r.name), sig)


def test_garch_forecasts_normal_has_nan_dof():
    from garch_risk.volatility import rolling_garch_forecasts
    r = _synthetic_returns(n=300)
    fc = rolling_garch_forecasts(r, window=250, refit_every=21, dist="normal")
    assert fc["nu"].isna().all()
    assert fc["skew"].isna().all()   # normal has neither dof nor skew


def test_window_too_large_raises():
    r = _synthetic_returns(n=100)
    with pytest.raises(ValueError):
        rolling_garch_volatility(r, window=250)
