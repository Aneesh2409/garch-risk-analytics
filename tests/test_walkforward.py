"""Tests for the walk-forward orchestrator.

Invariants, not findings: calendar-year folds, no look-ahead, exact pooled
coverage, determinism, and call-structure symmetry (frozen == reselected when
every fold reselects the frozen spec). Synthetic data, small window, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from garch_risk.walkforward import (
    CANDIDATE_GRID,
    Spec,
    _fold_boundaries,
    _select_spec,
    run_asset_walkforward,
    summarise,
)

WINDOW = 250


def _garch_like_returns(n=900, seed=1):
    rng = np.random.default_rng(seed)
    sig = np.empty(n)
    eps = np.empty(n)
    sig[0] = 0.012
    eps[0] = 0.0
    for t in range(1, n):
        sig[t] = np.sqrt(2e-6 + 0.08 * eps[t - 1] ** 2 + 0.90 * sig[t - 1] ** 2)
        eps[t] = sig[t] * rng.standard_t(6)
    eps -= 0.2 * np.maximum(-eps, 0) * np.abs(eps)
    idx = pd.bdate_range("2016-01-01", periods=n)
    return pd.Series(eps, index=idx, name="SYNTH")


@pytest.fixture(scope="module")
def returns():
    return _garch_like_returns()


@pytest.fixture(scope="module")
def result(returns):
    return run_asset_walkforward(returns, window=WINDOW, refit_every=5)


def test_grid_is_two_by_two_with_o_q_fixed():
    assert len(CANDIDATE_GRID) == 4
    assert {s.dist for s in CANDIDATE_GRID} == {"t", "skewt"}
    assert {s.p for s in CANDIDATE_GRID} == {1, 2}
    assert all(s.o == 1 and s.q == 1 for s in CANDIDATE_GRID)


def test_folds_are_calendar_years(returns):
    oos = returns.index[WINDOW:]
    folds = _fold_boundaries(oos)
    assert [y for y, _ in folds] == sorted(pd.unique(oos.year))
    # folds partition the OOS index exactly
    total = sum(len(idx) for _, idx in folds)
    assert total == len(oos)


def test_lookahead_assertion_fires(returns):
    oos = returns.index[WINDOW:]
    bad_train = returns.loc[:oos[10]]          # window reaching into OOS
    with pytest.raises(AssertionError):
        _select_spec(bad_train, oos[5], CANDIDATE_GRID)


def test_selected_specs_are_in_grid(result):
    assert result.frozen_spec in CANDIDATE_GRID
    assert all(f.selected_bic in CANDIDATE_GRID for f in result.folds)
    assert all(f.selected_aic in CANDIDATE_GRID for f in result.folds)
    assert all(len(f.fits) == 4 for f in result.folds)


def test_pooled_backtests_cover_every_oos_day(returns, result):
    n_oos = len(returns.index[WINDOW:])
    for alpha in (0.05, 0.01):
        assert result.baseline_backtests[alpha].n_obs == n_oos
        assert result.frozen_backtests[alpha].n_obs == n_oos
        assert result.reselected_backtests[alpha].n_obs == n_oos


def test_baseline_equals_frozen_when_specs_coincide(returns):
    """If the frozen spec IS the baseline spec, the two runs must coincide."""
    from garch_risk.walkforward import Spec
    # force the baseline to whatever the data's frozen spec turns out to be by
    # selecting it first, then re-running with that as the baseline.
    base = run_asset_walkforward(returns, window=WINDOW, refit_every=21,
                                 baseline_spec=Spec("t", 1, 1, 1))
    if base.baseline_equals_frozen:
        for alpha in (0.05, 0.01):
            assert (base.baseline_backtests[alpha].n_breaches
                    == base.frozen_backtests[alpha].n_breaches)
        assert base.qlike_distribution_gap == 0.0
    else:
        # frozen is skew-t here; force baseline = frozen and check coincidence
        forced = run_asset_walkforward(returns, window=WINDOW, refit_every=21,
                                       baseline_spec=base.frozen_spec)
        assert forced.baseline_equals_frozen
        for alpha in (0.05, 0.01):
            assert (forced.baseline_backtests[alpha].n_breaches
                    == forced.frozen_backtests[alpha].n_breaches)
        assert forced.qlike_distribution_gap == 0.0


def test_baseline_per_fold_rates_present(result):
    for alpha in (0.05, 0.01):
        assert len(result.per_fold_breach_rate_baseline[alpha]) == len(result.folds)


def test_deterministic(returns, result):
    again = run_asset_walkforward(returns, window=WINDOW, refit_every=5)
    for alpha in (0.05, 0.01):
        assert (result.frozen_backtests[alpha].n_breaches
                == again.frozen_backtests[alpha].n_breaches)
    assert result.qlike_oos_frozen == again.qlike_oos_frozen


def test_frozen_equals_reselected_when_specs_match(result):
    """If every fold reselects the frozen spec, the two runs must coincide."""
    if all(f.selected_bic == result.frozen_spec for f in result.folds):
        for alpha in (0.05, 0.01):
            assert (result.frozen_backtests[alpha].n_breaches
                    == result.reselected_backtests[alpha].n_breaches)
        assert result.qlike_selection_gap == 0.0
    else:
        pytest.skip("synthetic data selected a non-frozen spec in some fold")


def test_qlike_figures_finite(result):
    assert np.isfinite(result.qlike_in_sample)
    assert np.isfinite(result.qlike_oos_frozen)
    assert np.isfinite(result.qlike_oos_reselected)


def test_per_fold_breach_rates_present(result):
    for alpha in (0.05, 0.01):
        assert len(result.per_fold_breach_rate[alpha]) == len(result.folds)


def test_summarise_is_pure_ascii(result):
    text = summarise(result)
    assert text
    assert all(ord(c) < 128 for c in text)
