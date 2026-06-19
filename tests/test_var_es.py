"""Tests for the VaR/ES module.

Pins down:
  * Monte-Carlo VaR/ES against closed-form normal values,
  * the unit-variance standardisation of innovations (so sigma really is the
    loss-distribution standard deviation),
  * correct backtest behaviour: well-calibrated models pass, miscalibrated and
    clustered ones are rejected.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from garch_risk.var_es import (
    backtest_var,
    christoffersen_cc_test,
    christoffersen_independence_test,
    kupiec_pof_test,
    monte_carlo_var_es,
    rolling_var_es,
)


# --- Monte-Carlo VaR / ES -----------------------------------------------------

def test_mc_var_matches_normal_closed_form():
    """Normal MC VaR should match sigma * Phi^{-1}(alpha)."""
    sigma, alpha = 0.02, 0.05
    var, _ = monte_carlo_var_es(sigma, alpha, n_sims=400_000,
                                dist="normal", seed=1)
    expected = sigma * norm.ppf(alpha)          # ~ -0.0329
    assert var == pytest.approx(expected, rel=0.02)


def test_mc_es_matches_normal_closed_form():
    """Normal ES = -sigma * phi(z_alpha) / alpha."""
    sigma, alpha = 0.02, 0.05
    _, es = monte_carlo_var_es(sigma, alpha, n_sims=400_000,
                               dist="normal", seed=1)
    z = norm.ppf(alpha)
    expected = -sigma * norm.pdf(z) / alpha     # ~ -0.0413
    assert es == pytest.approx(expected, rel=0.02)


def test_innovations_are_unit_variance():
    """Innovations must have unit variance, so sigma is the loss-distribution
    standard deviation. The Student-t draw is rescaled by sqrt((dof-2)/dof)."""
    from garch_risk.var_es import _standardised_innovations
    rng = np.random.default_rng(0)
    z_norm = _standardised_innovations(1_000_000, "normal", 5.0, rng)
    z_t = _standardised_innovations(1_000_000, "t", 6.0, rng)
    assert z_norm.std() == pytest.approx(1.0, rel=0.01)
    assert z_t.std() == pytest.approx(1.0, rel=0.01)


def test_t_vs_normal_tail_crossover():
    """Standardised-t fat tails are a redistribution of mass, not an addition.

    At 5% the t quantile is marginally LESS extreme than the normal (still in
    the shoulder), but ES -- which averages the whole tail -- is more extreme.
    At 1% the t dominates on both VaR and ES. This documents the crossover
    near alpha ~ 3% so the behaviour isn't mistaken for a bug.
    """
    # 5%: normal VaR more extreme; t ES more extreme.
    vn5, en5 = monte_carlo_var_es(0.02, 0.05, n_sims=600_000, dist="normal", seed=11)
    vt5, et5 = monte_carlo_var_es(0.02, 0.05, n_sims=600_000, dist="t", dof=6, seed=11)
    assert vn5 < vt5          # normal VaR further into the tail
    assert et5 < en5          # t ES further into the tail

    # 1%: t more extreme on both.
    vn1, en1 = monte_carlo_var_es(0.02, 0.01, n_sims=600_000, dist="normal", seed=11)
    vt1, et1 = monte_carlo_var_es(0.02, 0.01, n_sims=600_000, dist="t", dof=6, seed=11)
    assert vt1 < vn1
    assert et1 < en1


def test_es_at_least_as_extreme_as_var():
    var, es = monte_carlo_var_es(0.02, 0.05, dist="t", dof=5, seed=3)
    assert es <= var < 0


def test_rolling_var_es_alignment_and_sign():
    idx = pd.bdate_range("2020-01-01", periods=50)
    sigma = pd.Series(np.linspace(0.01, 0.03, 50), index=idx, name="S")
    out = rolling_var_es(sigma, alpha=0.05, dist="t", dof=5, seed=4)
    assert list(out.columns) == ["VaR", "ES"]
    pd.testing.assert_index_equal(out.index, idx)
    assert (out["VaR"] < 0).all()
    assert (out["ES"] <= out["VaR"]).all()
    # VaR scales linearly with sigma -> larger sigma, more extreme VaR.
    assert out["VaR"].iloc[-1] < out["VaR"].iloc[0]


def test_dof_too_low_raises():
    with pytest.raises(ValueError):
        monte_carlo_var_es(0.02, 0.05, dist="t", dof=2)


# --- Backtests ----------------------------------------------------------------

def test_kupiec_well_calibrated_passes():
    # 13 breaches in 250 days at 5% expected (12.5) -> should not reject.
    lr, p = kupiec_pof_test(250, 13, 0.05)
    assert p > 0.5
    assert lr >= 0


def test_kupiec_too_many_breaches_rejected():
    lr, p = kupiec_pof_test(250, 25, 0.05)   # double the expected rate
    assert p < 0.05


def test_kupiec_zero_breaches_rejected():
    """A VaR that never breaches is too conservative -- Kupiec should flag it."""
    lr, p = kupiec_pof_test(250, 0, 0.05)
    assert p < 0.05


def test_independence_iid_passes_clustered_fails():
    rng = np.random.default_rng(0)
    iid = (rng.random(250) < 0.05).astype(int)
    _, p_iid = christoffersen_independence_test(iid)
    assert p_iid > 0.05

    clustered = np.array([0] * 230 + [1] * 20)
    lr_c, p_c = christoffersen_independence_test(clustered)
    assert p_c < 0.05


def test_cc_equals_uc_plus_ind():
    rng = np.random.default_rng(5)
    breaches = (rng.random(300) < 0.05).astype(int)
    lr_uc, _ = kupiec_pof_test(300, int(breaches.sum()), 0.05)
    lr_ind, _ = christoffersen_independence_test(breaches)
    lr_cc, _ = christoffersen_cc_test(300, int(breaches.sum()), breaches, 0.05)
    assert lr_cc == pytest.approx(lr_uc + lr_ind, abs=1e-9)


def test_backtest_var_integration():
    """End-to-end: well-calibrated VaR against matching returns passes."""
    rng = np.random.default_rng(7)
    n = 1000
    idx = pd.bdate_range("2018-01-01", periods=n)
    sigma = 0.015
    returns = pd.Series(rng.standard_normal(n) * sigma, index=idx)
    # A correctly specified normal 5% VaR for this constant-sigma series.
    var = pd.Series(np.full(n, sigma * norm.ppf(0.05)), index=idx)

    result = backtest_var(returns, var, alpha=0.05)
    assert result.n_obs == n
    # Observed breach rate should land near 5%.
    assert 0.03 < result.observed_rate < 0.07
    assert result.kupiec_pass
    assert result.independence_pass
    assert result.cc_pass


def test_backtest_var_requires_overlap():
    a = pd.Series([0.01], index=pd.to_datetime(["2020-01-01"]))
    b = pd.Series([-0.02], index=pd.to_datetime(["2021-01-01"]))
    with pytest.raises(ValueError):
        backtest_var(a, b)
