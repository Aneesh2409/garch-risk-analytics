"""Monte-Carlo Value-at-Risk / Expected Shortfall and coverage backtests.

The estimator turns a daily volatility forecast into a one-day-ahead loss
distribution and reads VaR and ES off it. Volatility forecasts are taken as
given (typically from :mod:`volatility`); risk figures are reported directly
from the model, with no post-hoc scaling applied to the estimates.

SIGN CONVENTION
---------------
VaR and ES are expressed in RETURN space as left-tail thresholds, so both are
normally negative. A VaR at level ``alpha`` is the return such that a worse
return occurs with probability ``alpha``; a backtest "breach" is a day whose
realised return falls below that threshold. The loss magnitude is ``-VaR``.

MONTE-CARLO CHOICE
------------------
For a pure Gaussian or Student-t scale family the simulated quantile coincides
with the analytic one, so we estimate the *standardised* tail quantile and
expected shortfall once by simulation and scale them by each day's volatility
forecast. Keeping it Monte-Carlo (rather than hard-coding an analytic
quantile) means non-scale extensions -- jump components, regime mixtures,
bootstrapped innovations -- drop in without re-deriving anything.

BACKTESTS
---------
* Kupiec (1995) unconditional-coverage POF test -- is the breach *rate* right?
* Christoffersen (1998) independence test -- are breaches *clustered*?
* Christoffersen conditional-coverage test -- the two jointly (LR_uc + LR_ind).

A VaR model can pass one and fail another, and the failures mean different
things: Kupiec catches a miscalibrated level, independence catches a model
that is slow to react to volatility clustering. Reporting both separates those
diagnoses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .config import N_SIMULATIONS, RANDOM_SEED


# --- Monte-Carlo VaR / ES -----------------------------------------------------

def _standardised_innovations(n_sims: int, dist: str, dof: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Draw unit-variance innovations from the chosen distribution.

    A raw Student-t with ``dof`` degrees of freedom has variance
    ``dof / (dof - 2)``; we rescale by ``sqrt((dof - 2) / dof)`` so the
    innovations have unit variance and a day's volatility forecast is exactly
    the standard deviation of its loss distribution.
    """
    if dist == "normal":
        return rng.standard_normal(n_sims)
    if dist == "t":
        if dof <= 2:
            raise ValueError("Student-t needs dof > 2 for finite variance")
        return rng.standard_t(dof, n_sims) * np.sqrt((dof - 2) / dof)
    raise ValueError(f"unknown dist {dist!r}; use 'normal' or 't'")


def _standardised_var_es(alpha: float, n_sims: int, dist: str, dof: float,
                         rng: np.random.Generator) -> tuple[float, float]:
    """Standardised (unit-vol) VaR and ES quantiles at level ``alpha``."""
    z = _standardised_innovations(n_sims, dist, dof, rng)
    var = float(np.quantile(z, alpha))
    tail = z[z <= var]
    es = float(tail.mean()) if tail.size else var
    return var, es


def monte_carlo_var_es(sigma: float, alpha: float = 0.05,
                       n_sims: int = N_SIMULATIONS, dist: str = "t",
                       dof: float = 5.0, seed: int = RANDOM_SEED
                       ) -> tuple[float, float]:
    """One-day VaR and ES for a single volatility forecast ``sigma``.

    Returns ``(var, es)`` as left-tail returns (negative). ES is at least as
    extreme as VaR.
    """
    rng = np.random.default_rng(seed)
    z_var, z_es = _standardised_var_es(alpha, n_sims, dist, dof, rng)
    return sigma * z_var, sigma * z_es


def rolling_var_es(sigma: pd.Series, alpha: float = 0.05,
                   n_sims: int = N_SIMULATIONS, dist: str = "t",
                   dof: float | pd.Series = 5.0,
                   seed: int = RANDOM_SEED) -> pd.DataFrame:
    """VaR and ES for each day in a volatility-forecast series.

    The standardised tail quantiles are estimated by simulation and scaled by
    each day's ``sigma``. ``dof`` may be a scalar or a Series aligned to
    ``sigma``: passing the GARCH-fitted dof (a Series) makes the loss
    distribution consistent with the volatility model's own innovations
    instead of relying on a fixed value. Standardised figures are computed
    once per distinct dof, so a per-day dof stays cheap.

    Returns a DataFrame indexed like ``sigma`` with columns ``VaR`` and ``ES``.
    """
    rng = np.random.default_rng(seed)
    sig = sigma.to_numpy()

    if np.isscalar(dof):
        z_var, z_es = _standardised_var_es(alpha, n_sims, dist, float(dof), rng)
        return pd.DataFrame({"VaR": sig * z_var, "ES": sig * z_es},
                            index=sigma.index)

    # Per-day dof: reuse the standardised tail figures for repeated dof values.
    dof_vals = (pd.Series(dof).reindex(sigma.index).ffill().bfill().to_numpy())
    cache: dict[float, tuple[float, float]] = {}
    z_var = np.empty(len(sig))
    z_es = np.empty(len(sig))
    for i, d in enumerate(dof_vals):
        key = round(float(d), 2)
        if key not in cache:
            cache[key] = _standardised_var_es(alpha, n_sims, dist, key, rng)
        z_var[i], z_es[i] = cache[key]
    return pd.DataFrame({"VaR": sig * z_var, "ES": sig * z_es},
                        index=sigma.index)


# --- Coverage backtests -------------------------------------------------------

def _safe_term(count: int, prob: float) -> float:
    """``count * log(prob)`` with the ``0 * log(0) = 0`` convention."""
    return count * np.log(prob) if (count > 0 and prob > 0) else 0.0


def kupiec_pof_test(n_obs: int, n_breaches: int, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures (unconditional coverage) test.

    Returns ``(LR_uc, p_value)``. ``LR_uc`` is chi-square(1) under the null
    that the true breach rate equals ``alpha``. A small p-value rejects the
    model -- whether from too many breaches (level too loose) or too few
    (level too conservative).
    """
    N, x, p = n_obs, n_breaches, alpha
    pi = x / N
    ll_null = _safe_term(N - x, 1 - p) + _safe_term(x, p)
    if x == 0 or x == N:
        ll_alt = 0.0
    else:
        ll_alt = _safe_term(N - x, 1 - pi) + _safe_term(x, pi)
    lr = -2.0 * (ll_null - ll_alt)
    return lr, float(chi2.sf(lr, 1))


def christoffersen_independence_test(breaches: np.ndarray) -> tuple[float, float]:
    """Christoffersen independence (no-clustering) test.

    Tests whether a breach today is independent of a breach yesterday, against
    a first-order Markov alternative. Returns ``(LR_ind, p_value)``,
    chi-square(1) under the null of independence.
    """
    b = np.asarray(breaches).astype(int)
    prev, cur = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_null = _safe_term(n00 + n10, 1 - pi) + _safe_term(n01 + n11, pi)
    ll_alt = (_safe_term(n00, 1 - pi01) + _safe_term(n01, pi01)
              + _safe_term(n10, 1 - pi11) + _safe_term(n11, pi11))
    lr = -2.0 * (ll_null - ll_alt)
    return lr, float(chi2.sf(lr, 1))


def christoffersen_cc_test(n_obs: int, n_breaches: int, breaches: np.ndarray,
                           alpha: float) -> tuple[float, float]:
    """Conditional-coverage test: ``LR_cc = LR_uc + LR_ind``, chi-square(2).

    Jointly tests correct breach rate *and* independence.
    """
    lr_uc, _ = kupiec_pof_test(n_obs, n_breaches, alpha)
    lr_ind, _ = christoffersen_independence_test(breaches)
    lr_cc = lr_uc + lr_ind
    return lr_cc, float(chi2.sf(lr_cc, 2))


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of backtesting a VaR series against realised returns."""
    n_obs: int
    n_breaches: int
    observed_rate: float
    expected_rate: float
    kupiec_lr: float
    kupiec_p: float
    independence_lr: float
    independence_p: float
    cc_lr: float
    cc_p: float
    significance: float

    @property
    def kupiec_pass(self) -> bool:
        return self.kupiec_p >= self.significance

    @property
    def independence_pass(self) -> bool:
        return self.independence_p >= self.significance

    @property
    def cc_pass(self) -> bool:
        return self.cc_p >= self.significance


def backtest_var(returns: pd.Series, var: pd.Series, alpha: float = 0.05,
                 significance: float = 0.05) -> BacktestResult:
    """Backtest a VaR series against realised returns over the common dates.

    A breach is a day whose realised return falls below the VaR threshold.
    Runs Kupiec, independence, and conditional-coverage tests at the given
    ``significance`` level.
    """
    aligned = pd.concat([returns.rename("ret"), var.rename("var")],
                        axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("returns and var have no overlapping, non-NaN dates")

    breaches = (aligned["ret"] < aligned["var"]).to_numpy().astype(int)
    n_obs = int(breaches.size)
    n_breaches = int(breaches.sum())

    lr_uc, p_uc = kupiec_pof_test(n_obs, n_breaches, alpha)
    lr_ind, p_ind = christoffersen_independence_test(breaches)
    lr_cc, p_cc = christoffersen_cc_test(n_obs, n_breaches, breaches, alpha)

    return BacktestResult(
        n_obs=n_obs,
        n_breaches=n_breaches,
        observed_rate=n_breaches / n_obs,
        expected_rate=alpha,
        kupiec_lr=lr_uc, kupiec_p=p_uc,
        independence_lr=lr_ind, independence_p=p_ind,
        cc_lr=lr_cc, cc_p=p_cc,
        significance=significance,
    )
