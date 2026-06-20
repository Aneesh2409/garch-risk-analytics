"""Volatility forecast evaluation: Mincer-Zarnowitz regression and QLIKE loss.

A VaR backtest checks one point in the tail. This module asks a different and
more fundamental question: is the GARCH *volatility* forecast itself any good?
That matters because the Greeks and the stress tests consume sigma directly --
they depend on the forecast's accuracy as a conditional standard deviation,
not on the VaR cutoff being perfectly calibrated.

REALISED PROXY
--------------
With daily data the realised variance on day t is proxied by the squared
return r_t^2. It is unbiased (E[r_t^2 | F_{t-1}] equals the conditional
variance) but very noisy -- a chi-square(1) around the truth. The noise is why
the Mincer-Zarnowitz R^2 is always low (often 0.05-0.20) even for a good
forecast; the slope, not the R^2, is the informative statistic.

MINCER-ZARNOWITZ
----------------
Regress realised variance on forecast variance:
    r_t^2 = a + b * sigma_t^2 + e_t.
An unbiased, efficient forecast has a = 0 and b = 1. Standard errors are
Newey-West (HAC), since the residuals are heteroskedastic and autocorrelated;
naive OLS errors would overstate significance. A joint Wald test of
(a, b) = (0, 1) summarises the result.

QLIKE
-----
QLIKE = mean( log(sigma^2) + r^2 / sigma^2 ). A loss function that is robust to
the noise in the realised proxy, used to *rank* competing forecasts (lower is
better). On its own it is not interpretable; against a benchmark it is, so
:func:`evaluate_volatility_forecast` compares the GARCH forecast to one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _newey_west_lags(n: int) -> int:
    """Rule-of-thumb HAC lag length: floor(4 * (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2 / 9)))


@dataclass(frozen=True)
class MZResult:
    """Mincer-Zarnowitz regression of realised on forecast variance."""
    intercept: float
    slope: float
    intercept_se: float
    slope_se: float
    r_squared: float
    joint_p: float          # p-value of the joint test (a, b) = (0, 1)
    n_obs: int
    significance: float

    @property
    def is_unbiased(self) -> bool:
        """Fail to reject (a, b) = (0, 1) at the chosen significance level."""
        return self.joint_p >= self.significance


def _align(realised_var: pd.Series, forecast_var: pd.Series
           ) -> tuple[np.ndarray, np.ndarray]:
    df = pd.concat([realised_var.rename("rv"), forecast_var.rename("fv")],
                   axis=1, join="inner").dropna()
    if df.empty:
        raise ValueError("no overlapping, non-NaN dates between the series")
    if (df["fv"] <= 0).any():
        raise ValueError("forecast variance must be strictly positive")
    return df["rv"].to_numpy(), df["fv"].to_numpy()


def mincer_zarnowitz(realised_var: pd.Series, forecast_var: pd.Series,
                     significance: float = 0.05) -> MZResult:
    """Mincer-Zarnowitz regression with Newey-West (HAC) standard errors."""
    rv, fv = _align(realised_var, forecast_var)
    X = sm.add_constant(fv)
    res = sm.OLS(rv, X).fit(cov_type="HAC",
                            cov_kwds={"maxlags": _newey_west_lags(len(rv))})
    a, b = res.params
    a_se, b_se = res.bse
    wald = res.wald_test((np.eye(2), np.array([0.0, 1.0])),
                         use_f=True, scalar=True)
    return MZResult(
        intercept=float(a), slope=float(b),
        intercept_se=float(a_se), slope_se=float(b_se),
        r_squared=float(res.rsquared),
        joint_p=float(wald.pvalue),
        n_obs=len(rv), significance=significance,
    )


def qlike(realised_var: pd.Series, forecast_var: pd.Series) -> float:
    """QLIKE loss (lower is better). Robust to noise in the realised proxy."""
    rv, fv = _align(realised_var, forecast_var)
    return float(np.mean(np.log(fv) + rv / fv))


@dataclass(frozen=True)
class ForecastEvaluation:
    """Bundle of forecast-quality diagnostics for one asset."""
    mz: MZResult
    qlike_forecast: float
    qlike_benchmark: float | None

    @property
    def beats_benchmark(self) -> bool | None:
        """True if the forecast's QLIKE is below the benchmark's."""
        if self.qlike_benchmark is None:
            return None
        return self.qlike_forecast < self.qlike_benchmark


def evaluate_volatility_forecast(returns: pd.Series, sigma_forecast: pd.Series,
                                 benchmark_sigma: pd.Series | None = None,
                                 significance: float = 0.05
                                 ) -> ForecastEvaluation:
    """Evaluate a daily volatility forecast against realised variance.

    ``returns`` and ``sigma_forecast`` are aligned on their common dates; the
    realised proxy is the squared return. If ``benchmark_sigma`` is given, its
    QLIKE is computed too so the forecast can be ranked against it (e.g. a
    naive rolling-window volatility).
    """
    realised_var = (returns ** 2).rename("rv")
    forecast_var = (sigma_forecast ** 2).rename("fv")

    mz = mincer_zarnowitz(realised_var, forecast_var, significance)
    ql = qlike(realised_var, forecast_var)
    ql_bench = (qlike(realised_var, benchmark_sigma ** 2)
                if benchmark_sigma is not None else None)
    return ForecastEvaluation(mz=mz, qlike_forecast=ql, qlike_benchmark=ql_bench)
