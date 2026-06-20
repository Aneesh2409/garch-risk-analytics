"""Volatility estimation: rolling GJR-GARCH(1,1)-t forecasts and a realised-vol
benchmark.

Two estimators, both producing DAILY volatility (raw return units).
Annualisation is the pricer's job, not this module's -- see :mod:`pricing`.

DESIGN NOTES
------------
* GJR-GARCH(1,1): the asymmetry term (``o=1``) captures the leverage effect --
  negative shocks raise next-day variance more than positive ones, which is
  exactly what you want for equity-index tail risk. The Student-t innovations
  (``dist="t"``) accommodate fat tails.
* Returns are rescaled by 100 before fitting. ``arch``'s optimiser is poorly
  conditioned on raw decimal returns (~0.01), so we fit in percentage units
  and divide the forecast back down -- this keeps the estimation well-behaved.
* Refitting is periodic. Parameters are re-estimated every ``refit_every``
  days; between refits they are held fixed while the one-step variance forecast
  is still rolled forward with each new observation (via ``arch``'s ``.fix()``).
  This is the standard speed/rigour compromise: far fewer optimisations, with
  forecasts that still react to yesterday's move.
* The realised-vol benchmark is a single vectorised ``.rolling().std()``.

NO LOOK-AHEAD
-------------
The forecast for day ``t`` is built only from returns up to day ``t-1``
(``train = returns[t-window : t]``, then a one-step-ahead forecast). The
returned series is indexed by the day each forecast is *for*, so it lines up
directly with the realised return on that day for backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from .config import GARCH_WINDOW, REALISED_VOL_WINDOW

# arch's optimiser likes data in roughly the 1-1000 range; daily returns in
# percent (x100) sit there comfortably.
_FIT_SCALE: float = 100.0


def realised_volatility(returns: pd.Series | pd.DataFrame,
                        window: int = REALISED_VOL_WINDOW
                        ) -> pd.Series | pd.DataFrame:
    """Rolling-window realised (sample) volatility, per day.

    Trailing ``window`` with ``min_periods=1`` so early rows fall back to an
    expanding window rather than producing NaNs. The very first row is NaN by
    construction (the sample standard deviation of a single point).
    """
    return returns.rolling(window, min_periods=1).std()


def _rolling_garch(returns: pd.Series, window: int, refit_every: int,
                   dist: str) -> pd.DataFrame:
    """Core rolling fit: returns a DataFrame of daily ``sigma`` and ``nu``.

    ``nu`` is the fitted Student-t degrees of freedom (NaN when ``dist`` is
    not ``"t"``), carried forward between refits alongside the other params.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")

    scaled = returns.to_numpy(dtype=float) * _FIT_SCALE
    n = len(scaled)
    if n <= window:
        raise ValueError(f"need more than {window} observations, got {n}")

    sigmas: list[float] = []
    nus: list[float] = []
    params = None

    for step, t in enumerate(range(window, n)):
        train = scaled[t - window:t]
        model = arch_model(train, vol="Garch", p=1, o=1, q=1,
                           mean="Zero", dist=dist)

        if params is None or step % refit_every == 0:
            res = model.fit(disp="off")          # re-estimate parameters
            params = res.params
        else:
            res = model.fix(params)              # hold params, roll variance

        fc = res.forecast(horizon=1, reindex=False)
        sigmas.append(np.sqrt(fc.variance.values[-1, 0]) / _FIT_SCALE)
        nus.append(float(params["nu"]) if dist == "t" else np.nan)

    return pd.DataFrame({"sigma": sigmas, "nu": nus},
                        index=returns.index[window:])


def rolling_garch_volatility(returns: pd.Series,
                             window: int = GARCH_WINDOW,
                             refit_every: int = 21,
                             dist: str = "t") -> pd.Series:
    """One-step-ahead daily volatility from a rolling GJR-GARCH(1,1).

    Parameters
    ----------
    returns
        Daily log-returns for a single asset (a named Series).
    window
        Trailing estimation window, in trading days.
    refit_every
        Re-estimate parameters every this many days. Between refits the
        parameters are held fixed and only the conditional-variance recursion
        is rolled forward. ``1`` reproduces full daily refitting.
    dist
        Innovation distribution passed to ``arch`` (``"t"`` for fat tails).

    Returns
    -------
    pandas.Series
        Daily volatility forecasts (decimal units), indexed by the day each
        forecast is *for* (i.e. ``returns.index[window:]``).
    """
    out = _rolling_garch(returns, window, refit_every, dist)
    return out["sigma"].rename(returns.name)


def rolling_garch_forecasts(returns: pd.Series,
                            window: int = GARCH_WINDOW,
                            refit_every: int = 21,
                            dist: str = "t") -> pd.DataFrame:
    """Rolling forecasts of both volatility and fitted Student-t dof.

    Returns a DataFrame with columns ``sigma`` (daily volatility) and ``nu``
    (the fitted degrees of freedom, NaN for non-t distributions), indexed by
    the day each forecast is *for*. Feeding ``nu`` into the VaR estimator
    makes the loss distribution consistent with the volatility model's own
    innovations, rather than relying on a fixed dof.
    """
    return _rolling_garch(returns, window, refit_every, dist)


def garch_volatility_by_asset(returns: pd.DataFrame,
                              window: int = GARCH_WINDOW,
                              refit_every: int = 21,
                              dist: str = "t") -> dict[str, pd.Series]:
    """Run :func:`rolling_garch_volatility` for each column of ``returns``."""
    return {
        col: rolling_garch_volatility(returns[col], window, refit_every, dist)
        for col in returns.columns
    }
