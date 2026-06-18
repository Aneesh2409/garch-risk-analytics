"""Black-Scholes-Merton pricing and Greeks.

THE ONE CONVENTION (read this before touching anything below)
-------------------------------------------------------------
Every function in this module obeys a single set of units. The original
research notebook mixed daily and annualised vol across three different
pricing functions; that bug silently mispriced everything fed through the
"annualised" path by a factor of ~sqrt(252). To make that class of error
impossible, the rules live in exactly one place:

1. Volatility (``sigma``) is ALWAYS annualised. A daily GARCH/realised sigma
   must be converted with :func:`annualise_vol` before it reaches any pricer.
2. Time is measured in TRADING DAYS, with ``TRADING_DAYS_PER_YEAR = 252``.
   ``days_to_expiry`` is a trading-day count; internally we convert to years
   via ``T_years = days / 252``. This keeps pricing consistent with a daily
   simulation that also steps one trading day at a time.
3. ``r`` is a continuously-compounded annual rate.

Greek units (stated so nobody has to reverse-engineer them):
    delta : dV/dS                         (per 1 unit of spot)
    gamma : d2V/dS2                       (per 1 unit of spot, squared)
    vega  : dV/dsigma per 1 vol POINT     (i.e. raw vega / 100)
    theta : dV/dt per TRADING DAY         (i.e. annual theta / 252)

If you need a different convention, convert at the call site -- do not add a
second convention in here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

TRADING_DAYS_PER_YEAR: int = 252

OptionType = Literal["call", "put"]


def annualise_vol(sigma_daily: float | np.ndarray,
                  periods: int = TRADING_DAYS_PER_YEAR) -> float | np.ndarray:
    """Convert a per-period (daily) volatility to an annualised one.

    GARCH and realised-vol estimators produce daily sigma. Pricers want
    annualised sigma. This is the *only* sanctioned bridge between them.
    """
    return sigma_daily * np.sqrt(periods)


def trading_days_to_years(days: float | np.ndarray,
                          periods: int = TRADING_DAYS_PER_YEAR) -> float | np.ndarray:
    """Convert a trading-day horizon to a year fraction."""
    return days / periods


@dataclass(frozen=True)
class Greeks:
    """Container for option Greeks in the conventions documented above."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float


def _d1_d2(S: float, K: float, T_years: float, sigma_annual: float, r: float
           ) -> tuple[float, float]:
    """Standard BSM d1/d2. Assumes positive, non-degenerate inputs."""
    vol_sqrt_t = sigma_annual * np.sqrt(T_years)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_annual ** 2) * T_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bsm_price(S: float, K: float, days_to_expiry: float, sigma_annual: float,
              r: float, option_type: OptionType) -> float:
    """Black-Scholes-Merton price of a European option (no dividends).

    Parameters
    ----------
    S, K
        Spot and strike, same currency units.
    days_to_expiry
        Trading days to expiry (see module convention).
    sigma_annual
        ANNUALISED volatility. Use :func:`annualise_vol` on daily sigma first.
    r
        Continuously-compounded annual risk-free rate.
    option_type
        ``"call"`` or ``"put"``.

    Returns
    -------
    float
        Present value of one contract. Degenerate inputs (expired or zero
        vol) collapse to discounted intrinsic value.
    """
    T = trading_days_to_years(days_to_expiry)
    if T <= 0 or sigma_annual <= 0 or S <= 0 or K <= 0:
        # Discounted intrinsic value -- the limit as vol/time -> 0.
        intrinsic = (S - K) if option_type == "call" else (K - S)
        return max(intrinsic, 0.0) * np.exp(-r * max(T, 0.0))

    d1, d2 = _d1_d2(S, K, T, sigma_annual, r)
    disc_k = K * np.exp(-r * T)
    if option_type == "call":
        return S * norm.cdf(d1) - disc_k * norm.cdf(d2)
    return disc_k * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_greeks(S: float, K: float, days_to_expiry: float, sigma_annual: float,
               r: float, option_type: OptionType) -> Greeks:
    """Price + Greeks for a European option, in the module's conventions.

    Vega is reported per 1 volatility point (raw vega / 100) and theta per
    trading day (annual theta / 252), which is how a risk desk reads them.
    """
    T = trading_days_to_years(days_to_expiry)
    price = bsm_price(S, K, days_to_expiry, sigma_annual, r, option_type)

    if T <= 0 or sigma_annual <= 0 or S <= 0 or K <= 0:
        return Greeks(price=price, delta=0.0, gamma=0.0, vega=0.0, theta=0.0)

    d1, d2 = _d1_d2(S, K, T, sigma_annual, r)
    pdf_d1 = norm.pdf(d1)
    sqrt_t = np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta_annual = (-S * pdf_d1 * sigma_annual / (2 * sqrt_t)
                        - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = -norm.cdf(-d1)
        theta_annual = (-S * pdf_d1 * sigma_annual / (2 * sqrt_t)
                        + r * K * np.exp(-r * T) * norm.cdf(-d2))

    gamma = pdf_d1 / (S * sigma_annual * sqrt_t)
    vega_raw = S * pdf_d1 * sqrt_t            # dV/dsigma (per 1.00 of vol)

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega_raw / 100.0,                 # per 1 vol POINT
        theta=theta_annual / TRADING_DAYS_PER_YEAR,  # per trading day
    )
