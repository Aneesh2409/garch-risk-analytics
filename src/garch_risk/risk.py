"""Stress testing and P&L decomposition for the option book.

Two related questions about a sudden market move:

1. How much does the book make or lose? (:func:`full_revaluation_pnl`,
   :func:`stress_grid` -- joint price/volatility shocks.)
2. How well do the standard *approximations* to that P&L hold up?
   (:func:`taylor_pnl`, :func:`pnl_curve`.)

The second is the interesting one. A risk system that summarises an options
book by its delta alone is making a linear approximation to a curved payoff.
This module puts three estimates side by side:

* **Linear (delta-normal):** dV ~ sum(delta * dS). First order in spot.
* **Quadratic (delta-gamma):** adds 0.5 * sum(gamma * dS^2). Second order.
* **Full revaluation:** reprice every option at the shocked state -- the truth.

For small moves all three agree. As the shock grows the linear estimate peels
away first, the quadratic holds longer, and the gap between them is exactly the
risk a delta-only view misses. The P&L curve is price-only (volatility held at
base) so the three are directly comparable; volatility stress is handled by
full revaluation in :func:`stress_grid`, where there is no approximation to
compare against and so no unit ambiguity.

Strikes are resolved from the base spot (``K = moneyness * S_base``) and held
fixed across all shocks, consistent with :mod:`greeks`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OptionPosition
from .greeks import position_greeks, resolve_strike
from .pricing import annualise_vol, bsm_price

Spots = dict[str, float]
Sigmas = dict[str, float]
Strikes = dict[str, float]


def _resolve_strikes(portfolio: tuple[OptionPosition, ...],
                     base_spots: Spots) -> Strikes:
    return {pos.id: resolve_strike(pos, base_spots[pos.underlying])
            for pos in portfolio}


def portfolio_value(portfolio: tuple[OptionPosition, ...], spots: Spots,
                    sigmas_daily: Sigmas, r: float, strikes: Strikes) -> float:
    """Total (quantity-weighted) value of the book at a given market state."""
    total = 0.0
    for pos in portfolio:
        sigma_annual = annualise_vol(sigmas_daily[pos.underlying])
        px = bsm_price(spots[pos.underlying], strikes[pos.id],
                       pos.days_to_expiry, sigma_annual, r, pos.option_type)
        total += pos.quantity * px
    return total


def aggregate_greeks_by_asset(portfolio: tuple[OptionPosition, ...],
                              spots: Spots, sigmas_daily: Sigmas, r: float,
                              strikes: Strikes) -> dict[str, dict[str, float]]:
    """Quantity-weighted delta and gamma per underlying at the given state."""
    agg: dict[str, dict[str, float]] = {}
    for pos in portfolio:
        g = position_greeks(pos, spots[pos.underlying], strikes[pos.id],
                            pos.days_to_expiry, sigmas_daily[pos.underlying], r)
        bucket = agg.setdefault(pos.underlying, {"delta": 0.0, "gamma": 0.0})
        bucket["delta"] += g.delta
        bucket["gamma"] += g.gamma
    return agg


def full_revaluation_pnl(portfolio: tuple[OptionPosition, ...],
                         base_spots: Spots, base_sigmas: Sigmas, r: float,
                         price_shock: float, vol_shock: float = 0.0,
                         strikes: Strikes | None = None) -> float:
    """P&L from fully repricing the book under a joint price/vol shock.

    ``price_shock`` and ``vol_shock`` are relative: ``-0.1`` is a 10% drop in
    spot, ``0.5`` a 50% increase in volatility. The shock is instantaneous --
    time to expiry does not change.
    """
    strikes = strikes or _resolve_strikes(portfolio, base_spots)
    base = portfolio_value(portfolio, base_spots, base_sigmas, r, strikes)

    shocked_spots = {a: s * (1 + price_shock) for a, s in base_spots.items()}
    shocked_sigmas = {a: sig * (1 + vol_shock) for a, sig in base_sigmas.items()}
    shocked = portfolio_value(portfolio, shocked_spots, shocked_sigmas, r, strikes)
    return shocked - base


def taylor_pnl(portfolio: tuple[OptionPosition, ...], base_spots: Spots,
               base_sigmas: Sigmas, r: float, price_shock: float,
               strikes: Strikes | None = None) -> tuple[float, float]:
    """Linear and quadratic (delta / delta-gamma) P&L for a price shock.

    Returns ``(linear, quadratic)``. Both are price-only approximations; the
    quadratic adds the gamma term ``0.5 * gamma * dS^2`` to the linear one.
    """
    strikes = strikes or _resolve_strikes(portfolio, base_spots)
    agg = aggregate_greeks_by_asset(portfolio, base_spots, base_sigmas, r, strikes)

    linear = 0.0
    gamma_term = 0.0
    for asset, g in agg.items():
        d_spot = price_shock * base_spots[asset]
        linear += g["delta"] * d_spot
        gamma_term += 0.5 * g["gamma"] * d_spot ** 2
    return linear, linear + gamma_term


def pnl_curve(portfolio: tuple[OptionPosition, ...], base_spots: Spots,
              base_sigmas: Sigmas, r: float,
              shocks: np.ndarray | None = None,
              strikes: Strikes | None = None) -> pd.DataFrame:
    """Linear, quadratic and full-revaluation P&L across a range of price shocks.

    Volatility is held at base so the three columns are directly comparable.
    Indexed by price shock; columns ``Linear``, ``Quadratic``, ``FullReval``.
    """
    if shocks is None:
        shocks = np.linspace(-0.30, 0.30, 61)
    strikes = strikes or _resolve_strikes(portfolio, base_spots)

    rows = []
    for s in shocks:
        linear, quad = taylor_pnl(portfolio, base_spots, base_sigmas, r, s, strikes)
        full = full_revaluation_pnl(portfolio, base_spots, base_sigmas, r, s,
                                     vol_shock=0.0, strikes=strikes)
        rows.append({"PriceShock": s, "Linear": linear,
                     "Quadratic": quad, "FullReval": full})
    return pd.DataFrame(rows).set_index("PriceShock")


def stress_grid(portfolio: tuple[OptionPosition, ...], base_spots: Spots,
                base_sigmas: Sigmas, r: float,
                price_shocks: np.ndarray | None = None,
                vol_shocks: np.ndarray | None = None,
                strikes: Strikes | None = None) -> pd.DataFrame:
    """Full-revaluation P&L over a grid of joint price and volatility shocks.

    Returns a DataFrame whose rows are price shocks, columns are vol shocks,
    and cells are the book's P&L -- a stress surface.
    """
    if price_shocks is None:
        price_shocks = np.linspace(-0.20, 0.20, 9)
    if vol_shocks is None:
        vol_shocks = np.linspace(-0.50, 1.00, 7)
    strikes = strikes or _resolve_strikes(portfolio, base_spots)

    data = {
        round(float(v), 4): [
            full_revaluation_pnl(portfolio, base_spots, base_sigmas, r,
                                 float(p), float(v), strikes)
            for p in price_shocks
        ]
        for v in vol_shocks
    }
    return pd.DataFrame(data, index=[round(float(p), 4) for p in price_shocks])
