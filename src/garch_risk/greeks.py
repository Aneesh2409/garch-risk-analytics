"""Portfolio Greeks through time -- with the strike pinned at inception.

THE BUG THIS MODULE FIXES
-------------------------
The original notebook computed the strike as ``K = moneyness * S_t`` *inside
the daily loop*, i.e. it re-derived the strike from each day's spot. That makes
every option permanently at-the-money: its delta sits near 0.5 and never
responds to the underlying moving, which defeats the entire point of tracking
Greeks over time. (Tellingly, the notebook's mark-to-market function got it
right -- ``K = moneyness * S_0`` -- so the same notebook held both versions.)

Here the strike is resolved ONCE, on the first day of the horizon, via
:func:`resolve_strike`, and stays fixed. Delta and gamma then evolve as the
spot drifts away from that fixed strike, which is the whole story we want to
show.

Volatility inputs are DAILY (the natural output of the GARCH / realised-vol
estimators); annualisation happens here, at the single point where we call
into the pricer, per the convention documented in :mod:`pricing`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ASSETS, OptionPosition
from .pricing import Greeks, annualise_vol, bsm_greeks

_GREEK_COLS = ("Delta", "Gamma", "Vega", "Theta")


def resolve_strike(pos: OptionPosition, spot_at_inception: float) -> float:
    """Pin the strike to inception spot. Called once, never re-floated."""
    return pos.moneyness * spot_at_inception


def position_greeks(pos: OptionPosition, S: float, strike: float,
                    days_remaining: float, sigma_daily: float,
                    r: float) -> Greeks:
    """Quantity-scaled Greeks for one position at a given state.

    ``sigma_daily`` is annualised internally before pricing. The returned
    Greeks are multiplied by the (signed) position quantity, so a short
    position contributes negative delta/gamma/etc.
    """
    g = bsm_greeks(S, strike, days_remaining, annualise_vol(sigma_daily),
                   r, pos.option_type)
    q = pos.quantity
    return Greeks(
        price=g.price * q,
        delta=g.delta * q,
        gamma=g.gamma * q,
        vega=g.vega * q,
        theta=g.theta * q,
    )


def daily_portfolio_greeks(portfolio: tuple[OptionPosition, ...],
                           spot_paths: dict[str, np.ndarray],
                           sigma_paths: dict[str, np.ndarray],
                           r: float) -> pd.DataFrame:
    """Greeks for the whole book, per day, per asset, plus a 'Total' row.

    Parameters
    ----------
    portfolio
        The option book.
    spot_paths
        ``asset -> 1D array`` of spot through time (a realised path or a
        representative simulated path). ``spot_paths[asset][0]`` is the
        inception spot used to fix every strike on that asset.
    sigma_paths
        ``asset -> 1D array`` of DAILY volatility through time, aligned to
        ``spot_paths``.
    r
        Annual risk-free rate.

    Returns
    -------
    pandas.DataFrame
        MultiIndexed by ``(Day, Asset)`` with columns Delta/Gamma/Vega/Theta.
        Strikes are fixed at inception; positions past expiry stop
        contributing.
    """
    # Fix every strike once, up front, from each asset's inception spot.
    strikes = {
        pos.id: resolve_strike(pos, spot_paths[pos.underlying][0])
        for pos in portfolio
    }

    horizon = min(len(spot_paths[a]) for a in spot_paths)
    rows: list[dict] = []

    for t in range(horizon):
        by_asset = {a: dict.fromkeys(_GREEK_COLS, 0.0) for a in ASSETS}

        for pos in portfolio:
            days_remaining = pos.days_to_expiry - t   # trading days
            if days_remaining <= 0:
                continue
            S_t = spot_paths[pos.underlying][t]
            sigma_t = sigma_paths[pos.underlying][t]
            g = position_greeks(pos, S_t, strikes[pos.id],
                                days_remaining, sigma_t, r)
            acc = by_asset[pos.underlying]
            acc["Delta"] += g.delta
            acc["Gamma"] += g.gamma
            acc["Vega"] += g.vega
            acc["Theta"] += g.theta

        for asset in ASSETS:
            rows.append({"Day": t, "Asset": asset, **by_asset[asset]})
        total = {c: sum(by_asset[a][c] for a in ASSETS) for c in _GREEK_COLS}
        rows.append({"Day": t, "Asset": "Total", **total})

    return pd.DataFrame(rows).set_index(["Day", "Asset"])
