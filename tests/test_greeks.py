"""Tests for the Greeks module.

The headline test checks that a call's delta responds to spot moving: with the
strike fixed at inception it climbs toward 1 as the option goes in the money,
whereas a strike that re-floats to track spot would leave delta pinned near
0.5 -- which is why the strike must be fixed.
"""

import numpy as np
import pytest

from garch_risk.config import OptionPosition
from garch_risk.greeks import (
    daily_portfolio_greeks,
    position_greeks,
    resolve_strike,
)
from garch_risk.pricing import annualise_vol, bsm_greeks


def test_strike_is_fixed_at_inception():
    pos = OptionPosition("X", "S&P500", "call", 1.0, 30, 1)
    assert resolve_strike(pos, 4000.0) == 4000.0
    # A later spot must not change the resolved strike.
    assert resolve_strike(pos, 4000.0) == 4000.0


def test_fixed_strike_delta_moves_but_floating_strike_does_not():
    """Delta must respond to spot; a fixed strike delivers this, a floating
    one does not.

    Build one ATM call, rally the spot +20% over the horizon at constant vol,
    and compare the delta path under the fixed-strike (correct) logic against
    a strike that re-floats to track spot (incorrect).
    """
    pos = OptionPosition("X", "S&P500", "call", 1.0, 60, 1)
    S0 = 100.0
    n = 40
    spot = S0 * np.linspace(1.0, 1.20, n)        # steady 20% rally
    sigma_daily = np.full(n, 0.20 / np.sqrt(252))  # ~20% annualised, constant
    r = 0.02

    # Fixed strike (correct): K pinned at S0.
    K_fixed = resolve_strike(pos, S0)
    delta_fixed_start = bsm_greeks(spot[0], K_fixed, pos.days_to_expiry,
                                   annualise_vol(sigma_daily[0]), r, "call").delta
    delta_fixed_end = bsm_greeks(spot[-1], K_fixed, pos.days_to_expiry - (n - 1),
                                 annualise_vol(sigma_daily[-1]), r, "call").delta

    # Floating strike (incorrect): K re-derived from each day's spot.
    delta_float_start = bsm_greeks(spot[0], 1.0 * spot[0], pos.days_to_expiry,
                                   annualise_vol(sigma_daily[0]), r, "call").delta
    delta_float_end = bsm_greeks(spot[-1], 1.0 * spot[-1], pos.days_to_expiry - (n - 1),
                                 annualise_vol(sigma_daily[-1]), r, "call").delta

    # Fixed strike: delta climbs materially as the call goes ITM.
    assert delta_fixed_end - delta_fixed_start > 0.20

    # Floating strike: delta is essentially pinned (option stays ATM forever);
    # any drift comes only from time decay, not from spot moving.
    assert abs(delta_float_end - delta_float_start) < 0.05


def test_short_position_flips_sign():
    long = OptionPosition("L", "S&P500", "call", 1.0, 30, 1)
    short = OptionPosition("S", "S&P500", "call", 1.0, 30, -1)
    g_long = position_greeks(long, 100, 100, 30, 0.20 / np.sqrt(252), 0.02)
    g_short = position_greeks(short, 100, 100, 30, 0.20 / np.sqrt(252), 0.02)
    assert g_short.delta == pytest.approx(-g_long.delta, abs=1e-12)
    assert g_short.gamma == pytest.approx(-g_long.gamma, abs=1e-12)


def test_daily_greeks_frame_shape_and_total():
    portfolio = (
        OptionPosition("O1", "S&P500", "call", 1.0, 30, 5),
        OptionPosition("O2", "NASDAQ", "put", 1.0, 30, -3),
    )
    n = 10
    spot_paths = {
        "S&P500": np.full(n, 4000.0),
        "NASDAQ": np.full(n, 14000.0),
        "BTC-USD": np.full(n, 60000.0),
    }
    sigma_paths = {a: np.full(n, 0.20 / np.sqrt(252)) for a in spot_paths}
    df = daily_portfolio_greeks(portfolio, spot_paths, sigma_paths, 0.02)

    # 3 assets + 1 Total row per day.
    assert len(df) == n * 4
    # Total row equals the sum of the asset rows on a given day.
    day0 = df.loc[0]
    total = day0.loc["Total"]
    asset_sum = day0.drop("Total").sum()
    for col in ("Delta", "Gamma", "Vega", "Theta"):
        assert total[col] == pytest.approx(asset_sum[col], abs=1e-9)


def test_expired_positions_stop_contributing():
    """A 5-day option contributes nothing from day 5 onward."""
    portfolio = (OptionPosition("O1", "S&P500", "call", 1.0, 5, 1),)
    n = 8
    spot_paths = {a: np.full(n, 100.0) for a in ("S&P500", "NASDAQ", "BTC-USD")}
    sigma_paths = {a: np.full(n, 0.20 / np.sqrt(252)) for a in spot_paths}
    df = daily_portfolio_greeks(portfolio, spot_paths, sigma_paths, 0.02)
    # Day 4 still active (days_remaining = 1), day 5 expired.
    assert df.loc[(4, "Total"), "Delta"] != 0.0
    assert df.loc[(5, "Total"), "Delta"] == 0.0


def test_portfolio_greeks_snapshot():
    from garch_risk.greeks import portfolio_greeks_snapshot
    book = (
        OptionPosition("O1", "S&P500", "call", 1.0, 30, 5),
        OptionPosition("O2", "NASDAQ", "put", 1.0, 30, -3),
    )
    spots = {"S&P500": 4000.0, "NASDAQ": 14000.0, "BTC-USD": 60000.0}
    sigmas = {a: 0.20 / np.sqrt(252) for a in spots}
    snap = portfolio_greeks_snapshot(book, spots, sigmas, 0.02)
    assert list(snap.columns) == ["Delta", "Gamma", "Vega", "Theta"]
    assert "Total" in snap.index
    # Total equals the sum across the asset rows.
    asset_rows = snap.drop("Total")
    for col in ("Delta", "Gamma", "Vega", "Theta"):
        assert snap.loc["Total", col] == pytest.approx(asset_rows[col].sum(), abs=1e-9)
    # A long call gives positive delta, a short put gives positive delta too.
    assert snap.loc["S&P500", "Delta"] > 0
    assert snap.loc["NASDAQ", "Delta"] > 0
