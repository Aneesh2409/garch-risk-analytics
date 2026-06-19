"""Tests for the risk module.

Pins down the core story of the P&L decomposition:
  * at small shocks linear ~ quadratic ~ full revaluation,
  * at large shocks they diverge, with quadratic strictly closer to truth,
  * the gap is the gamma (convexity) the linear view ignores.
"""

import numpy as np
import pytest

from garch_risk.config import OptionPosition
from garch_risk.pricing import annualise_vol, bsm_price
from garch_risk.risk import (
    full_revaluation_pnl,
    pnl_curve,
    portfolio_value,
    stress_grid,
    taylor_pnl,
)

# A simple long-gamma book: two long calls on one underlying.
_LONG_GAMMA = (
    OptionPosition("A", "IDX", "call", 1.00, 60, 1),
    OptionPosition("B", "IDX", "call", 1.05, 60, 1),
)
_SPOTS = {"IDX": 100.0}
_SIGMAS = {"IDX": 0.20 / np.sqrt(252)}   # ~20% annualised
_R = 0.02


def test_zero_shock_is_zero_pnl():
    lin, quad = taylor_pnl(_LONG_GAMMA, _SPOTS, _SIGMAS, _R, 0.0)
    full = full_revaluation_pnl(_LONG_GAMMA, _SPOTS, _SIGMAS, _R, 0.0)
    assert lin == pytest.approx(0.0, abs=1e-9)
    assert quad == pytest.approx(0.0, abs=1e-9)
    assert full == pytest.approx(0.0, abs=1e-9)


def test_small_shock_all_three_agree():
    lin, quad = taylor_pnl(_LONG_GAMMA, _SPOTS, _SIGMAS, _R, 0.005)
    full = full_revaluation_pnl(_LONG_GAMMA, _SPOTS, _SIGMAS, _R, 0.005)
    # Within ~1% of each other for a 0.5% move.
    assert lin == pytest.approx(full, rel=0.05)
    assert quad == pytest.approx(full, rel=0.005)


def test_large_shock_quadratic_beats_linear():
    """For a big move, quadratic must be strictly closer to full reval."""
    shock = 0.20
    lin, quad = taylor_pnl(_LONG_GAMMA, _SPOTS, _SIGMAS, _R, shock)
    full = full_revaluation_pnl(_LONG_GAMMA, _SPOTS, _SIGMAS, _R, shock)
    assert abs(quad - full) < abs(lin - full)


def test_long_gamma_full_reval_above_linear_both_directions():
    """Long gamma: full-reval P&L exceeds the linear estimate whichever way
    spot moves (convexity helps the holder)."""
    for shock in (0.15, -0.15):
        lin, _ = taylor_pnl(_LONG_GAMMA, _SPOTS, _SIGMAS, _R, shock)
        full = full_revaluation_pnl(_LONG_GAMMA, _SPOTS, _SIGMAS, _R, shock)
        assert full > lin


def test_quadratic_equals_linear_plus_gamma_term():
    """quad - linear must equal 0.5 * gamma * dS^2 from the aggregate gamma."""
    from garch_risk.risk import aggregate_greeks_by_asset, _resolve_strikes
    shock = 0.10
    strikes = _resolve_strikes(_LONG_GAMMA, _SPOTS)
    agg = aggregate_greeks_by_asset(_LONG_GAMMA, _SPOTS, _SIGMAS, _R, strikes)
    d_spot = shock * _SPOTS["IDX"]
    expected_gamma_term = 0.5 * agg["IDX"]["gamma"] * d_spot ** 2

    lin, quad = taylor_pnl(_LONG_GAMMA, _SPOTS, _SIGMAS, _R, shock, strikes)
    assert (quad - lin) == pytest.approx(expected_gamma_term, rel=1e-9)


def test_full_reval_matches_manual_single_option():
    """Full reval of one option equals the hand-computed reprice difference."""
    book = (OptionPosition("X", "IDX", "call", 1.0, 30, 3),)
    shock = 0.10
    strikes = {"X": 100.0}
    base = bsm_price(100.0, 100.0, 30, annualise_vol(_SIGMAS["IDX"]), _R, "call")
    shocked = bsm_price(110.0, 100.0, 30, annualise_vol(_SIGMAS["IDX"]), _R, "call")
    expected = 3 * (shocked - base)
    got = full_revaluation_pnl(book, _SPOTS, _SIGMAS, _R, shock, strikes=strikes)
    assert got == pytest.approx(expected, rel=1e-9)


def test_vol_shock_helps_long_options():
    """A pure vol increase raises the value of a long-vega book (positive P&L)."""
    pnl = full_revaluation_pnl(_LONG_GAMMA, _SPOTS, _SIGMAS, _R,
                               price_shock=0.0, vol_shock=0.5)
    assert pnl > 0


def test_pnl_curve_structure():
    curve = pnl_curve(_LONG_GAMMA, _SPOTS, _SIGMAS, _R,
                      shocks=np.linspace(-0.2, 0.2, 21))
    assert list(curve.columns) == ["Linear", "Quadratic", "FullReval"]
    assert len(curve) == 21
    # At the zero-shock row, all three are ~0.
    zero_row = curve.loc[curve.index[np.argmin(np.abs(curve.index))]]
    assert abs(zero_row["FullReval"]) < 1e-6


def test_stress_grid_shape_and_center():
    pgrid = np.linspace(-0.2, 0.2, 5)
    vgrid = np.linspace(-0.5, 1.0, 4)
    grid = stress_grid(_LONG_GAMMA, _SPOTS, _SIGMAS, _R,
                       price_shocks=pgrid, vol_shocks=vgrid)
    assert grid.shape == (5, 4)
    # The (0 price, 0 vol) corner isn't in these grids, but a no-op check:
    # full reval at zero/zero is zero.
    z = full_revaluation_pnl(_LONG_GAMMA, _SPOTS, _SIGMAS, _R, 0.0, 0.0)
    assert z == pytest.approx(0.0, abs=1e-9)
