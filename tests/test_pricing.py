"""Tests for the pricing module.

These exist mainly to make the original notebook's two silent bugs
impossible to reintroduce:
  * mismatched vol units across pricers (daily vs annualised), and
  * accidental drift in the Greek conventions.
"""

import numpy as np
import pytest

from garch_risk.pricing import (
    annualise_vol,
    bsm_greeks,
    bsm_price,
    TRADING_DAYS_PER_YEAR,
)


def test_textbook_call_value():
    """Hull's canonical example: S=K=100, T=1yr, r=5%, sigma=20% -> ~10.4506."""
    price = bsm_price(
        S=100, K=100, days_to_expiry=TRADING_DAYS_PER_YEAR,
        sigma_annual=0.20, r=0.05, option_type="call",
    )
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_textbook_put_value():
    """Same parameters, put side: ~5.5735."""
    price = bsm_price(
        S=100, K=100, days_to_expiry=TRADING_DAYS_PER_YEAR,
        sigma_annual=0.20, r=0.05, option_type="put",
    )
    assert price == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    """C - P must equal S - K*exp(-rT) for any valid inputs."""
    S, K, r = 105.0, 100.0, 0.03
    days = 120
    sigma = 0.25
    call = bsm_price(S, K, days, sigma, r, "call")
    put = bsm_price(S, K, days, sigma, r, "put")
    T = days / TRADING_DAYS_PER_YEAR
    assert (call - put) == pytest.approx(S - K * np.exp(-r * T), abs=1e-8)


def test_vol_convention_guard():
    """The original bug: feeding DAILY sigma into an annualised pricer.

    A daily sigma of ~0.0126 corresponds to ~20% annualised. If a caller
    forgets to annualise, the price collapses toward intrinsic. This test
    documents the size of that error so the convention can never silently
    drift again.
    """
    daily_sigma = 0.20 / np.sqrt(TRADING_DAYS_PER_YEAR)  # ~1.26% per day
    correct = bsm_price(100, 100, TRADING_DAYS_PER_YEAR,
                        annualise_vol(daily_sigma), 0.05, "call")
    wrong = bsm_price(100, 100, TRADING_DAYS_PER_YEAR,
                      daily_sigma, 0.05, "call")  # forgot to annualise
    assert correct == pytest.approx(10.4506, abs=1e-3)
    # Un-annualised, the ATM call collapses toward its discounted forward
    # intrinsic (~4.88) -- a ~53% underprice. Documented so the magnitude of
    # the original bug is on the record, not just its existence.
    assert wrong == pytest.approx(4.877, abs=1e-2)
    assert wrong < 0.55 * correct


def test_call_delta_bounds():
    """Call delta in (0,1); deep ITM -> ~1, deep OTM -> ~0."""
    deep_itm = bsm_greeks(200, 100, 60, 0.2, 0.03, "call").delta
    deep_otm = bsm_greeks(50, 100, 60, 0.2, 0.03, "call").delta
    assert deep_itm == pytest.approx(1.0, abs=1e-3)
    assert deep_otm == pytest.approx(0.0, abs=1e-3)


def test_put_call_delta_relationship():
    """call_delta - put_delta == 1 (no dividends)."""
    g_call = bsm_greeks(105, 100, 90, 0.25, 0.03, "call")
    g_put = bsm_greeks(105, 100, 90, 0.25, 0.03, "put")
    assert (g_call.delta - g_put.delta) == pytest.approx(1.0, abs=1e-9)


def test_gamma_vega_shared_across_call_put():
    """Gamma and vega are identical for a call/put at the same strike."""
    g_call = bsm_greeks(105, 100, 90, 0.25, 0.03, "call")
    g_put = bsm_greeks(105, 100, 90, 0.25, 0.03, "put")
    assert g_call.gamma == pytest.approx(g_put.gamma, abs=1e-12)
    assert g_call.vega == pytest.approx(g_put.vega, abs=1e-12)


def test_expired_option_is_intrinsic():
    """Zero time to expiry collapses to intrinsic value with zero Greeks."""
    g = bsm_greeks(120, 100, 0, 0.2, 0.03, "call")
    assert g.price == pytest.approx(20.0, abs=1e-9)
    assert g.delta == 0.0 and g.gamma == 0.0
