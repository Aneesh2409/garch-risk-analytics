"""Tests for the data layer.

The network download is not exercised here; instead a synthetic price source
is injected so the calendar-alignment and return logic is tested deterministically.
"""

import numpy as np
import pandas as pd
import pytest

from garch_risk.data import align_calendar, load_returns, log_returns


def _synthetic_prices_with_weekends():
    """7-day calendar: equities are NaN on weekends, BTC trades every day."""
    idx = pd.date_range("2022-01-03", periods=14, freq="D")  # Mon -> Sun x2
    is_weekend = idx.weekday >= 5
    eq = np.where(is_weekend, np.nan, np.linspace(100, 113, 14))
    btc = np.linspace(40000, 41300, 14)  # trades all 14 days
    return pd.DataFrame({"S&P500": eq, "NASDAQ": eq, "BTC-USD": btc}, index=idx)


def test_log_returns_hand_computed():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    r = log_returns(prices)
    assert r["A"].iloc[0] == pytest.approx(np.log(110 / 100))
    assert r["A"].iloc[1] == pytest.approx(np.log(99 / 110))
    assert len(r) == 2  # first row dropped


def test_strip_weekends_removes_weekend_rows():
    prices = _synthetic_prices_with_weekends()
    aligned = align_calendar(prices, strip_weekends=True)
    # No Saturday/Sunday survives, and nothing is NaN.
    assert (aligned.index.weekday < 5).all()
    assert not aligned.isna().any().any()


def test_keep_weekends_forward_fills_equities():
    prices = _synthetic_prices_with_weekends()
    aligned = align_calendar(prices, strip_weekends=False)
    # Weekend rows are retained...
    assert (aligned.index.weekday >= 5).any()
    # ...and equities are forward-filled, so a weekend equity return is zero
    # while BTC still moves.
    rets = log_returns(aligned)
    weekend_rets = rets[rets.index.weekday >= 5]
    assert (weekend_rets["S&P500"].abs() < 1e-12).all()
    assert (weekend_rets["BTC-USD"].abs() > 0).any()


def test_load_returns_with_injected_downloader():
    prices = _synthetic_prices_with_weekends()

    def fake_downloader(**kwargs):
        return prices

    rets = load_returns(strip_weekends=True, downloader=fake_downloader)
    assert list(rets.columns) == ["S&P500", "NASDAQ", "BTC-USD"]
    assert (rets.index.weekday < 5).all()
    assert not rets.isna().any().any()


def test_alignment_sorts_index():
    prices = _synthetic_prices_with_weekends().iloc[::-1]  # reversed
    aligned = align_calendar(prices, strip_weekends=True)
    assert aligned.index.is_monotonic_increasing
