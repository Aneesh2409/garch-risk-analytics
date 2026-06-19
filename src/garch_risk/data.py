"""Data layer: price download, calendar alignment, and log returns.

I/O is isolated from logic. :func:`download_prices` is the only function that
touches the network; everything else operates on a plain price DataFrame, so
the cleaning and return-construction logic is deterministic and testable
without a live data feed.

CALENDAR ALIGNMENT
------------------
Equities trade ~252 days a year; BTC trades every day. The two have to share
one index before returns mean anything across assets. Two modes, set by
``strip_weekends``:

* ``True`` (default): keep only dates every asset genuinely traded -- in
  practice the equity calendar. BTC's weekend and holiday bars are dropped.
  Weekend gap risk in BTC is therefore not captured; this is a deliberate,
  documented simplification.
* ``False``: keep BTC's near-daily calendar and forward-fill equity prices
  across non-trading days. Equity weekend returns are then zero by
  construction, while BTC keeps its real weekend moves.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .config import LOOKBACK_YEARS, STRIP_WEEKENDS, TICKERS


def _default_dates(lookback_years: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=lookback_years)
    return start, end


def download_prices(tickers: dict[str, str] = TICKERS,
                    start: str | pd.Timestamp | None = None,
                    end: str | pd.Timestamp | None = None,
                    lookback_years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    Returns a DataFrame whose columns are the friendly asset names (the keys
    of ``tickers``), indexed by date. This is the only networked function in
    the package.
    """
    import yfinance as yf  # imported lazily so the rest of the package has no
    #                        hard runtime dependency on a network library.

    if start is None or end is None:
        default_start, default_end = _default_dates(lookback_years)
        start = start or default_start
        end = end or default_end

    symbols = list(tickers.values())
    raw = yf.download(symbols, start=start, end=end,
                      auto_adjust=True, progress=False)

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    # Map Yahoo symbols back to friendly names, preserving config order.
    inverse = {sym: name for name, sym in tickers.items()}
    close = close.rename(columns=inverse)
    return close[list(tickers.keys())]


def align_calendar(prices: pd.DataFrame,
                   strip_weekends: bool = STRIP_WEEKENDS) -> pd.DataFrame:
    """Put all assets on one calendar (see module docstring for the modes)."""
    prices = prices.sort_index()
    if strip_weekends:
        # Intersection of genuine observations -> the equity calendar.
        return prices.dropna(how="any")
    # Union calendar; carry equity prices across their non-trading days.
    return prices.ffill().dropna(how="any")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, with the first (undefined) row dropped."""
    return np.log(prices / prices.shift(1)).dropna(how="any")


def load_returns(tickers: dict[str, str] = TICKERS,
                 start: str | pd.Timestamp | None = None,
                 end: str | pd.Timestamp | None = None,
                 lookback_years: int = LOOKBACK_YEARS,
                 strip_weekends: bool = STRIP_WEEKENDS,
                 downloader: Callable[..., pd.DataFrame] = download_prices
                 ) -> pd.DataFrame:
    """End-to-end: download prices, align the calendar, return log returns.

    ``downloader`` is injectable so the pipeline can be exercised with a
    synthetic price source instead of a live feed.
    """
    prices = downloader(tickers=tickers, start=start, end=end,
                        lookback_years=lookback_years)
    aligned = align_calendar(prices, strip_weekends=strip_weekends)
    return log_returns(aligned)
