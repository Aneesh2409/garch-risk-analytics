"""Central configuration: model constants and the option portfolio definition.

A single home for every model constant and the option book, so nothing is
hard-coded inline and the portfolio is defined once, with types.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pricing import OptionType

# --- Market / model constants -------------------------------------------------
RISK_FREE_RATE: float = 0.02          # continuously-compounded annual rate
GARCH_WINDOW: int = 365               # rolling estimation window (trading days)
REALISED_VOL_WINDOW: int = 20         # window for realised-vol benchmark
N_SIMULATIONS: int = 10_000           # Monte-Carlo paths
RANDOM_SEED: int = 42

ASSETS: tuple[str, ...] = ("S&P500", "NASDAQ", "BTC-USD")

# Yahoo Finance tickers for each asset.
TICKERS: dict[str, str] = {
    "S&P500": "^GSPC",
    "NASDAQ": "^IXIC",
    "BTC-USD": "BTC-USD",
}

# Data-layer defaults.
LOOKBACK_YEARS: int = 10        # history pulled when no explicit dates given
STRIP_WEEKENDS: bool = True     # True: align to the equity trading calendar;
#                                 False: keep BTC's 7-day calendar, forward-
#                                 filling equities across weekends/holidays.


@dataclass(frozen=True)
class OptionPosition:
    """A single option line in the book.

    ``moneyness`` is the strike as a multiple of the spot AT INCEPTION
    (K = moneyness * S_0). It is resolved to a fixed strike once, on the
    first day of the horizon, and never re-floated -- see :mod:`greeks`.
    ``days_to_expiry`` is a TRADING-day count (252/yr convention).
    A negative ``quantity`` denotes a short position.
    """
    id: str
    underlying: str
    option_type: OptionType
    moneyness: float
    days_to_expiry: int
    quantity: int


# The option book: six lines across the three underlyings.
DEFAULT_PORTFOLIO: tuple[OptionPosition, ...] = (
    OptionPosition("O1", "S&P500", "call", 1.00, 30, 5),    # ATM call
    OptionPosition("O2", "S&P500", "put", 0.95, 60, 2),     # 5% OTM put
    OptionPosition("O3", "NASDAQ", "put", 1.00, 45, -3),    # short ATM put
    OptionPosition("O4", "NASDAQ", "call", 1.10, 90, 4),    # 10% OTM call
    OptionPosition("O5", "BTC-USD", "call", 0.95, 20, 1),   # 5% ITM call
    OptionPosition("O6", "BTC-USD", "put", 0.90, 30, 2),    # 10% OTM put
)
