# GARCH Risk Analytics

A modular toolkit for **volatility forecasting, tail-risk backtesting, and
option-portfolio risk analytics** across equities and crypto (S&P 500, NASDAQ,
BTC-USD), built as a tested, installable Python package.

> **Status: in active development.** The core pricing and Greeks engine is
> complete and unit-tested; volatility, VaR/ES, and stress-testing modules are
> being ported in. Headline results and figures will be added once the full
> pipeline runs end-to-end.

## What it does

The pipeline tells one story end to end:

1. **Forecast volatility** with a rolling GJR-GARCH(1,1)-t model.
2. **Simulate** the return distribution by Monte Carlo.
3. **Estimate VaR / Expected Shortfall** and **backtest** them
   (Kupiec unconditional-coverage + Christoffersen independence tests).
4. **Price an option book** on the simulated paths and compute portfolio
   **Greeks** (delta, gamma, vega, theta) through time.
5. **Compare** GARCH-implied Greeks against realised-vol Greeks.
6. **Stress-test** the book under joint price/vol shocks, and show where a
   linear (delta-normal) risk approximation diverges from full revaluation.

## Conventions (read before extending)

These are enforced in code and tests; do not introduce competing ones.

- **Volatility is always annualised** at the pricing boundary. Daily sigma is
  bridged via `annualise_vol(sigma_daily) = sigma_daily * sqrt(252)`.
- **One day-count: 252 trading days = 1 year**, everywhere. `days_to_expiry`
  is a trading-day count. This keeps time-decay consistent with a daily
  simulation that steps one trading day at a time.
- **Greek units:** vega per 1 volatility point; theta per trading day.
- **Strikes are fixed at inception** (`K = moneyness * S_0`) and never
  re-floated.

### Known limitations (disclosed, not hidden)

- BTC-USD returns are aligned to the equity trading calendar (weekends
  stripped) for cross-asset consistency, so weekend gap risk is not captured.
- A clean free implied-vol benchmark for BTC was not sourceable; realised vol
  is used as the BTC input. Deribit's DVOL index is the production path.

## Layout

```
src/garch_risk/
    pricing.py      # Black-Scholes-Merton price + Greeks (one vol convention)
    config.py       # constants + the typed option portfolio
    greeks.py       # fixed-strike portfolio Greeks through time
    volatility.py   # rolling GJR-GARCH(1,1)-t forecasts        [in progress]
    var_es.py       # Monte-Carlo VaR/ES + Kupiec/Christoffersen [in progress]
    risk.py         # shocks + linear-vs-second-order PnL        [in progress]
tests/              # pytest suite
notebooks/          # narrative demos that import the package
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
