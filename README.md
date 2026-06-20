# GARCH Risk Analytics

A modular Python toolkit for **volatility forecasting, tail-risk backtesting,
and option-portfolio risk analytics** across equities and crypto (S&P 500,
NASDAQ, BTC-USD), built as a tested, installable package with two demonstration
notebooks.

## What it does

The pipeline runs one coherent story end to end:

1. **Forecast volatility** with a rolling GJR-GARCH(1,1)-t model (asymmetric,
   fat-tailed, periodically refit, no look-ahead).
2. **Estimate VaR / Expected Shortfall** by Monte Carlo and **backtest** them
   with Kupiec (coverage), Christoffersen (independence), and the joint
   conditional-coverage test.
3. **Evaluate the volatility forecast itself** with Mincer-Zarnowitz regression
   and QLIKE against a naive benchmark.
4. **Price a six-option book** and compute its **Greeks** (delta, gamma, vega,
   theta) from the GARCH volatility, tracked through time.
5. **Stress-test** the book under joint price/volatility shocks, and decompose
   P&L into linear (delta-normal), quadratic (delta-gamma), and full
   revaluation -- showing exactly where the approximations break down.

## Selected findings

These come out of the backtests over roughly ten years of daily data; the
notebooks reproduce them with figures.

- **GJR-GARCH absorbs equities' fat tails into the conditional volatility.** The
  fitted Student-t degrees of freedom come back large and unstable for the
  equity indices (their standardised residuals are near-Gaussian once volatility
  is conditioned out) but low and stable for BTC, which retains genuinely fat
  conditional tails. Using each asset's *fitted* dof -- rather than a single
  fixed value -- is what calibrates the crypto VaR cleanly.
- **The VaR dynamics are sound; the calibration is honest about its limits.**
  The independence test passes across assets and confidence levels (breaches are
  not clustered), so the conditional volatility tracks turbulence correctly. The
  equity indices nonetheless slightly over-breach -- a documented limitation of
  symmetric-innovation GARCH for negatively-skewed equity returns, addressable
  with a skewed-t innovation (noted as future work rather than tuned away).
- **The volatility forecast beats a naive benchmark.** On QLIKE -- the loss
  function robust to the noise in the realised-variance proxy -- the GARCH
  forecast outperforms a 20-day rolling-volatility estimate on every asset.
- **Linear risk views understate option-book convexity.** For a long-gamma book,
  a delta-normal P&L estimate diverges sharply from full revaluation in a large
  move; the delta-gamma correction tracks further but still fails in the deep
  tail. Only full revaluation is reliable there.

## Conventions

Enforced in code and tests; do not introduce competing ones.

- **Volatility is always annualised** at the pricing boundary
  (`annualise_vol(sigma_daily) = sigma_daily * sqrt(252)`).
- **One day-count: 252 trading days = 1 year**, everywhere. `days_to_expiry` is
  a trading-day count.
- **Greek units:** vega per 1 volatility point; theta per trading day.
- **Strikes are fixed at inception** (`K = moneyness * S_0`) and never re-floated.
- **VaR / ES** are left-tail returns (negative); a breach is a day whose return
  falls below the VaR threshold. No post-hoc scaling is applied to the estimates.

## Layout

```
src/garch_risk/
    pricing.py          # Black-Scholes price + Greeks (single vol convention)
    config.py           # constants + the typed option portfolio
    volatility.py       # rolling GJR-GARCH(1,1)-t forecasts + realised vol
    volatility_eval.py  # Mincer-Zarnowitz + QLIKE forecast evaluation
    var_es.py           # Monte-Carlo VaR/ES + Kupiec/Christoffersen backtests
    greeks.py           # fixed-strike portfolio Greeks (snapshot + through time)
    risk.py             # price/vol stress grid + linear/quadratic/full-reval PnL
    data.py             # download, calendar alignment, log returns
    plots.py            # matplotlib charts + combined risk dashboard
tests/                  # pytest suite (71 tests)
notebooks/
    01_volatility_and_var.ipynb       # volatility forecasting + VaR/ES + evaluation
    02_option_portfolio_risk.ipynb    # Greeks + stress testing
scripts/
    smoke_test.py       # end-to-end VaR backtest on real data
    portfolio_demo.py   # end-to-end portfolio Greeks + stress on real data
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # 71 tests
```

Reproduce the headline results on live data:

```bash
python scripts/smoke_test.py        # volatility -> VaR/ES -> backtests
python scripts/portfolio_demo.py    # portfolio Greeks -> stress P&L
```

The notebooks pull data from Yahoo Finance and take a few minutes to run (the
rolling GARCH is re-estimated across the full history).

## Disclosed limitations

- Equity VaR slightly over-breaches at the tested confidence levels; the
  indicated fix is a skewed-t innovation to capture left-skew.
- BTC weekend bars are dropped (equity-calendar alignment), so weekend gap risk
  is not modelled; no free implied-volatility benchmark was available for BTC.
- The realised-variance proxy is the squared daily return, which is unbiased but
  noisy; intraday realised variance would sharpen the Mincer-Zarnowitz
  evaluation.
- Model-structure choices (estimation window, refit cadence, GARCH variant) are
  fixed rather than selected by walk-forward; walk-forward specification
  selection is the principled next step against overfitting.

## License

MIT -- see [LICENSE](LICENSE).
