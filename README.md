# GARCH Risk Analytics

A modular Python toolkit for **volatility forecasting, tail-risk backtesting,
and option-portfolio risk analytics** across equities and crypto (S&P 500,
NASDAQ, BTC-USD), built as a tested, installable package with three
demonstration notebooks. Model-specification choices are not asserted but
**validated out of sample** by walk-forward (rolling-origin) selection.

## What it does

The pipeline runs one coherent story end to end:

1. **Forecast volatility** with a rolling GJR-GARCH(1,1) model (asymmetric,
   fat-tailed, periodically refit, no look-ahead), with Student-t or Hansen
   skew-t innovations.
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

Wrapping the volatility and VaR stages, a **walk-forward validation layer**
selects the specification on training folds only, backtests pooled
out-of-sample breaches per asset, and attributes any improvement against a
forced baseline -- the principled guard against overfitting the specification.

## Selected findings

These come out of the backtests over roughly ten years of daily data; the
notebooks reproduce them with figures.

- **GJR-GARCH absorbs equities' fat tails into the conditional volatility.** The
  fitted Student-t degrees of freedom come back large and unstable for the
  equity indices (their standardised residuals are near-Gaussian once volatility
  is conditioned out) but low and stable for BTC, which retains genuinely fat
  conditional tails. Using each asset's *fitted* dof -- rather than a single
  fixed value -- is what calibrates the crypto VaR cleanly.
- **The equity over-breach is resolved by a skew-t innovation, validated out of
  sample.** Walk-forward selection chooses Hansen's skew-t for both equity
  indices in *every* fold and plain Student-t for BTC in every fold. On a
  controlled out-of-sample contrast, the original symmetric-t baseline
  over-breaches the equities (the S&P fails Kupiec at 95%) while the selected
  skew-t restores nominal coverage -- chosen by BIC on training data alone, not
  tuned. BTC, already well calibrated, is the negative control: its baseline and
  selected runs coincide exactly.
- **The selection does not overfit.** Re-selecting the specification every
  calendar year -- through COVID, the 2022 sell-off, and the 2023-25 recovery --
  reproduces the same choice each time (zero frozen-versus-reselected gap), and
  pooled breach rates move under half a percentage point across refit cadences.
- **The fix is a tail fix, not a variance fix.** Skew-t reshapes the VaR
  quantile while leaving the conditional variance -- and therefore QLIKE and the
  Greeks -- essentially unchanged.
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
- **Walk-forward** uses a sliding 1260-day training window, calendar-year
  out-of-sample folds, BIC selection (AIC reported as a sensitivity), and never
  lets a forecast or a selection see a day it is later scored on.

## Layout

```
src/garch_risk/
    pricing.py          # Black-Scholes price + Greeks (single vol convention)
    config.py           # constants + the typed option portfolio
    volatility.py       # rolling GJR-GARCH(p,o,q) forecasts (t / skew-t) + realised vol
    volatility_eval.py  # Mincer-Zarnowitz + QLIKE forecast evaluation
    var_es.py           # Monte-Carlo VaR/ES (normal / t / skew-t) + Kupiec/Christoffersen
    greeks.py           # fixed-strike portfolio Greeks (snapshot + through time)
    risk.py             # price/vol stress grid + linear/quadratic/full-reval PnL
    walkforward.py      # rolling-origin spec selection + pooled OOS backtests
    data.py             # download, calendar alignment, log returns
    plots.py            # matplotlib charts + combined risk dashboard
tests/                  # pytest suite (97 tests)
notebooks/
    01_volatility_and_var.ipynb       # volatility forecasting + VaR/ES + evaluation
    02_option_portfolio_risk.ipynb    # Greeks + stress testing
    03_walkforward_validation.ipynb   # rolling-origin spec selection + OOS validation
scripts/
    smoke_test.py       # end-to-end VaR backtest on real data
    portfolio_demo.py   # end-to-end portfolio Greeks + stress on real data
    run_walkforward.py  # end-to-end walk-forward spec selection + OOS backtests
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # 97 tests
```

Reproduce the headline results on live data:

```bash
python scripts/smoke_test.py        # volatility -> VaR/ES -> backtests
python scripts/portfolio_demo.py    # portfolio Greeks -> stress P&L
python scripts/run_walkforward.py   # rolling-origin spec selection + OOS validation
```

The notebooks pull data from Yahoo Finance and take a few minutes to run (the
rolling GARCH is re-estimated across the full history).

## Walk-forward validation

`walkforward.py` addresses a basic question about the base pipeline: were the
fixed specification and window fitting the process, or the sample?

- **Sliding window.** A 1260-day (about five-year) training window, both ends
  moving, so estimates stay comparable across folds and the window cannot
  memorise the whole history.
- **Calendar-year folds, annual reselection.** The specification is chosen once
  per out-of-sample year, on the window ending the day before that year begins,
  and held across it. Folds are anchored to dates, so a re-pull that shifts the
  start by a few rows leaves fold identity intact.
- **Small candidate grid.** Innovation distribution {t, skew-t} crossed with
  ARCH order p {1, 2}, holding o=1, q=1. Two clean axes -- one for tail shape,
  one for the marginal value of an extra ARCH lag that the criterion is expected
  to reject. (An o=2 leverage-order study is kept deliberately separate.)
- **Three runs, one set of out-of-sample days.** A forced *baseline*
  (symmetric-t everywhere), a *frozen* spec (selected on the first window and
  held), and a *reselected* spec (re-chosen each year). The baseline-vs-frozen
  contrast attributes the improvement to the distribution; the
  frozen-vs-reselected contrast measures whether adaptivity earns its keep.

`scripts/run_walkforward.py` runs all three assets and writes a text summary;
`notebooks/03_walkforward_validation.ipynb` reproduces the figures.

## Disclosed limitations

- BTC weekend bars are dropped (equity-calendar alignment), so weekend gap risk
  is not modelled; no free implied-volatility benchmark was available for BTC.
- The realised-variance proxy is the squared daily return, which is unbiased but
  noisy; intraday realised variance would sharpen the Mincer-Zarnowitz
  evaluation.
- The in-sample-versus-out-of-sample QLIKE figure compares a single in-sample
  fit against rolling forecasts, so it mixes generalisation with that procedural
  difference and rides the realised-variance level; it is reported as a
  diagnostic, with the frozen-versus-reselected and baseline-versus-frozen
  contrasts as the load-bearing overfit measures.
- At the 99% level the pooled out-of-sample breach count is small (around a
  dozen), so the conditional-coverage tests there have limited power; results
  are read at the 95% level first.
- The leverage-order study (o=2) and intraday realised variance are noted as
  next steps rather than implemented; an arbitrage-free implied-volatility
  surface is the staged longer-term direction.

## License

MIT -- see [LICENSE](LICENSE).
