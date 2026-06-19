"""End-to-end smoke test on real market data.

Runs the full pipeline -- download, GARCH volatility, VaR/ES, backtest -- on
real Yahoo Finance data and prints a plain-text report. This is the first
check that the package behaves on real returns (with their genuine fat tails
and volatility clustering), not just synthetic test data.

Run from the repo root, in the project's virtualenv:

    python scripts/smoke_test.py

It hits the network (Yahoo Finance), so it will not run in an offline sandbox.
Output is ASCII-only.
"""

import numpy as np

from garch_risk.data import load_returns
from garch_risk.var_es import backtest_var, rolling_var_es
from garch_risk.volatility import rolling_garch_volatility


def main() -> None:
    print("=" * 64)
    print("Downloading prices from Yahoo Finance (10y, equity calendar)...")
    rets = load_returns(lookback_years=10, strip_weekends=True)
    print(f"Returns shape : {rets.shape[0]} days x {rets.shape[1]} assets")
    print(f"Date range    : {rets.index.min().date()} to {rets.index.max().date()}")

    print("\nSample annualised volatility (sanity check):")
    for col in rets.columns:
        ann = rets[col].std() * np.sqrt(252) * 100.0
        print(f"  {col:9s}: {ann:5.1f}%   "
              f"(expect ~15-20% equities, ~40-70% BTC)")

    for asset in rets.columns:
        print("\n" + "-" * 64)
        print(f"Asset: {asset}")
        print("Fitting rolling GJR-GARCH(1,1)-t (window=365, refit_every=21)...")
        sigma = rolling_garch_volatility(rets[asset], window=365, refit_every=21)
        print(f"  Forecast days : {len(sigma)}")
        print(f"  Mean ann. vol : {sigma.mean() * np.sqrt(252) * 100:.1f}%")

        ve = rolling_var_es(sigma, alpha=0.05, dist="t", dof=5)
        res = backtest_var(rets[asset], ve["VaR"], alpha=0.05)

        print(f"  95% VaR backtest over {res.n_obs} days:")
        print(f"    Breaches    : {res.n_breaches} "
              f"(rate {res.observed_rate * 100:.2f}%, expected 5.00%)")
        print(f"    Kupiec      : LR={res.kupiec_lr:6.3f}  p={res.kupiec_p:.4f}"
              f"  -> {'PASS' if res.kupiec_pass else 'FAIL'}")
        print(f"    Independence: LR={res.independence_lr:6.3f}  "
              f"p={res.independence_p:.4f}  -> "
              f"{'PASS' if res.independence_pass else 'FAIL'}")
        print(f"    Cond. cover.: LR={res.cc_lr:6.3f}  p={res.cc_p:.4f}"
              f"  -> {'PASS' if res.cc_pass else 'FAIL'}")

    print("\n" + "=" * 64)
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
