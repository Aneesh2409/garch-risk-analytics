"""End-to-end portfolio demonstration on real market data.

Wires the whole pipeline together for the default six-option book:

    download prices -> rolling GJR-GARCH volatility -> portfolio Greeks at a
    snapshot -> price/vol shock P&L (linear vs quadratic vs full revaluation)
    -> a joint price/vol stress grid -> a short Greeks-through-time path.

This is the integration check: every module composing in one flow, on real
data, exactly as a user (or a demo notebook) would call them. Run from the
repo root in the project's virtualenv:

    python scripts/portfolio_demo.py

Hits the network (Yahoo Finance); ASCII-only output.
"""

import numpy as np

from garch_risk.config import DEFAULT_PORTFOLIO, RISK_FREE_RATE
from garch_risk.data import align_calendar, download_prices, log_returns
from garch_risk.greeks import daily_portfolio_greeks, portfolio_greeks_snapshot
from garch_risk.risk import full_revaluation_pnl, pnl_curve, portfolio_value
from garch_risk.volatility import rolling_garch_volatility

WINDOW, REFIT = 365, 21


def main() -> None:
    print("=" * 64)
    print("Loading prices and fitting GARCH volatility for each asset...")
    prices = align_calendar(download_prices(lookback_years=10), strip_weekends=True)
    rets = log_returns(prices)

    assets = list(prices.columns)
    sigma = {a: rolling_garch_volatility(rets[a], window=WINDOW, refit_every=REFIT)
             for a in assets}

    # Snapshot = the latest date for which every asset has a GARCH forecast.
    snap_date = min(sigma[a].index.max() for a in assets)
    base_spots = {a: float(prices[a].loc[snap_date]) for a in assets}
    base_sigmas = {a: float(sigma[a].loc[snap_date]) for a in assets}

    print(f"\nSnapshot date : {snap_date.date()}")
    print("Spots / GARCH daily vol (annualised):")
    for a in assets:
        print(f"  {a:9s}: spot={base_spots[a]:>12,.2f}   "
              f"vol={base_sigmas[a] * np.sqrt(252) * 100:5.1f}%")

    # --- Portfolio Greeks at the snapshot ------------------------------------
    print("\n" + "-" * 64)
    print("PORTFOLIO GREEKS (quantity-weighted, fixed strikes at snapshot spot):")
    snap = portfolio_greeks_snapshot(DEFAULT_PORTFOLIO, base_spots,
                                     base_sigmas, RISK_FREE_RATE)
    print(f"  {'Asset':10s}{'Delta':>12}{'Gamma':>12}{'Vega':>12}{'Theta':>12}")
    for asset, row in snap.iterrows():
        print(f"  {asset:10s}{row['Delta']:>12.3f}{row['Gamma']:>12.5f}"
              f"{row['Vega']:>12.3f}{row['Theta']:>12.3f}")

    strikes = {p.id: p.moneyness * base_spots[p.underlying] for p in DEFAULT_PORTFOLIO}
    base_value = portfolio_value(DEFAULT_PORTFOLIO, base_spots, base_sigmas,
                                 RISK_FREE_RATE, strikes)
    print(f"\n  Portfolio value at snapshot: {base_value:,.2f}")

    # --- Price-shock P&L decomposition ---------------------------------------
    print("\n" + "-" * 64)
    print("PRICE-SHOCK P&L (common % shock to all underlyings, vol held):")
    curve = pnl_curve(DEFAULT_PORTFOLIO, base_spots, base_sigmas, RISK_FREE_RATE,
                      shocks=np.array([-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]),
                      strikes=strikes)
    print(f"  {'shock':>7}{'Linear':>14}{'Quadratic':>14}{'FullReval':>14}")
    for s, row in curve.iterrows():
        print(f"  {s:>6.0%}{row['Linear']:>14,.2f}"
              f"{row['Quadratic']:>14,.2f}{row['FullReval']:>14,.2f}")

    # --- Joint price/vol stress scenarios ------------------------------------
    print("\n" + "-" * 64)
    print("JOINT STRESS SCENARIOS (full revaluation P&L):")
    scenarios = [(-0.10, 0.50), (-0.20, 1.00), (0.10, -0.25), (0.05, 0.30)]
    for p_shock, v_shock in scenarios:
        pnl = full_revaluation_pnl(DEFAULT_PORTFOLIO, base_spots, base_sigmas,
                                   RISK_FREE_RATE, p_shock, v_shock, strikes)
        print(f"  price {p_shock:>+5.0%}, vol {v_shock:>+5.0%}  ->  P&L {pnl:>12,.2f}")

    # --- Greeks through time -------------------------------------------------
    print("\n" + "-" * 64)
    print("GREEKS THROUGH TIME (last 60 trading days, strikes fixed at day 0):")
    window_days = 60
    spot_paths = {a: prices[a].loc[sigma[a].index].to_numpy()[-window_days:]
                  for a in assets}
    sigma_paths = {a: sigma[a].to_numpy()[-window_days:] for a in assets}
    evolution = daily_portfolio_greeks(DEFAULT_PORTFOLIO, spot_paths, sigma_paths,
                                 RISK_FREE_RATE)
    first = evolution.loc[(0, "Total")]
    last = evolution.loc[(window_days - 1, "Total")]
    print(f"  {'':10s}{'Delta':>12}{'Gamma':>12}{'Vega':>12}{'Theta':>12}")
    print(f"  {'day 0':10s}{first['Delta']:>12.3f}{first['Gamma']:>12.5f}"
          f"{first['Vega']:>12.3f}{first['Theta']:>12.3f}")
    print(f"  {'day 59':10s}{last['Delta']:>12.3f}{last['Gamma']:>12.5f}"
          f"{last['Vega']:>12.3f}{last['Theta']:>12.3f}")

    print("\n" + "=" * 64)
    print("Portfolio demo complete -- full pipeline composed end to end.")


if __name__ == "__main__":
    main()
