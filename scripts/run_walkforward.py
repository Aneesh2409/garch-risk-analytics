"""End-to-end walk-forward run on live data: rolling-origin specification
selection and pooled out-of-sample backtests for every asset.

Usage (from the repo root, package installed via ``pip install -e .``):

    python scripts/run_walkforward.py            # full run, refit every 5 days
    python scripts/run_walkforward.py 21         # coarser run, refit every 21

The optional argument sets the parameter-refit cadence in trading days (default
5). A larger value fits the GARCH parameters less often and so runs faster, at a
small cost in responsiveness; the pooled breach rates are insensitive to it
(they move by under half a percentage point between 5 and 21).

Each asset's summary is printed and written to a timestamped text file in the
current directory. ASCII-only output (Windows-console safe).
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime

from garch_risk.data import load_returns
from garch_risk.walkforward import (
    WF_REFIT_EVERY,
    WF_TRAIN_WINDOW,
    run_asset_walkforward,
    summarise,
)


def run_report(returns, refit_every: int = WF_REFIT_EVERY,
               window: int = WF_TRAIN_WINDOW) -> str:
    """Run the walk-forward for every column and return one combined report.

    Each asset is wrapped so a failure on one (e.g. a mid-roll convergence
    error) is logged and the rest still run, rather than aborting everything.
    """
    blocks: list[str] = []
    for asset in returns.columns:
        n_obs = int(returns[asset].notna().sum())
        print(f"[running] {asset:<10} ({n_obs} obs) ... ", end="", flush=True)
        try:
            res = run_asset_walkforward(returns[asset], window=window,
                                        refit_every=refit_every)
            blocks.append(summarise(res))
            print("done")
        except Exception:
            blocks.append(f"=== Walk-forward: {asset} ===\n"
                          f"FAILED:\n{traceback.format_exc()}")
            print("FAILED (logged, continuing)")
    return "\n\n".join(blocks)


def main(refit_every: int) -> None:
    print("=" * 64)
    print("GARCH walk-forward: rolling-origin spec selection + OOS backtests")
    print("=" * 64)
    print("Loading returns from Yahoo Finance (the only network step) ...")
    returns = load_returns()
    print(f"Loaded {len(returns)} aligned trading days: "
          f"{returns.index[0].date()} -> {returns.index[-1].date()}")
    print(f"Assets: {', '.join(returns.columns)}")
    print(f"Training window: {WF_TRAIN_WINDOW} days | "
          f"refit cadence: {refit_every} days")
    print("(GARCH fits across the full history -- expect minutes, not seconds.)")
    print()

    report = run_report(returns, refit_every=refit_every)

    print()
    print(report)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = f"walkforward_results_{stamp}.txt"
    with open(outpath, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print()
    print(f"[saved] {outpath}")


if __name__ == "__main__":
    refit = int(sys.argv[1]) if len(sys.argv) > 1 else WF_REFIT_EVERY
    main(refit)
