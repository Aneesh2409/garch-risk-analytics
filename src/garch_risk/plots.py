"""Plotting helpers for the notebooks and reports.

Every function returns a Matplotlib ``Figure`` and never calls ``show`` or
``savefig`` itself, so callers stay in control of display and output. A single
light style is applied throughout for a consistent look.

Conventions match the rest of the package: volatility inputs are DAILY and are
annualised here for display; VaR/ES are left-tail returns (negative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ANNUALISE = np.sqrt(252)

# A restrained, consistent palette.
_C = {
    "primary": "#2b6cb0",
    "accent": "#dd6b20",
    "muted": "#718096",
    "good": "#2f855a",
    "bad": "#c53030",
    "grid": "#e2e8f0",
}


def _style(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = ""
           ) -> plt.Axes:
    """Apply the shared look to an axis."""
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, color=_C["grid"], linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=9)
    return ax


def plot_volatility(sigma: pd.Series, realised: pd.Series | None = None,
                    title: str = "GARCH volatility forecast") -> Figure:
    """Annualised GARCH volatility through time, with an optional realised
    overlay. Inputs are daily; the axis is annualised percent."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sigma.index, sigma * ANNUALISE * 100, color=_C["primary"],
            linewidth=1.4, label="GARCH forecast", zorder=3)
    if realised is not None:
        r = realised.reindex(sigma.index)
        ax.plot(r.index, r * ANNUALISE * 100, color=_C["muted"],
                linewidth=1.0, alpha=0.8, label="Realised (rolling)", zorder=2)
        ax.legend(frameon=False, fontsize=9)
    _style(ax, title, "", "Annualised volatility (%)")
    fig.tight_layout()
    return fig


def plot_var_breaches(returns: pd.Series, var: pd.Series,
                      es: pd.Series | None = None,
                      title: str = "VaR backtest") -> Figure:
    """Daily returns with the VaR threshold and breaches highlighted."""
    df = pd.concat([returns.rename("ret"), var.rename("var")],
                   axis=1, join="inner").dropna()
    breaches = df[df["ret"] < df["var"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["ret"] * 100, color=_C["muted"], linewidth=0.6,
            alpha=0.7, label="Daily return", zorder=2)
    ax.plot(df.index, df["var"] * 100, color=_C["primary"], linewidth=1.3,
            label="VaR threshold", zorder=3)
    if es is not None:
        e = es.reindex(df.index)
        ax.plot(e.index, e * 100, color=_C["accent"], linewidth=1.0,
                linestyle="--", label="Expected shortfall", zorder=3)
    ax.scatter(breaches.index, breaches["ret"] * 100, color=_C["bad"], s=14,
               zorder=4, label=f"Breaches ({len(breaches)})")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    _style(ax, title, "", "Return (%)")
    fig.tight_layout()
    return fig


def plot_mz_scatter(realised_var: pd.Series, forecast_var: pd.Series,
                    slope: float | None = None, intercept: float | None = None,
                    title: str = "Mincer-Zarnowitz") -> Figure:
    """Realised vs forecast variance, with the fitted line and the 45-degree
    line (the latter is where a perfectly calibrated forecast would sit)."""
    df = pd.concat([realised_var.rename("rv"), forecast_var.rename("fv")],
                   axis=1, join="inner").dropna()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["fv"], df["rv"], s=8, color=_C["primary"], alpha=0.35, zorder=2)
    hi = max(df["fv"].max(), df["rv"].max())
    ax.plot([0, hi], [0, hi], color=_C["muted"], linestyle=":", linewidth=1.2,
            label="45 degrees (ideal)", zorder=3)
    if slope is not None and intercept is not None:
        xs = np.array([0, df["fv"].max()])
        ax.plot(xs, intercept + slope * xs, color=_C["accent"], linewidth=1.6,
                label=f"fit: slope={slope:.2f}", zorder=4)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, title, "Forecast variance", "Realised variance (r^2)")
    fig.tight_layout()
    return fig


def plot_greeks_evolution(greeks_df: pd.DataFrame, asset: str = "Total",
                          title: str = "Portfolio Greeks through time") -> Figure:
    """Four-panel evolution of Delta/Gamma/Vega/Theta for one asset row.

    Expects the MultiIndexed (Day, Asset) frame from
    :func:`garch_risk.greeks.daily_portfolio_greeks`.
    """
    sub = greeks_df.xs(asset, level="Asset")
    greeks = ("Delta", "Gamma", "Vega", "Theta")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, g in zip(axes.ravel(), greeks):
        ax.plot(sub.index, sub[g], color=_C["primary"], linewidth=1.4, zorder=3)
        ax.axhline(0, color=_C["muted"], linewidth=0.8, zorder=1)
        _style(ax, g, "Day", "")
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_pnl_curve(curve: pd.DataFrame,
                   title: str = "P&L decomposition vs price shock") -> Figure:
    """Linear / quadratic / full-revaluation P&L across price shocks.

    Expects the frame from :func:`garch_risk.risk.pnl_curve`.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = curve.index * 100
    ax.plot(x, curve["FullReval"], color=_C["primary"], linewidth=2.2,
            label="Full revaluation (truth)", zorder=4)
    ax.plot(x, curve["Quadratic"], color=_C["accent"], linewidth=1.6,
            linestyle="--", label="Quadratic (delta-gamma)", zorder=3)
    ax.plot(x, curve["Linear"], color=_C["muted"], linewidth=1.4,
            linestyle=":", label="Linear (delta-normal)", zorder=2)
    ax.axhline(0, color=_C["muted"], linewidth=0.8)
    ax.axvline(0, color=_C["muted"], linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, title, "Price shock (%)", "P&L")
    fig.tight_layout()
    return fig


def plot_stress_grid(grid: pd.DataFrame,
                     title: str = "Stress P&L (price vs vol shock)") -> Figure:
    """Heatmap of full-revaluation P&L over the price/vol shock grid.

    Expects the frame from :func:`garch_risk.risk.stress_grid` (index = price
    shocks, columns = vol shocks).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.abs(grid.to_numpy()).max()
    im = ax.imshow(grid.to_numpy(), cmap="RdYlGn", origin="lower",
                   aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"{c:+.0%}" for c in grid.columns], fontsize=8)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([f"{r:+.0%}" for r in grid.index], fontsize=8)
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            ax.text(j, i, f"{grid.iloc[i, j]:,.0f}", ha="center", va="center",
                    fontsize=7, color="#1a202c")
    fig.colorbar(im, ax=ax, label="P&L", fraction=0.046, pad=0.04)
    _style(ax, title, "Volatility shock", "Price shock")
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_risk_dashboard(sigma: pd.Series, returns: pd.Series, var: pd.Series,
                        curve: pd.DataFrame, grid: pd.DataFrame,
                        asset: str = "") -> Figure:
    """A 2x2 dashboard: volatility, VaR breaches, P&L curve, stress grid."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Volatility
    ax = axes[0, 0]
    ax.plot(sigma.index, sigma * ANNUALISE * 100, color=_C["primary"], lw=1.3)
    _style(ax, "Annualised volatility", "", "%")

    # VaR breaches
    ax = axes[0, 1]
    df = pd.concat([returns.rename("ret"), var.rename("var")],
                   axis=1, join="inner").dropna()
    br = df[df["ret"] < df["var"]]
    ax.plot(df.index, df["ret"] * 100, color=_C["muted"], lw=0.5, alpha=0.7)
    ax.plot(df.index, df["var"] * 100, color=_C["primary"], lw=1.2)
    ax.scatter(br.index, br["ret"] * 100, color=_C["bad"], s=10, zorder=4)
    _style(ax, f"VaR backtest ({len(br)} breaches)", "", "Return (%)")

    # PnL curve
    ax = axes[1, 0]
    x = curve.index * 100
    ax.plot(x, curve["FullReval"], color=_C["primary"], lw=2)
    ax.plot(x, curve["Quadratic"], color=_C["accent"], lw=1.4, ls="--")
    ax.plot(x, curve["Linear"], color=_C["muted"], lw=1.2, ls=":")
    ax.axhline(0, color=_C["muted"], lw=0.8)
    ax.axvline(0, color=_C["muted"], lw=0.8)
    _style(ax, "P&L decomposition", "Price shock (%)", "P&L")

    # Stress grid
    ax = axes[1, 1]
    vmax = np.abs(grid.to_numpy()).max()
    im = ax.imshow(grid.to_numpy(), cmap="RdYlGn", origin="lower",
                   aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"{c:+.0%}" for c in grid.columns], fontsize=7)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([f"{r:+.0%}" for r in grid.index], fontsize=7)
    _style(ax, "Stress P&L grid", "Vol shock", "Price shock")
    ax.grid(False)

    suptitle = f"Risk dashboard{' - ' + asset if asset else ''}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# --- Walk-forward validation -------------------------------------------------
# These consume the result objects from :mod:`garch_risk.walkforward`. They are
# duck-typed on the public attributes of ``AssetWalkForward`` (``folds``,
# ``*_backtests``, ``per_fold_breach_rate*``, ``frozen_spec``) so this module
# stays free of any import from the walk-forward layer.

_DIST_COLOUR = {"t": _C["primary"], "skewt": _C["accent"], "normal": _C["muted"]}


def plot_walkforward_backtest(results: dict, alphas: tuple[float, ...] = (0.05, 0.01),
                              title: str = "Pooled out-of-sample breach rates"
                              ) -> Figure:
    """Pooled OOS breach rate per asset for the baseline, frozen and reselected
    runs, one panel per confidence level.

    ``results`` maps asset name -> walk-forward result. The dashed line marks the
    nominal rate; bars above it over-breach. The baseline (original symmetric-t
    spec) versus frozen (selected spec) contrast is the headline of the panel.
    """
    assets = list(results)
    fig, axes = plt.subplots(1, len(alphas), figsize=(5.2 * len(alphas), 4.4),
                             squeeze=False)
    width = 0.26
    runs = (("baseline", "baseline_backtests", _C["muted"]),
            ("frozen", "frozen_backtests", _C["primary"]),
            ("reselected", "reselected_backtests", _C["good"]))
    x = np.arange(len(assets))
    for ax, alpha in zip(axes.ravel(), alphas):
        for k, (label, attr, colour) in enumerate(runs):
            rates = [getattr(results[a], attr)[alpha].observed_rate * 100
                     for a in assets]
            ax.bar(x + (k - 1) * width, rates, width, label=label,
                   color=colour, zorder=3)
        ax.axhline(alpha * 100, color=_C["bad"], linestyle="--", linewidth=1.2,
                   zorder=4, label=f"nominal {alpha:.0%}")
        ax.set_xticks(x)
        ax.set_xticklabels(assets)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        _style(ax, f"alpha = {alpha:.0%}", "", "Observed breach rate (%)")
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_walkforward_regime(results: dict, alpha: float = 0.05,
                            title: str = "Per-year breach rate: selected vs baseline"
                            ) -> Figure:
    """Per-fold (calendar-year) breach rate, frozen vs baseline, one panel per
    asset.

    Surfaces *where* the selected spec helps: a year whose baseline bar towers
    over the nominal line while the frozen bar sits near it is a regime in which
    the symmetric-t model under-covered the tail.
    """
    assets = list(results)
    fig, axes = plt.subplots(1, len(assets), figsize=(4.6 * len(assets), 4.2),
                             squeeze=False, sharey=True)
    width = 0.38
    for ax, asset in zip(axes.ravel(), assets):
        r = results[asset]
        froz = r.per_fold_breach_rate[alpha]
        base = r.per_fold_breach_rate_baseline[alpha]
        years = list(froz)
        x = np.arange(len(years))
        ax.bar(x - width / 2, [base[y] * 100 for y in years], width,
               label="baseline", color=_C["muted"], zorder=3)
        ax.bar(x + width / 2, [froz[y] * 100 for y in years], width,
               label="frozen", color=_C["primary"], zorder=3)
        ax.axhline(alpha * 100, color=_C["bad"], linestyle="--", linewidth=1.2,
                   zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45, ha="right")
        ax.legend(frameon=False, fontsize=8)
        _style(ax, asset, "", "Breach rate (%)" if asset == assets[0] else "")
    fig.suptitle(f"{title} (alpha = {alpha:.0%})", fontsize=13,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_walkforward_selection(results: dict,
                               title: str = "BIC-selected specification by fold"
                               ) -> Figure:
    """Grid of the BIC-selected specification for each asset and OOS year.

    Cell colour encodes the chosen innovation distribution; the text is the full
    spec label. A row of one colour means the distribution choice was stable
    across every regime.
    """
    assets = list(results)
    years = [f.year for f in results[assets[0]].folds]
    dists = sorted({f.selected_bic.dist
                    for a in assets for f in results[a].folds},
                   key=lambda d: list(_DIST_COLOUR).index(d))
    dist_idx = {d: i for i, d in enumerate(dists)}
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([_DIST_COLOUR[d] for d in dists])

    mat = np.zeros((len(assets), len(years)))
    labels = [[""] * len(years) for _ in assets]
    for i, a in enumerate(assets):
        for j, f in enumerate(results[a].folds):
            mat[i, j] = dist_idx[f.selected_bic.dist]
            labels[i][j] = f.selected_bic.label

    fig, ax = plt.subplots(figsize=(1.5 * len(years) + 1.5, 0.9 * len(assets) + 1.5))
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-0.5, vmax=len(dists) - 0.5)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(range(len(assets)))
    ax.set_yticklabels(assets)
    for i in range(len(assets)):
        for j in range(len(years)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=_DIST_COLOUR[d]) for d in dists]
    ax.legend(handles, dists, frameon=False, fontsize=9,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), title="distribution")
    _style(ax, title, "OOS year", "")
    ax.grid(False)
    fig.tight_layout()
    return fig
