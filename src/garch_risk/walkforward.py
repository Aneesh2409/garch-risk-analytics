"""Walk-forward (rolling-origin) specification selection and out-of-sample
evaluation -- the overfit defence for the volatility / VaR pipeline.

WHAT THIS ANSWERS
-----------------
The base pipeline fixes the model spec (GJR-GARCH(1,1)-t) and the estimation
window by hand. The motivating question is whether those choices fit the
*sample* rather than the *process*. Walk-forward addresses it by never letting a
forecast see a day it is later scored on, and by selecting the spec only on past
data.

DESIGN (all decisions locked upstream)
--------------------------------------
* Sliding training window of ``WF_TRAIN_WINDOW`` (1260) trading days. Both ends
  move; no expanding anchor, so every fold's estimates are comparable and the
  window cannot quietly memorise the whole history.
* OOS folds are CALENDAR YEARS. "Annual reselection": the spec is chosen once
  per fold, on the 1260 days ENDING THE DAY BEFORE the fold's first OOS day,
  then held fixed across that fold's days. Folds are anchored to dates, so a
  re-pull that shifts the start by a few rows leaves fold identity intact.
* Within a fold the chosen spec's PARAMETERS refit every ``WF_REFIT_EVERY`` (5)
  days; the conditional-variance recursion still rolls daily between refits.
  (This is exactly :func:`volatility._rolling_garch`'s contract, reused as-is.)
* Candidate grid: distribution {t, skewt} x ARCH order p {1, 2}, with o=1, q=1
  held fixed. Two orthogonal axes -- one for tail shape, one for the marginal
  value of an extra ARCH lag, which the criterion is expected to penalise. The
  o=2 leverage study is deliberately kept separate.
* Selection by BIC (primary); AIC is recorded alongside as a sensitivity. A
  candidate that fails to fit or does not converge is excluded from selection
  rather than allowed to win on an unreliable likelihood.

TWO OVERFIT FIGURES (they measure DIFFERENT objects -- label them precisely)
---------------------------------------------------------------------------
1. In-sample vs OOS QLIKE for the FROZEN spec -- does the *model* generalise.
2. Frozen vs reselected OOS gap -- does the *selection procedure* earn its
   keep. If reselection does WORSE out of sample, the selection is itself
   overfitting; that is a finding, not a bug.

The "frozen" spec is chosen causally, on the first training window only -- never
full-sample-optimal, or the comparison is rigged.

NO LOOK-AHEAD
-------------
Two independent guards: (a) spec selection asserts its training window ends
strictly before the fold it feeds; (b) the rolling forecast inherits
:func:`volatility._rolling_garch`'s ``train = returns[t-window:t]`` contract.
Pooled backtests are per asset across time -- never pooled across assets, which
would blend regimes and destroy the dof split.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

from .var_es import backtest_var, rolling_var_es, BacktestResult
from .volatility import _FIT_SCALE, _rolling_garch
from .volatility_eval import qlike

# --- Walk-forward constants ---------------------------------------------------
WF_TRAIN_WINDOW: int = 1260      # sliding training window (trading days, ~5y)
WF_REFIT_EVERY: int = 5          # parameter refit cadence within a fold
WF_ALPHAS: tuple[float, ...] = (0.05, 0.01)


@dataclass(frozen=True)
class Spec:
    """A GARCH specification: distribution plus (p, o, q) orders."""
    dist: str
    p: int = 1
    o: int = 1
    q: int = 1

    @property
    def label(self) -> str:
        return f"{self.dist}({self.p},{self.o},{self.q})"


# The 2x2 grid. o=1, q=1 fixed; vary distribution and ARCH order p only.
CANDIDATE_GRID: tuple[Spec, ...] = (
    Spec("t", 1, 1, 1),
    Spec("t", 2, 1, 1),
    Spec("skewt", 1, 1, 1),
    Spec("skewt", 2, 1, 1),
)

# The "did selection actually change anything?" control: the original repo
# specification, forced across every fold. Backtesting this alongside the
# selected spec attributes any improvement to the distribution choice, with the
# walk-forward window and fold structure held identical. For an asset whose
# frozen spec already IS the baseline (e.g. BTC -> t), the two runs coincide
# exactly -- which is itself the finding that the asset never needed skew.
BASELINE_SPEC: Spec = Spec("t", 1, 1, 1)


# --- Result containers --------------------------------------------------------

@dataclass(frozen=True)
class SpecFit:
    """One candidate's fit on a training window: information criteria + status."""
    spec: Spec
    bic: float
    aic: float
    converged: bool


@dataclass(frozen=True)
class FoldSelection:
    """Per-fold record: which spec each criterion picked, with the full table."""
    year: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_oos: int
    fits: tuple[SpecFit, ...]
    selected_bic: Spec
    selected_aic: Spec


@dataclass(frozen=True)
class AssetWalkForward:
    """Everything the walk-forward produced for a single asset."""
    asset: str
    folds: tuple[FoldSelection, ...]
    frozen_spec: Spec
    baseline_spec: Spec
    # pooled OOS backtests, keyed by alpha
    baseline_backtests: dict[float, BacktestResult]
    frozen_backtests: dict[float, BacktestResult]
    reselected_backtests: dict[float, BacktestResult]
    # overfit figure 1: frozen spec, in-sample vs pooled-OOS QLIKE
    qlike_in_sample: float
    qlike_oos_frozen: float
    # overfit figure 2: pooled-OOS QLIKE, frozen vs reselected
    qlike_oos_reselected: float
    # attribution: pooled-OOS QLIKE of the forced baseline spec
    qlike_oos_baseline: float
    # per-fold OOS breach rate, keyed by (alpha -> {year: rate})
    per_fold_breach_rate: dict[float, dict[int, float]]
    per_fold_breach_rate_baseline: dict[float, dict[int, float]]

    @property
    def qlike_generalisation_gap(self) -> float:
        """OOS minus in-sample QLIKE for the frozen spec (positive = worse OOS)."""
        return self.qlike_oos_frozen - self.qlike_in_sample

    @property
    def qlike_selection_gap(self) -> float:
        """Frozen minus reselected OOS QLIKE (positive = reselection helped)."""
        return self.qlike_oos_frozen - self.qlike_oos_reselected

    @property
    def qlike_distribution_gap(self) -> float:
        """Baseline minus frozen OOS QLIKE.

        Positive means the selected distribution improves the *variance*
        forecast, not just the tail. Near zero means skew-t fixes the tail
        quantile (VaR) while leaving sigma -- and therefore QLIKE and the
        Greeks -- essentially unchanged.
        """
        return self.qlike_oos_baseline - self.qlike_oos_frozen

    @property
    def baseline_equals_frozen(self) -> bool:
        """True when the forced baseline spec is the frozen spec (no contrast)."""
        return self.baseline_spec == self.frozen_spec


# --- Spec selection -----------------------------------------------------------

def _fit_spec(train_scaled: np.ndarray, spec: Spec) -> SpecFit:
    """Fit one candidate on a (scaled) training window; report BIC/AIC/status.

    A fit that raises or fails to converge comes back with infinite criteria and
    ``converged=False`` so selection can exclude it.
    """
    model = arch_model(train_scaled, vol="Garch", p=spec.p, o=spec.o, q=spec.q,
                       mean="Zero", dist=spec.dist)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp="off")
        converged = int(getattr(res, "convergence_flag", 0)) == 0
        bic, aic = float(res.bic), float(res.aic)
        if not np.isfinite(bic):
            converged = False
        return SpecFit(spec, bic, aic, converged)
    except Exception:
        return SpecFit(spec, float("inf"), float("inf"), False)


def _select_spec(train_returns: pd.Series, fold_start: pd.Timestamp,
                 grid: tuple[Spec, ...]) -> tuple[tuple[SpecFit, ...], Spec, Spec]:
    """Select the BIC- and AIC-best spec on a training window.

    Asserts the training window ends strictly before ``fold_start`` -- the #1
    place walk-forward silently leaks, so it is a hard check, not a comment.
    """
    if train_returns.index[-1] >= fold_start:
        raise AssertionError(
            "look-ahead: training window overlaps or reaches the OOS fold "
            f"(train ends {train_returns.index[-1]}, fold starts {fold_start})")

    scaled = train_returns.to_numpy(dtype=float) * _FIT_SCALE
    fits = tuple(_fit_spec(scaled, s) for s in grid)
    usable = [f for f in fits if f.converged and np.isfinite(f.bic)]
    pool = usable if usable else list(fits)
    sel_bic = min(pool, key=lambda f: f.bic).spec
    sel_aic = min(pool, key=lambda f: f.aic).spec
    return fits, sel_bic, sel_aic


# --- Folds and rolling forecasts ----------------------------------------------

def _fold_boundaries(oos_index: pd.DatetimeIndex
                     ) -> list[tuple[int, pd.DatetimeIndex]]:
    """Group the OOS index into calendar-year folds, in chronological order."""
    years = oos_index.year
    return [(int(y), oos_index[years == y]) for y in sorted(pd.unique(years))]


def _cached_rolling(returns: pd.Series, spec: Spec, window: int,
                    refit_every: int, cache: dict[Spec, pd.DataFrame]
                    ) -> pd.DataFrame:
    """Run the rolling fit for ``spec`` once over the full series and cache it.

    Running over the full series (not per-fold) keeps the refit schedule's phase
    identical across frozen and reselected runs and makes the forecasts
    reproducible; each fold simply slices the days it needs.
    """
    if spec not in cache:
        cache[spec] = _rolling_garch(returns, window, refit_every, spec.dist,
                                     spec.p, spec.o, spec.q)
    return cache[spec]


def _var_series(forecasts: pd.DataFrame, spec: Spec, alpha: float) -> pd.Series:
    """One-day VaR series for a slice of rolling forecasts under ``spec``."""
    sigma = forecasts["sigma"]
    if spec.dist == "skewt":
        out = rolling_var_es(sigma, alpha=alpha, dist="skewt",
                             dof=forecasts["nu"], skew=forecasts["skew"])
    elif spec.dist == "t":
        out = rolling_var_es(sigma, alpha=alpha, dist="t", dof=forecasts["nu"])
    else:
        out = rolling_var_es(sigma, alpha=alpha, dist="normal")
    return out["VaR"]


def _insample_qlike(train_returns: pd.Series, spec: Spec) -> float:
    """QLIKE of ``spec`` fit on, and scored on, the same training window."""
    scaled = train_returns.to_numpy(dtype=float) * _FIT_SCALE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(scaled, vol="Garch", p=spec.p, o=spec.o, q=spec.q,
                         mean="Zero", dist=spec.dist).fit(disp="off")
    cond_sigma = np.asarray(res.conditional_volatility) / _FIT_SCALE
    forecast_var = pd.Series(cond_sigma ** 2, index=train_returns.index)
    realised_var = train_returns ** 2
    return qlike(realised_var, forecast_var)


# --- Orchestration ------------------------------------------------------------

def run_asset_walkforward(returns: pd.Series,
                          window: int = WF_TRAIN_WINDOW,
                          refit_every: int = WF_REFIT_EVERY,
                          grid: tuple[Spec, ...] = CANDIDATE_GRID,
                          alphas: tuple[float, ...] = WF_ALPHAS,
                          baseline_spec: Spec = BASELINE_SPEC
                          ) -> AssetWalkForward:
    """Run the full walk-forward for one asset's daily return series.

    Returns an :class:`AssetWalkForward` with per-fold selections; pooled
    baseline / frozen / reselected OOS backtests at each ``alpha``; both overfit
    figures; and the distribution-attribution QLIKE. ``baseline_spec`` is forced
    across every fold as the "original model" control.
    """
    if len(returns) <= window:
        raise ValueError(f"need more than {window} observations, got {len(returns)}")

    oos_index = returns.index[window:]
    folds = _fold_boundaries(oos_index)
    cache: dict[Spec, pd.DataFrame] = {}

    def _train_window_for(fold_first: pd.Timestamp) -> pd.Series:
        pos = returns.index.get_loc(fold_first)
        return returns.iloc[pos - window:pos]

    # Frozen spec: selected on the FIRST fold's training window, causally.
    first_fold_first = folds[0][1][0]
    train0 = _train_window_for(first_fold_first)
    _, frozen_spec, _ = _select_spec(train0, first_fold_first, grid)

    # Per-fold reselection + assembly of the reselected OOS forecast pieces.
    fold_selections: list[FoldSelection] = []
    reselected_pieces: list[tuple[Spec, pd.DataFrame]] = []
    for year, fidx in folds:
        train = _train_window_for(fidx[0])
        fits, sel_bic, sel_aic = _select_spec(train, fidx[0], grid)
        fold_selections.append(FoldSelection(
            year=year, train_start=train.index[0], train_end=train.index[-1],
            n_oos=len(fidx), fits=fits,
            selected_bic=sel_bic, selected_aic=sel_aic))
        run = _cached_rolling(returns, sel_bic, window, refit_every, cache)
        reselected_pieces.append((sel_bic, run.loc[fidx]))

    # Frozen OOS forecasts: one run over the full series (cached), sliced per
    # fold below. Computing the frozen VaR fold-by-fold -- the same call
    # structure as the reselected path -- means any fold whose reselected spec
    # equals the frozen spec contributes an IDENTICAL VaR slice to both series.
    # The frozen-vs-reselected gap then reflects real spec differences only, not
    # Monte-Carlo noise from differing call structures.
    frozen_run = _cached_rolling(returns, frozen_spec, window, refit_every, cache)
    frozen_oos = frozen_run.loc[oos_index]

    # Forced baseline run (original spec across every fold), same call structure.
    baseline_run = _cached_rolling(returns, baseline_spec, window, refit_every,
                                   cache)
    baseline_oos = baseline_run.loc[oos_index]

    # Pooled per-asset backtests at each alpha.
    baseline_bt: dict[float, BacktestResult] = {}
    frozen_bt: dict[float, BacktestResult] = {}
    reselected_bt: dict[float, BacktestResult] = {}
    per_fold_breach_rate: dict[float, dict[int, float]] = {}
    per_fold_breach_rate_baseline: dict[float, dict[int, float]] = {}
    for alpha in alphas:
        base_pieces: list[pd.Series] = []
        frozen_pieces: list[pd.Series] = []
        resel_pieces: list[pd.Series] = []
        rates: dict[int, float] = {}
        rates_base: dict[int, float] = {}
        for (year, fidx), (sp, sl) in zip(folds, reselected_pieces):
            fpiece = _var_series(frozen_run.loc[fidx], frozen_spec, alpha)
            bpiece = _var_series(baseline_run.loc[fidx], baseline_spec, alpha)
            frozen_pieces.append(fpiece)
            base_pieces.append(bpiece)
            resel_pieces.append(_var_series(sl, sp, alpha))
            rr = returns.loc[fpiece.index]
            rates[year] = float((rr < fpiece).mean())
            rates_base[year] = float((rr < bpiece).mean())

        bvar = pd.concat(base_pieces).sort_index()
        fvar = pd.concat(frozen_pieces).sort_index()
        rvar = pd.concat(resel_pieces).sort_index()
        baseline_bt[alpha] = backtest_var(returns.loc[bvar.index], bvar,
                                          alpha=alpha)
        frozen_bt[alpha] = backtest_var(returns.loc[fvar.index], fvar, alpha=alpha)
        reselected_bt[alpha] = backtest_var(returns.loc[rvar.index], rvar,
                                            alpha=alpha)
        per_fold_breach_rate[alpha] = rates
        per_fold_breach_rate_baseline[alpha] = rates_base

    # Overfit figure 1: in-sample vs pooled-OOS QLIKE for the frozen spec.
    qlike_is = _insample_qlike(train0, frozen_spec)
    qlike_oos_frozen = qlike(returns.loc[frozen_oos.index] ** 2,
                             frozen_oos["sigma"] ** 2)

    # Overfit figure 2: pooled-OOS QLIKE, reselected.
    resel_sigma = pd.concat([sl["sigma"]
                             for _, sl in reselected_pieces]).sort_index()
    qlike_oos_reselected = qlike(returns.loc[resel_sigma.index] ** 2,
                                 resel_sigma ** 2)

    # Attribution: pooled-OOS QLIKE of the forced baseline spec.
    qlike_oos_baseline = qlike(returns.loc[baseline_oos.index] ** 2,
                               baseline_oos["sigma"] ** 2)

    return AssetWalkForward(
        asset=str(returns.name),
        folds=tuple(fold_selections),
        frozen_spec=frozen_spec,
        baseline_spec=baseline_spec,
        baseline_backtests=baseline_bt,
        frozen_backtests=frozen_bt,
        reselected_backtests=reselected_bt,
        qlike_in_sample=qlike_is,
        qlike_oos_frozen=qlike_oos_frozen,
        qlike_oos_reselected=qlike_oos_reselected,
        qlike_oos_baseline=qlike_oos_baseline,
        per_fold_breach_rate=per_fold_breach_rate,
        per_fold_breach_rate_baseline=per_fold_breach_rate_baseline,
    )


def run_walkforward(returns: pd.DataFrame,
                    window: int = WF_TRAIN_WINDOW,
                    refit_every: int = WF_REFIT_EVERY,
                    grid: tuple[Spec, ...] = CANDIDATE_GRID,
                    alphas: tuple[float, ...] = WF_ALPHAS,
                    baseline_spec: Spec = BASELINE_SPEC
                    ) -> dict[str, AssetWalkForward]:
    """Run :func:`run_asset_walkforward` for each column of ``returns``."""
    return {
        col: run_asset_walkforward(returns[col], window, refit_every, grid,
                                   alphas, baseline_spec)
        for col in returns.columns
    }


# --- ASCII reporting (Windows-console safe) -----------------------------------

def summarise(result: AssetWalkForward) -> str:
    """A compact, ASCII-only text summary of one asset's walk-forward."""
    L: list[str] = []
    L.append(f"=== Walk-forward: {result.asset} ===")
    L.append(f"Frozen spec (first-window, causal): {result.frozen_spec.label}"
             f"  |  baseline (forced): {result.baseline_spec.label}"
             + ("  [baseline == frozen]" if result.baseline_equals_frozen else ""))
    L.append("")
    L.append("Per-fold selection (training window -> OOS year):")
    L.append(f"  {'year':>4}  {'n_oos':>5}  {'BIC pick':>14}  {'AIC pick':>14}")
    for f in result.folds:
        L.append(f"  {f.year:>4}  {f.n_oos:>5}  {f.selected_bic.label:>14}  "
                 f"{f.selected_aic.label:>14}")
    L.append("")
    for alpha in sorted(result.frozen_backtests):
        bb = result.baseline_backtests[alpha]
        fb = result.frozen_backtests[alpha]
        rb = result.reselected_backtests[alpha]
        L.append(f"Pooled OOS backtest @ alpha={alpha:.2f} "
                 f"(expected breach rate {alpha:.2%}):")
        L.append(f"  {'run':>10}  {'breaches':>8}  {'rate':>7}  "
                 f"{'Kupiec p':>9}  {'Indep p':>8}  {'CC p':>6}")
        for name, b in (("baseline", bb), ("frozen", fb), ("reselected", rb)):
            L.append(f"  {name:>10}  {b.n_breaches:>8}  {b.observed_rate:>7.2%}  "
                     f"{b.kupiec_p:>9.3f}  {b.independence_p:>8.3f}  "
                     f"{b.cc_p:>6.3f}")
        L.append("")
    L.append("Overfit / attribution figures (QLIKE; lower is better):")
    L.append(f"  [1] frozen in-sample = {result.qlike_in_sample:.5f}  "
             f"OOS = {result.qlike_oos_frozen:.5f}  "
             f"gap = {result.qlike_generalisation_gap:+.5f}")
    L.append(f"  [2] frozen OOS = {result.qlike_oos_frozen:.5f}  "
             f"reselected OOS = {result.qlike_oos_reselected:.5f}  "
             f"gap = {result.qlike_selection_gap:+.5f} "
             f"({'reselection helped' if result.qlike_selection_gap > 0 else 'no gain / hurt'})")
    L.append(f"  [3] baseline OOS = {result.qlike_oos_baseline:.5f}  "
             f"frozen OOS = {result.qlike_oos_frozen:.5f}  "
             f"dist gap = {result.qlike_distribution_gap:+.5f} "
             f"({'spec improves variance' if abs(result.qlike_distribution_gap) > 1e-4 else 'tail-only fix; sigma ~ unchanged'})")
    L.append("")
    L.append("Per-fold breach rate (regime breakdown; frozen vs baseline):")
    for alpha in sorted(result.per_fold_breach_rate):
        froz = result.per_fold_breach_rate[alpha]
        base = result.per_fold_breach_rate_baseline[alpha]
        froz_cells = "  ".join(f"{y}:{r:.2%}" for y, r in froz.items())
        base_cells = "  ".join(f"{y}:{r:.2%}" for y, r in base.items())
        L.append(f"  alpha={alpha:.2f}  frozen    {froz_cells}")
        L.append(f"  alpha={alpha:.2f}  baseline  {base_cells}")
    return "\n".join(L)
