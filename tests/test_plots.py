"""Tests for the plotting module.

These confirm each function runs without error and returns a Figure on
representative inputs. Visual quality is judged from the rendered notebooks,
not here; the point of these tests is to catch interface/shape breakage.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend; no display required

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from garch_risk.config import DEFAULT_PORTFOLIO, RISK_FREE_RATE
from garch_risk.greeks import daily_portfolio_greeks
from garch_risk.plots import (
    plot_greeks_evolution,
    plot_mz_scatter,
    plot_pnl_curve,
    plot_risk_dashboard,
    plot_stress_grid,
    plot_var_breaches,
    plot_volatility,
)
from garch_risk.risk import pnl_curve, stress_grid


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    returns = pd.Series(rng.standard_normal(300) * 0.012, index=idx, name="S&P500")
    sigma = pd.Series(np.abs(rng.normal(0.012, 0.002, 300)), index=idx)
    var = -1.65 * sigma
    es = -2.06 * sigma
    return idx, returns, sigma, var, es


def test_plot_volatility(synthetic):
    _, _, sigma, _, _ = synthetic
    fig = plot_volatility(sigma, realised=sigma * 1.05)
    assert isinstance(fig, Figure)


def test_plot_var_breaches(synthetic):
    _, returns, _, var, es = synthetic
    fig = plot_var_breaches(returns, var, es)
    assert isinstance(fig, Figure)


def test_plot_mz_scatter(synthetic):
    _, returns, sigma, _, _ = synthetic
    fig = plot_mz_scatter(returns ** 2, sigma ** 2, slope=0.65, intercept=1e-5)
    assert isinstance(fig, Figure)


def test_plot_pnl_curve():
    spots = {"S&P500": 4000.0, "NASDAQ": 14000.0, "BTC-USD": 60000.0}
    sigmas = {a: 0.012 for a in spots}
    curve = pnl_curve(DEFAULT_PORTFOLIO, spots, sigmas, RISK_FREE_RATE,
                      shocks=np.linspace(-0.2, 0.2, 11))
    fig = plot_pnl_curve(curve)
    assert isinstance(fig, Figure)


def test_plot_stress_grid():
    spots = {"S&P500": 4000.0, "NASDAQ": 14000.0, "BTC-USD": 60000.0}
    sigmas = {a: 0.012 for a in spots}
    grid = stress_grid(DEFAULT_PORTFOLIO, spots, sigmas, RISK_FREE_RATE,
                       price_shocks=np.linspace(-0.1, 0.1, 5),
                       vol_shocks=np.linspace(-0.3, 0.6, 4))
    fig = plot_stress_grid(grid)
    assert isinstance(fig, Figure)


def test_plot_greeks_evolution():
    n = 40
    spot_paths = {a: np.full(n, s) for a, s in
                  {"S&P500": 4000.0, "NASDAQ": 14000.0, "BTC-USD": 60000.0}.items()}
    sigma_paths = {a: np.full(n, 0.012) for a in spot_paths}
    gdf = daily_portfolio_greeks(DEFAULT_PORTFOLIO, spot_paths, sigma_paths,
                                 RISK_FREE_RATE)
    fig = plot_greeks_evolution(gdf, asset="Total")
    assert isinstance(fig, Figure)


def test_plot_risk_dashboard(synthetic):
    _, returns, sigma, var, _ = synthetic
    spots = {"S&P500": 4000.0, "NASDAQ": 14000.0, "BTC-USD": 60000.0}
    sigmas = {a: 0.012 for a in spots}
    curve = pnl_curve(DEFAULT_PORTFOLIO, spots, sigmas, RISK_FREE_RATE,
                      shocks=np.linspace(-0.2, 0.2, 11))
    grid = stress_grid(DEFAULT_PORTFOLIO, spots, sigmas, RISK_FREE_RATE,
                       price_shocks=np.linspace(-0.1, 0.1, 5),
                       vol_shocks=np.linspace(-0.3, 0.6, 4))
    fig = plot_risk_dashboard(sigma, returns, var, curve, grid, asset="S&P500")
    assert isinstance(fig, Figure)
