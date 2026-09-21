from __future__ import annotations

import pandas as pd
import pytest

from nse_screener.fundamentals.wacc import (
    BETA_SENSITIVITY,
    ERP_SENSITIVITY,
    InsufficientReturnHistoryError,
    WACCInputs,
    compute_wacc,
)

_INDEX_RETURN_PATTERN = [0.01, -0.008, 0.015, -0.012, 0.005, -0.003]
_BETA_TRUE = 1.5


def _weekly_series(returns: list[float], start: float = 100.0) -> pd.Series:
    dates = pd.date_range("2022-01-02", periods=len(returns) + 1, freq="W")
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return pd.Series(prices, index=dates)


def _correlated_series(weeks: int = 120) -> tuple[pd.Series, pd.Series]:
    repeats = weeks // len(_INDEX_RETURN_PATTERN) + 1
    index_returns = (_INDEX_RETURN_PATTERN * repeats)[:weeks]
    stock_returns = [_BETA_TRUE * r for r in index_returns]
    return _weekly_series(stock_returns), _weekly_series(index_returns)


def _inputs(total_debt: float = 20000.0, prior_total_debt: float = 18000.0) -> WACCInputs:
    return WACCInputs(
        symbol="AAA",
        market_cap=100_000.0,
        total_debt=total_debt,
        prior_total_debt=prior_total_debt,
        interest_expense=1900.0,
        effective_tax_rate=0.25,
    )


def test_beta_on_exactly_correlated_synthetic_series():
    stock, index = _correlated_series()
    result = compute_wacc(_inputs(), stock, index, risk_free_rate=0.07, equity_risk_premium=0.05)

    # zero-noise construction: stock_return = 1.5 * index_return exactly,
    # so cov/var must recover beta_true exactly regardless of distribution
    assert result.raw_beta == pytest.approx(_BETA_TRUE)
    assert result.adjusted_beta == pytest.approx(0.67 * _BETA_TRUE + 0.33 * 1.0)


def test_wacc_point_matches_documented_formula():
    stock, index = _correlated_series()
    rf, erp = 0.07, 0.05
    result = compute_wacc(_inputs(), stock, index, risk_free_rate=rf, equity_risk_premium=erp)

    expected_cost_of_equity = rf + result.adjusted_beta * erp
    expected_cost_of_debt = 1900.0 / ((20000.0 + 18000.0) / 2.0)
    expected_after_tax_kd = expected_cost_of_debt * (1 - 0.25)
    e, d = 100_000.0, 20_000.0
    expected_wacc = (e / (d + e)) * expected_cost_of_equity + (d / (d + e)) * expected_after_tax_kd

    assert result.cost_of_equity == pytest.approx(expected_cost_of_equity)
    assert result.cost_of_debt == pytest.approx(expected_cost_of_debt)
    assert result.after_tax_cost_of_debt == pytest.approx(expected_after_tax_kd)
    assert result.wacc_point == pytest.approx(expected_wacc)


def test_sensitivity_band_brackets_the_point_estimate():
    stock, index = _correlated_series()
    result = compute_wacc(_inputs(), stock, index, risk_free_rate=0.07, equity_risk_premium=0.05)

    assert result.wacc_low < result.wacc_point < result.wacc_high

    # band should exactly match beta +-0.2, erp +-0.01 per the spec
    rf, erp = 0.07, 0.05
    beta = result.adjusted_beta
    e, d = 100_000.0, 20_000.0
    expected_low_coe = rf + (beta - BETA_SENSITIVITY) * (erp - ERP_SENSITIVITY)
    expected_low = (e / (d + e)) * expected_low_coe + (d / (d + e)) * result.after_tax_cost_of_debt
    assert result.wacc_low == pytest.approx(expected_low)


def test_unlevered_when_total_debt_is_zero():
    stock, index = _correlated_series()
    result = compute_wacc(
        _inputs(total_debt=0.0, prior_total_debt=0.0),
        stock,
        index,
        risk_free_rate=0.07,
        equity_risk_premium=0.05,
    )
    assert result.unlevered is True
    assert result.wacc_point == pytest.approx(result.cost_of_equity)
    assert result.cost_of_debt == pytest.approx(0.0)


def test_raises_below_minimum_weekly_observations():
    stock, index = _correlated_series(weeks=20)
    with pytest.raises(InsufficientReturnHistoryError):
        compute_wacc(_inputs(), stock, index, risk_free_rate=0.07, equity_risk_premium=0.05)


def test_raises_on_zero_index_variance():
    flat_returns = [0.0] * 100
    stock, index = _weekly_series(flat_returns), _weekly_series(flat_returns)
    with pytest.raises(ValueError, match="variance is zero"):
        compute_wacc(_inputs(), stock, index, risk_free_rate=0.07, equity_risk_premium=0.05)
