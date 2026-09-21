from __future__ import annotations

from datetime import date

import pytest

from nse_screener.fundamentals.reorganize import ReorganizedStatements
from nse_screener.fundamentals.roic import DecompositionError, roic_analysis
from nse_screener.fundamentals.wacc import WACCResult


def _reorganized(**overrides: float) -> ReorganizedStatements:
    base = dict(
        symbol="AAA",
        period_end=date(2024, 3, 31),
        revenue=1000.0,
        ebita=210.0,
        effective_tax_rate=0.20,
        tax_rate_clamped=False,
        operating_tax=42.0,
        nopat=168.0,
        operating_working_capital=140.0,
        excess_cash=30.0,
        invested_capital=550.0,
        average_invested_capital=533.0,
        increase_in_invested_capital=34.0,
        free_cash_flow=134.0,
    )
    base.update(overrides)
    return ReorganizedStatements(**base)


def _wacc(**overrides: float) -> WACCResult:
    base = dict(
        symbol="AAA",
        raw_beta=1.2,
        adjusted_beta=1.134,
        cost_of_equity=0.10,
        cost_of_debt=0.08,
        after_tax_cost_of_debt=0.06,
        wacc_low=0.08,
        wacc_point=0.10,
        wacc_high=0.12,
        unlevered=False,
        weekly_observations=104,
        risk_free_rate=0.07,
        equity_risk_premium=0.05,
    )
    base.update(overrides)
    return WACCResult(**base)


def test_roic_decomposition_identity_and_value_creation():
    r = _reorganized()  # roic = 168/533 = 0.31519..., algebraically == margin*turnover*(1-tax)
    result = roic_analysis(r, _wacc())

    expected_roic = 168.0 / 533.0
    assert result.roic == pytest.approx(expected_roic)
    assert result.operating_margin == pytest.approx(210.0 / 1000.0)
    assert result.capital_turnover == pytest.approx(1000.0 / 533.0)
    assert result.economic_spread == pytest.approx(expected_roic - 0.10)
    assert result.economic_profit == pytest.approx((expected_roic - 0.10) * 533.0)
    assert result.value_creating is True


def test_spread_sign_uncertain_when_band_straddles_zero():
    # nopat = ebita * (1 - tax) = 66.625 * 0.8 = 53.3; roic = 53.3/533 = 0.10 exactly,
    # sitting exactly between wacc_low=0.08 and wacc_high=0.12
    r = _reorganized(ebita=66.625, nopat=53.3)
    result = roic_analysis(r, _wacc())

    assert result.roic == pytest.approx(0.10)
    assert result.spread_at_wacc_low == pytest.approx(0.02)
    assert result.spread_at_wacc_high == pytest.approx(-0.02)
    assert result.spread_sign_uncertain is True


def test_spread_sign_not_uncertain_when_band_does_not_straddle_zero():
    r = _reorganized()  # roic ~0.315, well above wacc_high=0.12
    result = roic_analysis(r, _wacc())
    assert result.spread_sign_uncertain is False


def test_decomposition_error_on_inconsistent_inputs():
    # nopat deliberately inconsistent with ebita*(1-tax): roic and
    # margin*turnover*(1-tax) will disagree by far more than 0.5pp
    r = _reorganized(nopat=999.0)
    with pytest.raises(DecompositionError):
        roic_analysis(r, _wacc())


def test_raises_on_zero_average_invested_capital():
    r = _reorganized(average_invested_capital=0.0)
    with pytest.raises(ValueError, match="average_invested_capital"):
        roic_analysis(r, _wacc())


def test_raises_on_non_positive_revenue():
    r = _reorganized(revenue=0.0)
    with pytest.raises(ValueError, match="revenue"):
        roic_analysis(r, _wacc())
