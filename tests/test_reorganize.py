from __future__ import annotations

from datetime import date

import pytest

from nse_screener.fundamentals.reorganize import (
    RawOperatingFinancials,
    ReorganizationError,
    reorganize,
)

_PRIOR = RawOperatingFinancials(
    symbol="AAA",
    period_end=date(2023, 3, 31),
    revenue=900.0,
    operating_income=180.0,
    amortization_of_intangibles=8.0,
    tax_expense=36.0,
    pretax_income=180.0,
    current_assets=280.0,
    current_liabilities=140.0,
    cash_and_equivalents=45.0,
    short_term_debt=15.0,
    net_ppe=380.0,
    net_other_operating_assets=8.0,
)

_CURRENT = RawOperatingFinancials(
    symbol="AAA",
    period_end=date(2024, 3, 31),
    revenue=1000.0,
    operating_income=200.0,
    amortization_of_intangibles=10.0,
    tax_expense=40.0,
    pretax_income=200.0,
    current_assets=300.0,
    current_liabilities=150.0,
    cash_and_equivalents=50.0,
    short_term_debt=20.0,
    net_ppe=400.0,
    net_other_operating_assets=10.0,
    prior=_PRIOR,
)


def test_reorganize_hand_computed_fixture():
    # ebita = 200 + 10 = 210; tax_rate = 40/200 = 0.20 (unclamped); tax = 42; nopat = 168
    # excess_cash = max(0, 50 - 0.02*1000) = 30; owc = (300-30)-(150-20) = 140
    # current_ic = 140 + 400 + 10 = 550
    # prior: excess_cash = max(0, 45 - 0.02*900) = 27; owc = (280-27)-(140-15) = 128
    # prior_ic = 128 + 380 + 8 = 516
    # average_ic = 533; increase_ic = 34; fcf = 168 - 34 = 134
    r = reorganize(_CURRENT)

    assert r.ebita == pytest.approx(210.0)
    assert r.effective_tax_rate == pytest.approx(0.20)
    assert r.tax_rate_clamped is False
    assert r.operating_tax == pytest.approx(42.0)
    assert r.nopat == pytest.approx(168.0)
    assert r.excess_cash == pytest.approx(30.0)
    assert r.operating_working_capital == pytest.approx(140.0)
    assert r.invested_capital == pytest.approx(550.0)
    assert r.average_invested_capital == pytest.approx(533.0)
    assert r.increase_in_invested_capital == pytest.approx(34.0)
    assert r.free_cash_flow == pytest.approx(134.0)


def test_tax_rate_clamped_above_max_is_flagged():
    f = _CURRENT.model_copy(update={"tax_expense": 180.0})  # raw rate = 0.90
    r = reorganize(f)
    assert r.tax_rate_clamped is True
    assert r.effective_tax_rate == pytest.approx(0.50)


def test_tax_rate_clamped_below_min_is_flagged():
    f = _CURRENT.model_copy(update={"tax_expense": 10.0})  # raw rate = 0.05
    r = reorganize(f)
    assert r.tax_rate_clamped is True
    assert r.effective_tax_rate == pytest.approx(0.10)


def test_tax_rate_within_band_is_not_clamped():
    r = reorganize(_CURRENT)  # raw rate = 0.20, within [0.10, 0.50]
    assert r.tax_rate_clamped is False


def test_raises_on_non_positive_pretax_income():
    f = _CURRENT.model_copy(update={"pretax_income": 0.0})
    with pytest.raises(ReorganizationError, match="pretax_income"):
        reorganize(f)


def test_raises_when_prior_period_missing():
    f = _CURRENT.model_copy(update={"prior": None})
    with pytest.raises(ReorganizationError, match="prior period"):
        reorganize(f)


def test_excess_cash_is_zero_when_cash_below_threshold():
    f = _CURRENT.model_copy(update={"cash_and_equivalents": 5.0})  # < 2% of 1000 = 20
    r = reorganize(f)
    assert r.excess_cash == pytest.approx(0.0)
