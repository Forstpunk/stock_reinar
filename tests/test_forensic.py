from __future__ import annotations

from datetime import date

from nse_screener.data import FundamentalData
from nse_screener.factors.forensic import forensic_screen

_PRIOR_KWARGS = dict(
    symbol="AAA",
    period_end=date(2023, 3, 31),
    total_revenue=1000.0,
    gross_profit=400.0,
    net_income=100.0,
    operating_cash_flow=110.0,
    total_assets=1000.0,
    total_equity=600.0,
    total_debt=200.0,
    current_assets=300.0,
    current_liabilities=150.0,
    shares_outstanding=100.0,
    accounts_receivable=100.0,
    inventory=80.0,
)

_CURRENT_KWARGS = dict(
    symbol="AAA",
    period_end=date(2024, 3, 31),
    total_revenue=1100.0,
    gross_profit=440.0,
    net_income=110.0,
    operating_cash_flow=120.0,
    total_assets=1100.0,
    total_equity=650.0,
    total_debt=210.0,
    current_assets=330.0,
    current_liabilities=160.0,
    shares_outstanding=100.0,
    accounts_receivable=110.0,
    inventory=88.0,
)


def _make(overrides_current: dict | None = None, overrides_prior: dict | None = None):
    current = FundamentalData(**{**_CURRENT_KWARGS, **(overrides_current or {})})
    prior = FundamentalData(**{**_PRIOR_KWARGS, **(overrides_prior or {})})
    return current, prior


def test_clean_pair_has_no_flags():
    current, prior = _make()
    result = forensic_screen(current, prior)
    assert result.flags == []
    assert result.disqualified is False


def test_cfo_ni_ratio_flag_fires_alone():
    current, prior = _make({"operating_cash_flow": 50.0})
    result = forensic_screen(current, prior)
    assert len(result.flags) == 1
    assert "cfo_ni_ratio" in result.flags[0]
    assert result.disqualified is False


def test_accrual_ratio_flag_fires_alone():
    current, prior = _make({"net_income": 700.0, "operating_cash_flow": 560.0})
    result = forensic_screen(current, prior)
    assert len(result.flags) == 1
    assert "accrual_ratio" in result.flags[0]
    assert result.disqualified is False


def test_dso_growth_flag_fires_alone():
    current, prior = _make({"accounts_receivable": 300.0})
    result = forensic_screen(current, prior)
    assert len(result.flags) == 1
    assert "dso_growth" in result.flags[0]
    assert result.disqualified is False


def test_inventory_growth_flag_fires_alone():
    current, prior = _make({"inventory": 200.0})
    result = forensic_screen(current, prior)
    assert len(result.flags) == 1
    assert "inventory_growth" in result.flags[0]
    assert result.disqualified is False


def test_leverage_jump_flag_fires_alone():
    current, prior = _make({"total_debt": 400.0})
    result = forensic_screen(current, prior)
    assert len(result.flags) == 1
    assert "leverage_jump" in result.flags[0]
    assert result.disqualified is False


def test_disqualified_at_exactly_two_flags():
    current, prior = _make({"operating_cash_flow": 50.0, "total_debt": 400.0})
    result = forensic_screen(current, prior)
    assert len(result.flags) == 2
    assert result.disqualified is True


def test_not_disqualified_at_one_flag_below_threshold():
    current, prior = _make({"operating_cash_flow": 50.0}, {})
    result = forensic_screen(current, prior, disqualify_flag_count=2)
    assert len(result.flags) == 1
    assert result.disqualified is False
