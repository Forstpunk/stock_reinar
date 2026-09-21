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


# -- Undefined ratios are absence of evidence, not evidence --------------------


def _asset_light_pair():
    """Debt-free, inventory-free services company: CFO > NI, positive earnings."""
    base = dict(
        symbol="T",
        period_end=date(2025, 3, 31),
        total_revenue=1000.0,
        gross_profit=400.0,
        net_income=100.0,
        operating_cash_flow=150.0,
        total_assets=2000.0,
        total_equity=1000.0,
        total_debt=0.0,
        current_assets=500.0,
        current_liabilities=250.0,
        shares_outstanding=100.0,
        accounts_receivable=100.0,
        inventory=0.0,
    )
    prior = FundamentalData(**{**base, "period_end": date(2024, 3, 31)})
    current = FundamentalData(**base)
    return current, prior


def test_debt_free_inventory_free_healthy_company_is_not_disqualified():
    current, prior = _asset_light_pair()
    result = forensic_screen(current, prior)
    assert result.flags == []
    assert result.disqualified is False
    assert len(result.undefined_checks) == 2
    assert any("inventory" in check for check in result.undefined_checks)
    assert any("leverage" in check for check in result.undefined_checks)


def test_zero_net_income_is_undefined_check_not_flag():
    current, prior = _make({"net_income": 0.0})
    result = forensic_screen(current, prior)
    assert result.flags == []
    assert result.disqualified is False
    assert len(result.undefined_checks) == 1
    assert "cfo_ni_ratio" in result.undefined_checks[0]


def test_two_genuine_breaches_still_disqualify():
    current, prior = _make({"operating_cash_flow": 50.0, "total_debt": 400.0})
    result = forensic_screen(current, prior)
    assert len(result.flags) == 2
    assert result.undefined_checks == []
    assert result.disqualified is True


def test_one_genuine_flag_plus_three_undefined_checks_is_not_disqualified():
    # Genuine breach: accruals far above cash flow (accrual_ratio > 0.10).
    # Undefined: zero net income (cfo_ni), zero inventory both periods
    # (inventory growth), zero debt both periods (leverage jump).
    current, prior = _make(
        {"net_income": 0.0, "operating_cash_flow": -200.0, "inventory": 0.0, "total_debt": 0.0},
        {"inventory": 0.0, "total_debt": 0.0},
    )
    result = forensic_screen(current, prior)
    assert len(result.flags) == 1
    assert "accrual_ratio" in result.flags[0]
    assert len(result.undefined_checks) == 3
    assert result.disqualified is False


# -- Immaterial line items cannot support a growth comparison ------------------


def _asset_heavy_it_services_pair(**current_overrides: float):
    """TCS-like: 29cr inventory on a 150,000cr balance sheet (0.02% of assets).

    The year-on-year inventory move (21cr -> 29cr, +38%) is large in
    percentage terms and meaningless in economic terms.
    """
    base = dict(
        symbol="TCSLIKE",
        period_end=date(2025, 3, 31),
        total_revenue=255000.0,
        gross_profit=110000.0,
        net_income=48000.0,
        operating_cash_flow=52000.0,
        total_assets=150000.0,
        total_equity=95000.0,
        total_debt=8000.0,
        current_assets=90000.0,
        current_liabilities=40000.0,
        shares_outstanding=362.0,
        accounts_receivable=45000.0,
        inventory=29.0,
    )
    prior = FundamentalData(
        **{
            **base,
            "period_end": date(2024, 3, 31),
            "total_revenue": 240000.0,
            "inventory": 21.0,
        }
    )
    current = FundamentalData(**{**base, **current_overrides})
    return current, prior


def test_immaterial_inventory_is_undefined_check_not_flag():
    current, prior = _asset_heavy_it_services_pair()
    result = forensic_screen(current, prior)
    assert result.flags == []
    assert result.disqualified is False
    assert any("inventory" in check and "immaterial" in check for check in result.undefined_checks)


def test_immaterial_inventory_plus_one_genuine_flag_is_not_disqualified():
    # CFO/NI = 35000/48000 = 0.73 -- a real flag. Immaterial inventory must
    # not supply the second flag that would disqualify a healthy company.
    current, prior = _asset_heavy_it_services_pair(operating_cash_flow=35000.0)
    result = forensic_screen(current, prior)
    assert len(result.flags) == 1
    assert "cfo_ni_ratio" in result.flags[0]
    assert result.disqualified is False


def test_material_inventory_build_still_flags():
    # TATASTEEL-like: inventory ~15% of assets; a genuine build against flat
    # COGS must still fire -- the materiality gate is not a loophole.
    base = dict(
        symbol="STEELLIKE",
        period_end=date(2025, 3, 31),
        total_revenue=220000.0,
        gross_profit=60000.0,
        net_income=8000.0,
        operating_cash_flow=20000.0,
        total_assets=280000.0,
        total_equity=90000.0,
        total_debt=80000.0,
        current_assets=90000.0,
        current_liabilities=70000.0,
        shares_outstanding=1248.0,
        accounts_receivable=8000.0,
        inventory=44000.0,
    )
    prior = FundamentalData(
        **{**base, "period_end": date(2024, 3, 31), "inventory": 30000.0}
    )
    result = forensic_screen(FundamentalData(**base), prior)
    assert any("inventory_growth" in flag for flag in result.flags)


def test_immaterial_receivables_is_undefined_check_not_flag():
    # Receivables 0.5% of assets in both periods, and nearly tripling: the
    # DSO comparison is arithmetically valid and economically meaningless.
    current, prior = _make(
        {"total_assets": 100000.0, "accounts_receivable": 500.0},
        {"total_assets": 100000.0, "accounts_receivable": 180.0},
    )
    result = forensic_screen(current, prior)
    assert not any("dso_growth" in flag for flag in result.flags)
    assert any(
        "dso_growth" in check and "immaterial" in check for check in result.undefined_checks
    )


def test_inventory_at_exactly_materiality_threshold_is_evaluated():
    # Exactly 2.0% of assets in both periods: the check applies, and a build
    # against flat COGS fires.
    current, prior = _make(
        {"total_assets": 10000.0, "inventory": 200.0},
        {"total_assets": 5000.0, "inventory": 100.0},
    )
    result = forensic_screen(current, prior)
    assert any("inventory_growth" in flag for flag in result.flags)
    assert not any("inventory" in check for check in result.undefined_checks)


# -- Negative equity makes leverage incomparable -------------------------------


def test_negative_prior_equity_is_undefined_check_not_silent_pass():
    # prior D/E = -2.0, current D/E = +0.32: a sign change in equity is not a
    # comparable leverage trajectory.
    current, prior = _make({}, {"total_equity": -200.0, "total_debt": 400.0})
    result = forensic_screen(current, prior)
    assert result.flags == []
    assert any(
        "leverage_jump" in check and "prior" in check and "-200" in check
        for check in result.undefined_checks
    )


def test_negative_current_equity_is_undefined_check_not_silent_pass():
    current, prior = _make({"total_equity": -50.0})
    result = forensic_screen(current, prior)
    assert not any("leverage_jump" in flag for flag in result.flags)
    assert any(
        "leverage_jump" in check and "current" in check and "-50" in check
        for check in result.undefined_checks
    )


def test_genuine_leverage_jump_with_positive_equity_still_flags():
    current, prior = _make({"total_debt": 400.0})
    result = forensic_screen(current, prior)
    assert len(result.flags) == 1
    assert "leverage_jump" in result.flags[0]
    assert result.undefined_checks == []
