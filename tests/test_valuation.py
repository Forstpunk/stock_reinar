from __future__ import annotations

import pytest

from nse_screener.workbench.valuation import UNSTABLE_WACC_MARGIN, apply_graham_label, value_range

_SCENARIOS = {"conservative": 0.02, "base": 0.05, "optimistic": 0.08}


def _range(current_price: float):
    return value_range(
        symbol="AAA",
        nopat=100.0,
        roic=0.20,
        wacc=0.10,
        scenarios=_SCENARIOS,
        net_debt=200.0,
        shares=100.0,
        current_price=current_price,
    )


def test_per_share_values_match_hand_computed_scenarios():
    result = _range(current_price=15.0)

    assert result.per_share_values["conservative"] == pytest.approx(9.25)
    assert result.per_share_values["base"] == pytest.approx(13.0)
    assert result.per_share_values["optimistic"] == pytest.approx(28.0)
    assert result.low == pytest.approx(9.25)
    assert result.high == pytest.approx(28.0)


def test_price_below_range_has_discount_to_low():
    result = _range(current_price=5.0)
    assert result.price_position == "BELOW_RANGE"
    assert result.discount_to_low == pytest.approx((9.25 - 5.0) / 9.25)


def test_price_inside_range_has_no_discount():
    result = _range(current_price=15.0)
    assert result.price_position == "INSIDE_RANGE"
    assert result.discount_to_low is None


def test_price_above_range_has_no_discount():
    result = _range(current_price=30.0)
    assert result.price_position == "ABOVE_RANGE"
    assert result.discount_to_low is None


def test_requires_at_least_three_scenarios():
    with pytest.raises(ValueError, match="at least 3"):
        value_range(
            symbol="AAA", nopat=100.0, roic=0.20, wacc=0.10,
            scenarios={"conservative": 0.02, "base": 0.05},
            net_debt=200.0, shares=100.0, current_price=15.0,
        )


def test_raises_when_scenario_growth_exceeds_roic():
    with pytest.raises(ValueError, match="roic"):
        value_range(
            symbol="AAA", nopat=100.0, roic=0.20, wacc=0.10,
            scenarios={**_SCENARIOS, "aggressive": 0.25},
            net_debt=200.0, shares=100.0, current_price=15.0,
        )


def test_raises_when_scenario_growth_exceeds_wacc():
    with pytest.raises(ValueError, match="wacc"):
        value_range(
            symbol="AAA", nopat=100.0, roic=0.20, wacc=0.10,
            scenarios={**_SCENARIOS, "aggressive": 0.12},
            net_debt=200.0, shares=100.0, current_price=15.0,
        )


def test_no_scenario_flagged_unstable_when_all_comfortably_below_wacc():
    result = _range(current_price=15.0)  # wacc=0.10, max scenario growth=0.08
    assert result.unstable_scenarios == []


def test_scenario_within_margin_of_wacc_is_flagged_unstable():
    scenarios = {
        "conservative": 0.02,
        "base": 0.05,
        "optimistic": 0.10 - UNSTABLE_WACC_MARGIN + 0.001,
    }
    result = value_range(
        symbol="AAA", nopat=100.0, roic=0.20, wacc=0.10, scenarios=scenarios,
        net_debt=200.0, shares=100.0, current_price=15.0,
    )
    assert result.unstable_scenarios == ["optimistic"]
    # the value is still computed and present, just flagged, not hidden or rejected
    assert "optimistic" in result.per_share_values


def test_scenario_at_exactly_the_margin_boundary_is_not_flagged():
    scenarios = {"conservative": 0.02, "base": 0.05, "optimistic": 0.10 - UNSTABLE_WACC_MARGIN}
    result = value_range(
        symbol="AAA", nopat=100.0, roic=0.20, wacc=0.10, scenarios=scenarios,
        net_debt=200.0, shares=100.0, current_price=15.0,
    )
    assert result.unstable_scenarios == []


def test_value_range_loads_from_json_missing_unstable_scenarios_field():
    # Regression test: a session persisted before unstable_scenarios existed
    # must still load -- this is exactly what broke a real saved session.
    from nse_screener.workbench.valuation import ValueRange

    old_schema_json = (
        '{"symbol": "AAA", "per_share_values": {"base": 10.0}, "low": 10.0, '
        '"high": 10.0, "current_price": 8.0, "price_position": "BELOW_RANGE", '
        '"discount_to_low": 0.2}'
    )
    result = ValueRange.model_validate_json(old_schema_json)
    assert result.unstable_scenarios == []


def test_graham_label_defaults_to_none():
    result = _range(current_price=15.0)
    assert result.graham_label is None
    assert result.label_justification is None


def test_investment_label_requires_justification():
    result = _range(current_price=15.0)
    with pytest.raises(ValueError, match="justification"):
        apply_graham_label(result, "INVESTMENT", justification=None)


def test_investment_label_with_justification_succeeds():
    result = _range(current_price=15.0)
    labeled = apply_graham_label(result, "INVESTMENT", justification="thorough analysis done")
    assert labeled.graham_label == "INVESTMENT"
    assert labeled.label_justification == "thorough analysis done"


def test_speculation_label_does_not_require_justification():
    result = _range(current_price=15.0)
    labeled = apply_graham_label(result, "SPECULATION")
    assert labeled.graham_label == "SPECULATION"
