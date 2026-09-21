from __future__ import annotations

import pytest

from nse_screener.workbench.expectations import (
    GROWTH_SEARCH_ROIC_MARGIN,
    NoSolutionError,
    SteadyStateViolationError,
    UnstableSolutionError,
    _two_stage_value,
    two_stage_implied_growth,
)

_NOPAT = 100.0
_ROIC = 0.20
_WACC = 0.10
_YEARS = 10


def test_round_trip_recovers_the_true_growth_rate():
    g_true = 0.05  # safely below both roic and wacc, and >=1pp away from wacc
    enterprise_value, _ = _two_stage_value(_NOPAT, g_true, _ROIC, _WACC, _YEARS)

    result = two_stage_implied_growth(
        symbol="AAA",
        enterprise_value=enterprise_value,
        nopat=_NOPAT,
        roic=_ROIC,
        wacc_low=_WACC,
        wacc_point=_WACC,
        wacc_high=_WACC,
        years_listed=10,
        explicit_years=_YEARS,
    )

    assert result.implied_growth_at_wacc_point == pytest.approx(g_true, abs=1e-6)
    assert result.implied_growth_at_wacc_low == pytest.approx(g_true, abs=1e-6)
    assert result.implied_growth_at_wacc_high == pytest.approx(g_true, abs=1e-6)


def test_round_trip_recovers_growth_above_wacc_but_below_roic():
    # Regression test: an earlier version of this module capped the search
    # range at min(roic, wacc), which wrongly excluded this entire region --
    # exactly where a high-ROIC company's real growth story lives. wacc=0.10,
    # roic=0.20: g_true=0.13 is above wacc, comfortably below roic.
    g_true = 0.13
    enterprise_value, _ = _two_stage_value(_NOPAT, g_true, _ROIC, _WACC, _YEARS)

    result = two_stage_implied_growth(
        symbol="AAA",
        enterprise_value=enterprise_value,
        nopat=_NOPAT,
        roic=_ROIC,
        wacc_low=_WACC,
        wacc_point=_WACC,
        wacc_high=_WACC,
        years_listed=10,
        explicit_years=_YEARS,
    )

    assert result.implied_growth_at_wacc_point == pytest.approx(g_true, abs=1e-6)


def test_continuing_value_share_and_terminal_dominant_flag_are_consistent():
    g_true = 0.03
    enterprise_value, continuing_pv = _two_stage_value(_NOPAT, g_true, _ROIC, _WACC, _YEARS)

    result = two_stage_implied_growth(
        symbol="AAA",
        enterprise_value=enterprise_value,
        nopat=_NOPAT,
        roic=_ROIC,
        wacc_low=_WACC,
        wacc_point=_WACC,
        wacc_high=_WACC,
        years_listed=10,
    )

    expected_share = continuing_pv / enterprise_value
    assert result.continuing_value_share_of_total == pytest.approx(expected_share)
    assert result.terminal_value_dominant == (expected_share > 0.75)


def test_unstable_solution_error_near_roic():
    # g_true within 1pp of roic=0.20 -> the solved root should trip the guard
    # (reinvestment rate g/roic approaching 100%, not a WACC-proximity issue --
    # this two-stage formulation has no WACC-g singularity, see the module docstring)
    g_true = 0.195
    enterprise_value, _ = _two_stage_value(_NOPAT, g_true, _ROIC, _WACC, _YEARS)

    with pytest.raises(UnstableSolutionError):
        two_stage_implied_growth(
            symbol="AAA",
            enterprise_value=enterprise_value,
            nopat=_NOPAT,
            roic=_ROIC,
            wacc_low=_WACC,
            wacc_point=_WACC,
            wacc_high=_WACC,
            years_listed=10,
        )


def test_no_solution_error_when_target_exceeds_max_achievable_value():
    upper_bound_g = _ROIC - GROWTH_SEARCH_ROIC_MARGIN
    max_achievable_value, _ = _two_stage_value(_NOPAT, upper_bound_g, _ROIC, _WACC, _YEARS)
    unreachable_target = max_achievable_value * 10  # nothing in range can reach this

    with pytest.raises(NoSolutionError):
        two_stage_implied_growth(
            symbol="AAA",
            enterprise_value=unreachable_target,
            nopat=_NOPAT,
            roic=_ROIC,
            wacc_low=_WACC,
            wacc_point=_WACC,
            wacc_high=_WACC,
            years_listed=10,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"roic": -0.05},
        {"nopat": -10.0},
        {"years_listed": 3},
    ],
)
def test_steady_state_violation_error(kwargs):
    base = dict(
        symbol="AAA",
        enterprise_value=1000.0,
        nopat=_NOPAT,
        roic=_ROIC,
        wacc_low=_WACC,
        wacc_point=_WACC,
        wacc_high=_WACC,
        years_listed=10,
    )
    base.update(kwargs)
    with pytest.raises(SteadyStateViolationError):
        two_stage_implied_growth(**base)


def test_plausibility_fields_default_to_none_pending_human_input():
    g_true = 0.05
    enterprise_value, _ = _two_stage_value(_NOPAT, g_true, _ROIC, _WACC, _YEARS)
    result = two_stage_implied_growth(
        symbol="AAA",
        enterprise_value=enterprise_value,
        nopat=_NOPAT,
        roic=_ROIC,
        wacc_low=_WACC,
        wacc_point=_WACC,
        wacc_high=_WACC,
        years_listed=10,
    )
    assert result.plausibility_assessment is None
    assert result.plausibility_verdict is None
