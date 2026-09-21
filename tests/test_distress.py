from __future__ import annotations

import pytest

from nse_screener.fundamentals.distress import (
    DISTRESS_THRESHOLD,
    SAFE_THRESHOLD,
    DistressInputs,
    altman_z_score,
    classify_z_band,
)
from nse_screener.fundamentals.peers import NotApplicableForSector, SectorInfo

_INDUSTRIAL = SectorInfo(symbol="AAA", sector="Energy", industry="Oil Gas")
_BANK = SectorInfo(symbol="BBB", sector="Financial Services", industry="Private Sector Bank")


def _inputs(**overrides: float) -> DistressInputs:
    base = dict(
        symbol="AAA",
        working_capital=150.0,
        total_assets=1000.0,
        retained_earnings=300.0,
        ebit=120.0,
        market_cap=2000.0,
        total_liabilities=500.0,
        revenue=1100.0,
        interest_expense=40.0,
        net_debt=300.0,
        ebitda=200.0,
        current_assets=400.0,
        current_liabilities=200.0,
        inventory=100.0,
        cash_and_equivalents=50.0,
    )
    base.update(overrides)
    return DistressInputs(**base)


def test_z_score_and_coverage_metrics_match_hand_computed_values():
    result = altman_z_score(_inputs(), _INDUSTRIAL)

    expected_z = 1.2 * 0.15 + 1.4 * 0.3 + 3.3 * 0.12 + 0.6 * 4.0 + 1.0 * 1.1
    assert result.z_score == pytest.approx(expected_z)
    assert result.verdict == "SAFE"
    assert result.interest_coverage == pytest.approx(3.0)
    assert result.net_debt_to_ebitda == pytest.approx(1.5)
    assert result.current_ratio == pytest.approx(2.0)
    assert result.acid_test_ratio == pytest.approx(1.5)
    assert result.cash_ratio == pytest.approx(0.25)


def test_raises_not_applicable_for_financial_sector():
    with pytest.raises(NotApplicableForSector):
        altman_z_score(_inputs(), _BANK)


def test_band_boundary_at_exactly_safe_threshold():
    assert classify_z_band(SAFE_THRESHOLD) == "GREY"  # not strictly > 2.99
    assert classify_z_band(SAFE_THRESHOLD + 0.001) == "SAFE"


def test_band_boundary_at_exactly_distress_threshold():
    assert classify_z_band(DISTRESS_THRESHOLD) == "GREY"  # >= 1.81
    assert classify_z_band(DISTRESS_THRESHOLD - 0.001) == "DISTRESS"


def test_interest_coverage_is_none_for_debt_free_company():
    result = altman_z_score(_inputs(interest_expense=0.0), _INDUSTRIAL)
    assert result.interest_coverage is None


def test_raises_on_zero_ebitda():
    with pytest.raises(ValueError, match="ebitda"):
        altman_z_score(_inputs(ebitda=0.0), _INDUSTRIAL)


def test_raises_on_non_positive_total_liabilities():
    with pytest.raises(ValueError, match="total_liabilities"):
        altman_z_score(_inputs(total_liabilities=0.0), _INDUSTRIAL)
