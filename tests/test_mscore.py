from __future__ import annotations

from datetime import date, datetime

import pytest

from nse_screener.workbench.mscore import (
    BORDERLINE_THRESHOLD,
    ELEVATED_THRESHOLD,
    AccountingVerdict,
    MScoreDataError,
    MScoreInputs,
    ReasoningRequiredError,
    beneish_m_score,
    classify_m_score_band,
    combine_accounting_verdict,
)

_PRIOR = MScoreInputs(
    symbol="AAA",
    period_end=date(2023, 3, 31),
    revenue=1000.0,
    gross_profit=400.0,
    accounts_receivable=100.0,
    current_assets=300.0,
    net_ppe=380.0,
    total_assets=1000.0,
    depreciation_expense=40.0,
    sga_expense=90.0,
    total_debt=200.0,
    income_continuing_ops=100.0,
    operating_cash_flow=110.0,
)

_CURRENT = MScoreInputs(
    symbol="AAA",
    period_end=date(2024, 3, 31),
    revenue=1100.0,
    gross_profit=440.0,
    accounts_receivable=110.0,
    current_assets=330.0,
    net_ppe=400.0,
    total_assets=1100.0,
    depreciation_expense=50.0,
    sga_expense=100.0,
    total_debt=210.0,
    income_continuing_ops=110.0,
    operating_cash_flow=120.0,
    prior=_PRIOR,
)


def test_each_index_matches_hand_computed_formula():
    result = beneish_m_score(_CURRENT)

    assert result.dsri == pytest.approx((110.0 / 1100.0) / (100.0 / 1000.0))
    assert result.gmi == pytest.approx((400.0 / 1000.0) / (440.0 / 1100.0))
    assert result.aqi == pytest.approx(
        (1 - (330.0 + 400.0) / 1100.0) / (1 - (300.0 + 380.0) / 1000.0)
    )
    assert result.sgi == pytest.approx(1100.0 / 1000.0)
    assert result.depi == pytest.approx(
        (40.0 / (380.0 + 40.0)) / (50.0 / (400.0 + 50.0))
    )
    assert result.sgai == pytest.approx(
        (100.0 / 1100.0) / (90.0 / 1000.0)
    )
    assert result.lvgi == pytest.approx(
        (210.0 / 1100.0) / (200.0 / 1000.0)
    )
    assert result.tata == pytest.approx((110.0 - 120.0) / 1100.0)


def test_m_score_combination_formula_and_band_are_consistent():
    result = beneish_m_score(_CURRENT)

    expected_m = (
        -4.84
        + 0.920 * result.dsri
        + 0.528 * result.gmi
        + 0.404 * result.aqi
        + 0.892 * result.sgi
        + 0.115 * result.depi
        - 0.172 * result.sgai
        + 4.679 * result.tata
        - 0.327 * result.lvgi
    )
    assert result.m_score == pytest.approx(expected_m)
    assert result.band == classify_m_score_band(result.m_score)


def test_band_boundary_at_exactly_elevated_threshold():
    assert classify_m_score_band(ELEVATED_THRESHOLD) == "borderline"  # not > threshold
    assert classify_m_score_band(ELEVATED_THRESHOLD + 0.0001) == "elevated"


def test_band_boundary_at_exactly_borderline_threshold():
    assert classify_m_score_band(BORDERLINE_THRESHOLD) == "normal"  # not > threshold
    assert classify_m_score_band(BORDERLINE_THRESHOLD + 0.0001) == "borderline"


def test_raises_on_missing_prior_year_data():
    current_no_prior = _CURRENT.model_copy(update={"prior": None})
    with pytest.raises(MScoreDataError, match="prior period"):
        beneish_m_score(current_no_prior)


def test_raises_on_zero_current_gross_margin():
    f = _CURRENT.model_copy(update={"gross_profit": 0.0})
    with pytest.raises(MScoreDataError, match="gross margin"):
        beneish_m_score(f)


# -- Combined accounting verdict ----------------------------------------------


def _mscore_result(band: str, m_score: float = -3.0) -> object:
    result = beneish_m_score(_CURRENT)
    return result.model_copy(update={"band": band, "m_score": m_score})


_ANALYST = "Test Analyst"
_ENTERED_AT = datetime(2024, 1, 1)


def test_clean_when_neither_method_fires():
    verdict = combine_accounting_verdict(
        [], _mscore_result("normal"), analyst=_ANALYST, entered_at=_ENTERED_AT
    )
    assert verdict.verdict == "CLEAN"
    assert verdict.analyst == _ANALYST
    assert verdict.entered_at == _ENTERED_AT


def test_investigate_when_only_forensic_fires():
    verdict = combine_accounting_verdict(
        ["flag one", "flag two"], _mscore_result("normal"),
        analyst=_ANALYST, entered_at=_ENTERED_AT, reasoning="checking the flags",
    )
    assert verdict.verdict == "INVESTIGATE"


def test_investigate_when_only_mscore_fires():
    verdict = combine_accounting_verdict(
        [], _mscore_result("elevated", m_score=-1.0),
        analyst=_ANALYST, entered_at=_ENTERED_AT, reasoning="checking elevated m-score",
    )
    assert verdict.verdict == "INVESTIGATE"


def test_disqualified_when_both_methods_agree():
    verdict = combine_accounting_verdict(
        ["flag one", "flag two"], _mscore_result("elevated", m_score=-1.0),
        analyst=_ANALYST, entered_at=_ENTERED_AT,
    )
    assert verdict.verdict == "DISQUALIFIED"


def test_investigate_requires_reasoning():
    with pytest.raises(ReasoningRequiredError):
        combine_accounting_verdict(
            ["flag one", "flag two"], _mscore_result("normal"),
            analyst=_ANALYST, entered_at=_ENTERED_AT,
        )


def test_disqualified_does_not_require_reasoning():
    verdict = combine_accounting_verdict(
        ["flag one", "flag two"], _mscore_result("elevated", m_score=-1.0),
        analyst=_ANALYST, entered_at=_ENTERED_AT,
    )
    assert isinstance(verdict, AccountingVerdict)


def test_distress_alone_forces_at_least_investigate():
    verdict = combine_accounting_verdict(
        [], _mscore_result("normal"), analyst=_ANALYST, entered_at=_ENTERED_AT,
        distress_verdict="DISTRESS", reasoning="Z-score flagged",
    )
    assert verdict.verdict == "INVESTIGATE"
    assert verdict.distress_verdict == "DISTRESS"


def test_safe_and_grey_distress_do_not_force_investigate():
    verdict = combine_accounting_verdict(
        [], _mscore_result("normal"), analyst=_ANALYST, entered_at=_ENTERED_AT,
        distress_verdict="SAFE",
    )
    assert verdict.verdict == "CLEAN"
