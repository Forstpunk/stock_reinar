from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from nse_screener.workbench.expectations import TwoStageExpectations
from nse_screener.workbench.mscore import AccountingVerdict
from nse_screener.workbench.session import (
    AnalysisStage,
    BusinessContext,
    FinancialSummary,
    ProspectiveAnalysis,
    StageOutOfOrderError,
    complete_session,
    enter_accounting_verdict,
    enter_business_context,
    enter_financial_summary,
    enter_prospective,
    load_session,
    mark_incomplete,
    missing_requirements,
    save_session,
    start_session,
)
from nse_screener.workbench.valuation import ValueRange

_ENTERED_AT = datetime(2024, 1, 1)


def _business_context() -> BusinessContext:
    return BusinessContext(
        industry_economics="x" * 100,
        competitive_position="y" * 100,
        revenue_drivers=["driver one", "driver two"],
        analyst="Analyst A",
        entered_at=_ENTERED_AT,
    )


def _verdict(verdict: str = "CLEAN") -> AccountingVerdict:
    return AccountingVerdict(
        forensic_flags=["f1", "f2"] if verdict == "DISQUALIFIED" else [],
        m_score=-1.0 if verdict == "DISQUALIFIED" else -3.0,
        m_score_band="elevated" if verdict == "DISQUALIFIED" else "normal",
        verdict=verdict,
        reasoning="",
        analyst="Analyst A",
        entered_at=_ENTERED_AT,
    )


def _financial_summary() -> FinancialSummary:
    return FinancialSummary(analyst="Analyst A", entered_at=_ENTERED_AT)


def _implied_expectations(plausibility_verdict=None) -> TwoStageExpectations:
    return TwoStageExpectations(
        symbol="AAA",
        implied_growth_at_wacc_low=0.03,
        implied_growth_at_wacc_point=0.05,
        implied_growth_at_wacc_high=0.07,
        continuing_value_share_of_total=0.6,
        terminal_value_dominant=False,
        historical_revenue_cagr_3y=None,
        historical_revenue_cagr_5y=None,
        historical_nopat_cagr_3y=None,
        historical_nopat_cagr_5y=None,
        explicit_years=10,
        plausibility_verdict=plausibility_verdict,
    )


def _value_range(graham_label=None) -> ValueRange:
    return ValueRange(
        symbol="AAA",
        per_share_values={"base": 10.0},
        low=10.0,
        high=10.0,
        current_price=8.0,
        price_position="BELOW_RANGE",
        discount_to_low=0.2,
        unstable_scenarios=[],
        graham_label=graham_label,
        label_justification="thorough analysis" if graham_label == "INVESTMENT" else None,
    )


def _prospective(plausibility_verdict=None, graham_label=None) -> ProspectiveAnalysis:
    return ProspectiveAnalysis(
        implied_expectations=_implied_expectations(plausibility_verdict),
        value_range=_value_range(graham_label),
        position_plan=None,
        analyst="Analyst A",
        entered_at=_ENTERED_AT,
    )


def _fresh():
    return start_session("AAA", now=_ENTERED_AT)


def _at_stage2():
    return enter_business_context(_fresh(), _business_context())


def _at_stage3():
    return enter_accounting_verdict(_at_stage2(), _verdict("CLEAN"))


def _at_stage4():
    return enter_financial_summary(_at_stage3(), _financial_summary())


def test_start_session_begins_at_stage_1():
    session = _fresh()
    assert session.stage == AnalysisStage.BUSINESS_STRATEGY
    assert session.terminal_state is None


def test_stage_out_of_order_on_every_skip_attempt():
    fresh = _fresh()
    with pytest.raises(StageOutOfOrderError):
        enter_accounting_verdict(fresh, _verdict())
    with pytest.raises(StageOutOfOrderError):
        enter_financial_summary(fresh, _financial_summary())
    with pytest.raises(StageOutOfOrderError):
        enter_prospective(fresh, _prospective())
    with pytest.raises(StageOutOfOrderError):
        complete_session(fresh)


def test_stage_out_of_order_when_re_entering_a_passed_stage():
    at2 = _at_stage2()
    with pytest.raises(StageOutOfOrderError):
        enter_business_context(at2, _business_context())


def test_business_context_enforces_minimum_length():
    with pytest.raises(ValidationError):
        BusinessContext(
            industry_economics="too short",
            competitive_position="y" * 100,
            revenue_drivers=["a", "b"],
            analyst="A",
            entered_at=_ENTERED_AT,
        )


def test_business_context_enforces_minimum_revenue_drivers():
    with pytest.raises(ValidationError):
        BusinessContext(
            industry_economics="x" * 100,
            competitive_position="y" * 100,
            revenue_drivers=["only one"],
            analyst="A",
            entered_at=_ENTERED_AT,
        )


def test_enter_business_context_advances_to_stage_2():
    session = _at_stage2()
    assert session.stage == AnalysisStage.ACCOUNTING_QUALITY
    assert session.business_context is not None


def test_clean_accounting_verdict_advances_to_stage_3():
    session = _at_stage3()
    assert session.stage == AnalysisStage.FINANCIAL_ANALYSIS
    assert session.terminal_state is None


def test_disqualified_verdict_blocks_advancement():
    at2 = _at_stage2()
    disqualified = enter_accounting_verdict(at2, _verdict("DISQUALIFIED"))
    assert disqualified.terminal_state == "DISQUALIFIED"
    assert disqualified.stage == AnalysisStage.ACCOUNTING_QUALITY  # never advanced
    with pytest.raises(StageOutOfOrderError):
        enter_financial_summary(disqualified, _financial_summary())


def test_missing_requirements_lists_business_context_first():
    assert missing_requirements(_fresh()) == ["business_context (Palepu step 1)"]


def test_missing_requirements_lists_accounting_verdict_next():
    assert missing_requirements(_at_stage2()) == ["accounting_verdict (Palepu step 2)"]


def test_complete_session_requires_human_gates():
    at4 = _at_stage4()
    with_prospective = enter_prospective(at4, _prospective())  # no plausibility/label set
    with pytest.raises(ValueError, match="missing"):
        complete_session(with_prospective)


def test_complete_session_succeeds_when_all_gates_filled():
    at4 = _at_stage4()
    with_prospective = enter_prospective(
        at4, _prospective(plausibility_verdict="FAIR", graham_label="INVESTMENT")
    )
    completed = complete_session(with_prospective)
    assert completed.terminal_state == "ANALYSIS_COMPLETE"
    assert completed.stage == AnalysisStage.COMPLETE


def test_mark_incomplete_sets_terminal_state():
    session = mark_incomplete(_fresh())
    assert session.terminal_state == "INCOMPLETE"


def test_mark_incomplete_raises_if_already_terminated():
    disqualified = enter_accounting_verdict(_at_stage2(), _verdict("DISQUALIFIED"))
    with pytest.raises(ValueError, match="already terminated"):
        mark_incomplete(disqualified)


def test_save_and_load_session_roundtrip(tmp_path):
    session = _at_stage3()
    path = save_session(session, root=tmp_path)
    assert path.exists()
    assert path.name == "AAA_2024-01-01.json"

    loaded = load_session(path)
    assert loaded.symbol == session.symbol
    assert loaded.stage == session.stage
    assert loaded.accounting_verdict is not None
    assert loaded.accounting_verdict.verdict == "CLEAN"
