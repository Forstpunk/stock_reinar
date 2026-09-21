"""Analysis session state machine -- the spine of Stage 2.

Palepu's ordering exists to prevent the most expensive analytical error:
valuing numbers before checking whether the numbers are real. This
module enforces that order as a state machine -- stages advance one at a
time, in order, with no bypass flag. A skipped stage is a programming
error, not a shortcut.

Three stages carry irreducible human judgment (business context, the
accounting-quality reasoning note, and stage 4's plausibility/Graham
gates). Where that input is absent, the session simply cannot advance
past the point it's needed -- this module never fabricates it.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from nse_screener.fundamentals.multiples import Multiples
from nse_screener.fundamentals.roic import ROICAnalysis
from nse_screener.workbench.expectations import TwoStageExpectations
from nse_screener.workbench.mscore import AccountingVerdict
from nse_screener.workbench.sizing import PositionPlan
from nse_screener.workbench.valuation import ValueRange


class AnalysisStage(str, Enum):
    BUSINESS_STRATEGY = "1_business_strategy"
    ACCOUNTING_QUALITY = "2_accounting_quality"
    FINANCIAL_ANALYSIS = "3_financial_analysis"
    PROSPECTIVE = "4_prospective"
    COMPLETE = "complete"


class BusinessContext(BaseModel):
    """Palepu step 1. Cannot be computed -- see the module docstring.

    The 100-character minimum exists to make placeholder input
    inconvenient, not to measure quality.
    """

    model_config = ConfigDict(frozen=True)

    industry_economics: str = Field(min_length=100)
    competitive_position: str = Field(min_length=100)
    revenue_drivers: list[str] = Field(min_length=2)
    analyst: str
    entered_at: datetime


class FinancialSummary(BaseModel):
    """Palepu step 3. Multi-year trend/earnings-power history is out of
    scope for now -- see ``roic.py``'s module docstring for why."""

    model_config = ConfigDict(frozen=True)

    roic_analysis: Optional[ROICAnalysis] = None
    multiples: Optional[Multiples] = None
    peer_percentiles: dict[str, float] = {}
    analyst: str
    entered_at: datetime


class ProspectiveAnalysis(BaseModel):
    """Palepu step 4: reverse-DCF implied expectations, the value range,
    and position sizing -- bundled together as the "prospective" stage."""

    model_config = ConfigDict(frozen=True)

    implied_expectations: TwoStageExpectations
    value_range: Optional[ValueRange] = None
    position_plan: Optional[PositionPlan] = None
    analyst: str
    entered_at: datetime


class AnalysisSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    created_at: datetime
    stage: AnalysisStage
    business_context: Optional[BusinessContext] = None
    accounting_verdict: Optional[AccountingVerdict] = None
    financial_summary: Optional[FinancialSummary] = None
    prospective: Optional[ProspectiveAnalysis] = None
    #: ``ANALYSIS_COMPLETE`` (renamed from the build spec's "QUALIFIED" --
    #: see project notes: a "QUALIFIED" label reads as a disguised buy
    #: signal, which this system's constraints explicitly forbid),
    #: ``DISQUALIFIED``, or ``INCOMPLETE``. ``None`` while still in progress.
    terminal_state: Optional[Literal["ANALYSIS_COMPLETE", "DISQUALIFIED", "INCOMPLETE"]] = None


class StageOutOfOrderError(Exception):
    """Raised when a stage-entry function is called out of sequence, or on
    a session that has already reached a terminal state."""

    def __init__(self, symbol: str, attempted: AnalysisStage, current: AnalysisStage) -> None:
        self.symbol = symbol
        self.attempted = attempted
        self.current = current
        super().__init__(
            f"{symbol}: cannot enter stage {attempted.value!r}, session is at "
            f"{current.value!r}"
        )


def start_session(symbol: str, now: Optional[datetime] = None) -> AnalysisSession:
    """Create a fresh session at stage 1. Persist it with ``save_session``
    if the analysis is going to span more than one sitting."""
    return AnalysisSession(
        symbol=symbol,
        created_at=now or datetime.now(),
        stage=AnalysisStage.BUSINESS_STRATEGY,
    )


def _require_stage(session: AnalysisSession, expected: AnalysisStage) -> None:
    if session.terminal_state is not None or session.stage != expected:
        raise StageOutOfOrderError(session.symbol, expected, session.stage)


def enter_business_context(
    session: AnalysisSession, context: BusinessContext
) -> AnalysisSession:
    """Step 1. Requires human input by construction -- ``BusinessContext``
    cannot be built without it (see its field constraints)."""
    _require_stage(session, AnalysisStage.BUSINESS_STRATEGY)
    return session.model_copy(
        update={"business_context": context, "stage": AnalysisStage.ACCOUNTING_QUALITY}
    )


def enter_accounting_verdict(
    session: AnalysisSession, verdict: AccountingVerdict
) -> AnalysisSession:
    """Step 2. A ``DISQUALIFIED`` verdict terminates the whole session here --
    stage does not advance, and no price is ever attached to a disqualified
    company."""
    _require_stage(session, AnalysisStage.ACCOUNTING_QUALITY)
    if verdict.verdict == "DISQUALIFIED":
        return session.model_copy(
            update={"accounting_verdict": verdict, "terminal_state": "DISQUALIFIED"}
        )
    return session.model_copy(
        update={"accounting_verdict": verdict, "stage": AnalysisStage.FINANCIAL_ANALYSIS}
    )


def enter_financial_summary(
    session: AnalysisSession, summary: FinancialSummary
) -> AnalysisSession:
    """Step 3."""
    _require_stage(session, AnalysisStage.FINANCIAL_ANALYSIS)
    return session.model_copy(
        update={"financial_summary": summary, "stage": AnalysisStage.PROSPECTIVE}
    )


def enter_prospective(
    session: AnalysisSession, prospective: ProspectiveAnalysis
) -> AnalysisSession:
    """Step 4. Does not itself advance to COMPLETE -- ``complete_session``
    additionally requires the human plausibility/Graham gates to be filled."""
    _require_stage(session, AnalysisStage.PROSPECTIVE)
    return session.model_copy(update={"prospective": prospective})


def missing_requirements(session: AnalysisSession) -> list[str]:
    """What's missing before this session can be marked ``ANALYSIS_COMPLETE``,
    in Palepu order. Used both by ``complete_session`` and by a future
    ``report`` command to render the "ANALYSIS INCOMPLETE" banner."""
    if session.business_context is None:
        return ["business_context (Palepu step 1)"]
    if session.accounting_verdict is None:
        return ["accounting_verdict (Palepu step 2)"]

    missing: list[str] = []
    if (
        session.accounting_verdict.verdict == "INVESTIGATE"
        and not session.accounting_verdict.reasoning.strip()
    ):
        missing.append("accounting_verdict.reasoning")

    if session.financial_summary is None:
        missing.append("financial_summary (Palepu step 3)")
        return missing

    if session.prospective is None:
        missing.append("prospective analysis (Palepu step 4)")
        return missing
    if session.prospective.implied_expectations.plausibility_verdict is None:
        missing.append("prospective.implied_expectations.plausibility_verdict")
    if (
        session.prospective.value_range is not None
        and session.prospective.value_range.graham_label is None
    ):
        missing.append("prospective.value_range.graham_label")

    return missing


def complete_session(session: AnalysisSession) -> AnalysisSession:
    """Mark the session ``ANALYSIS_COMPLETE``. Raises ``ValueError`` naming
    what's missing if ``missing_requirements`` is non-empty -- completion
    is never granted with an unfilled human gate."""
    _require_stage(session, AnalysisStage.PROSPECTIVE)
    missing = missing_requirements(session)
    if missing:
        raise ValueError(f"{session.symbol}: cannot complete, missing: {', '.join(missing)}")
    return session.model_copy(
        update={"stage": AnalysisStage.COMPLETE, "terminal_state": "ANALYSIS_COMPLETE"}
    )


def mark_incomplete(session: AnalysisSession) -> AnalysisSession:
    """Explicitly close out a session that will not be finished -- e.g. the
    analyst has no business context to give. INCOMPLETE is a valid terminal
    state, not an error state."""
    if session.terminal_state is not None:
        raise ValueError(
            f"{session.symbol}: session already terminated as {session.terminal_state!r}"
        )
    return session.model_copy(update={"terminal_state": "INCOMPLETE"})


def session_path(symbol: str, as_of: date, root: Path = Path("sessions")) -> Path:
    return root / f"{symbol}_{as_of.isoformat()}.json"


def save_session(session: AnalysisSession, root: Path = Path("sessions")) -> Path:
    """Persist as JSON under ``sessions/{symbol}_{date}.json`` so analysis can
    span days. ``date`` is the session's creation date, not today's."""
    path = session_path(session.symbol, session.created_at.date(), root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_session(path: Path) -> AnalysisSession:
    return AnalysisSession.model_validate_json(path.read_text(encoding="utf-8"))
