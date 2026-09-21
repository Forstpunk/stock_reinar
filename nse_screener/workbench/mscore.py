"""Beneish M-Score, and its combination with the v1 forensic screen.

Both this model and v1's forensic accrual screen examine accruals from
different angles -- run them together, not as substitutes, and combine
into one verdict without double-counting.

Holdout validation identifies roughly 76% of manipulators at ~17% false
positives -- meaning about one in four manipulators is missed and
roughly one in six flagged firms is clean. A documented failure: the
model did **not** detect the Toshiba manipulation, while an Altman
Z-score (``fundamentals/distress.py``) did flag distress there. A clean
M-Score is not evidence of clean accounting.

TATA (total accruals to total assets) is the most heavily weighted
variable (coefficient 4.679, the largest in the model), which is why
this overlaps with the v1 accrual screen -- both are fundamentally
accrual-quality checks.

The model is US-derived; applicability to Indian reporting standards is
untested.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

#: Beneish (1999) coefficients, in the order DSRI, GMI, AQI, SGI, DEPI,
#: SGAI, TATA, LVGI. The intercept and TATA's weight (largest magnitude)
#: are fixed, published constants -- not tuned here.
_INTERCEPT: float = -4.84
_W_DSRI: float = 0.920
_W_GMI: float = 0.528
_W_AQI: float = 0.404
_W_SGI: float = 0.892
_W_DEPI: float = 0.115
_W_SGAI: float = -0.172
_W_TATA: float = 4.679
_W_LVGI: float = -0.327

#: Report both bands -- the literature uses each depending on the
#: assumed cost ratio of false negatives to false positives.
ELEVATED_THRESHOLD: float = -1.78
BORDERLINE_THRESHOLD: float = -2.22


class MScoreInputs(BaseModel):
    """Raw line items the eight Beneish indices need for one annual period.

    Several of these (PPE, SG&A, depreciation) are not part of
    ``data.FundamentalData`` -- this is a deliberately separate input
    model; wiring a live fetch for these fields is a distinct task.
    ``income_continuing_ops`` is commonly approximated with net income
    when income from continuing operations is not reported separately --
    document that substitution at the call site if used.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    period_end: date
    revenue: float
    gross_profit: float
    accounts_receivable: float
    current_assets: float
    net_ppe: float
    total_assets: float
    depreciation_expense: float
    sga_expense: float
    total_debt: float
    income_continuing_ops: float
    operating_cash_flow: float
    prior: Optional["MScoreInputs"] = None


MScoreInputs.model_rebuild()


class MScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    m_score: float
    band: Literal["normal", "borderline", "elevated"]
    dsri: float
    gmi: float
    aqi: float
    sgi: float
    depi: float
    sgai: float
    tata: float
    lvgi: float


class MScoreDataError(Exception):
    """Raised when the M-Score cannot be computed from the given inputs."""

    def __init__(self, symbol: str, reason: str) -> None:
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"{symbol}: {reason}")


def beneish_m_score(f: MScoreInputs) -> MScoreResult:
    """Compute the eight Beneish indices and the combined M-Score.

    Requires ``f.prior`` and non-zero denominators throughout (revenue,
    total assets, gross margin, depreciation base, leverage, prior-period
    "soft asset" share) -- raises ``MScoreDataError`` naming the problem
    otherwise, never substitutes a default.
    """
    if f.prior is None:
        raise MScoreDataError(f.symbol, "prior period required, two consecutive years needed")
    t, p = f, f.prior

    if t.revenue <= 0 or p.revenue <= 0:
        raise MScoreDataError(f.symbol, "revenue must be positive in both periods")
    if t.total_assets <= 0 or p.total_assets <= 0:
        raise MScoreDataError(f.symbol, "total_assets must be positive in both periods")

    gm_t, gm_p = t.gross_profit / t.revenue, p.gross_profit / p.revenue
    if gm_t == 0:
        raise MScoreDataError(f.symbol, "current-period gross margin is zero, GMI undefined")
    dsri = (t.accounts_receivable / t.revenue) / (p.accounts_receivable / p.revenue)
    gmi = gm_p / gm_t

    soft_t = 1 - (t.current_assets + t.net_ppe) / t.total_assets
    soft_p = 1 - (p.current_assets + p.net_ppe) / p.total_assets
    if soft_p == 0:
        raise MScoreDataError(f.symbol, "prior-period soft-asset share is zero, AQI undefined")
    aqi = soft_t / soft_p

    sgi = t.revenue / p.revenue

    dep_rate_t = _depreciation_rate(f.symbol, t)
    dep_rate_p = _depreciation_rate(f.symbol, p)
    if dep_rate_t == 0:
        raise MScoreDataError(f.symbol, "current-period depreciation rate is zero, DEPI undefined")
    depi = dep_rate_p / dep_rate_t

    sga_t, sga_p = t.sga_expense / t.revenue, p.sga_expense / p.revenue
    if sga_p == 0:
        raise MScoreDataError(f.symbol, "prior-period SG&A ratio is zero, SGAI undefined")
    sgai = sga_t / sga_p

    leverage_t, leverage_p = t.total_debt / t.total_assets, p.total_debt / p.total_assets
    if leverage_p == 0:
        raise MScoreDataError(f.symbol, "prior-period leverage is zero, LVGI undefined")
    lvgi = leverage_t / leverage_p

    tata = (t.income_continuing_ops - t.operating_cash_flow) / t.total_assets

    m_score = (
        _INTERCEPT
        + _W_DSRI * dsri
        + _W_GMI * gmi
        + _W_AQI * aqi
        + _W_SGI * sgi
        + _W_DEPI * depi
        + _W_SGAI * sgai
        + _W_TATA * tata
        + _W_LVGI * lvgi
    )

    return MScoreResult(
        symbol=f.symbol,
        m_score=m_score,
        band=classify_m_score_band(m_score),
        dsri=dsri,
        gmi=gmi,
        aqi=aqi,
        sgi=sgi,
        depi=depi,
        sgai=sgai,
        tata=tata,
        lvgi=lvgi,
    )


def classify_m_score_band(m_score: float) -> Literal["normal", "borderline", "elevated"]:
    """Boundary is a strict '>' at both thresholds -- a score of exactly -1.78 or
    -2.22 falls into the lower band, matching the documented '> threshold' rule."""
    if m_score > ELEVATED_THRESHOLD:
        return "elevated"
    if m_score > BORDERLINE_THRESHOLD:
        return "borderline"
    return "normal"


def _depreciation_rate(symbol: str, period: MScoreInputs) -> float:
    denom = period.net_ppe + period.depreciation_expense
    if denom <= 0:
        raise MScoreDataError(symbol, "net_ppe + depreciation_expense must be positive")
    return period.depreciation_expense / denom


class AccountingVerdict(BaseModel):
    """Combined accounting-quality verdict for one company.

    Two independent methods (the v1 heuristic forensic screen and this
    module's M-Score) are combined without double-counting: agreement
    between independent methods is stronger evidence than either alone,
    but a single method firing is a prompt to investigate, not proof.
    """

    model_config = ConfigDict(frozen=True)

    forensic_flags: list[str]
    m_score: float
    m_score_band: Literal["normal", "borderline", "elevated"]
    distress_verdict: Optional[Literal["SAFE", "GREY", "DISTRESS"]] = None
    verdict: Literal["CLEAN", "INVESTIGATE", "DISQUALIFIED"]
    reasoning: str
    analyst: str
    entered_at: datetime


class ReasoningRequiredError(Exception):
    """Raised when an INVESTIGATE verdict is requested without an analyst note."""


def combine_accounting_verdict(
    forensic_flags: list[str],
    m_score: MScoreResult,
    analyst: str,
    entered_at: datetime,
    forensic_disqualify_flag_count: int = 2,
    distress_verdict: Optional[Literal["SAFE", "GREY", "DISTRESS"]] = None,
    reasoning: str = "",
) -> AccountingVerdict:
    """Combine the v1 forensic flags, the M-Score band, and (optionally) the
    Altman Z-Score distress verdict into one verdict.

    ``DISQUALIFIED`` only when the forensic screen and M-Score -- the two
    accrual-focused methods -- agree (>= the forensic disqualify
    threshold *and* an elevated M-Score). Either firing alone, or a
    ``distress_verdict`` of ``"DISTRESS"`` (a third, independent method,
    per v3 section 10), yields at least ``INVESTIGATE``, which requires a
    non-empty ``reasoning`` -- raises ``ReasoningRequiredError``
    otherwise, since an unexplained "investigate" is not a completed
    analysis step. ``CLEAN`` only when nothing fires.
    """
    forensic_disqualifies = len(forensic_flags) >= forensic_disqualify_flag_count
    mscore_elevated = m_score.band == "elevated"
    distressed = distress_verdict == "DISTRESS"

    if forensic_disqualifies and mscore_elevated:
        verdict: Literal["CLEAN", "INVESTIGATE", "DISQUALIFIED"] = "DISQUALIFIED"
    elif forensic_disqualifies or mscore_elevated or distressed:
        verdict = "INVESTIGATE"
    else:
        verdict = "CLEAN"

    if verdict == "INVESTIGATE" and not reasoning.strip():
        raise ReasoningRequiredError(
            "an INVESTIGATE verdict requires a non-empty analyst reasoning note"
        )

    return AccountingVerdict(
        forensic_flags=forensic_flags,
        m_score=m_score.m_score,
        m_score_band=m_score.band,
        distress_verdict=distress_verdict,
        verdict=verdict,
        reasoning=reasoning,
        analyst=analyst,
        entered_at=entered_at,
    )
