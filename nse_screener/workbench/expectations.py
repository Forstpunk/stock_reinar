"""Reverse DCF / implied expectations -- built directly as two-stage.

Forward DCF is unreliable because terminal value typically drives
60-80% of total enterprise value, making the output an artifact of a
perpetuity assumption. Solving for what the price already implies
converts an unreliable forecast into a testable plausibility question.

Built two-stage from the start (v3 section 9): the single-stage
perpetuity is only valid at steady state. Stage 1 grows NOPAT explicitly
for ``explicit_years`` at the solved growth rate, reinvesting at
``g / roic`` each year (the McKinsey key-value-driver reinvestment
rate). Stage 2 assumes ROIC fades to WACC -- competitive advantage
periods end -- at which point the standard key-value-driver formula
collapses to ``NOPAT / WACC`` regardless of the terminal growth
assumption (an algebraic identity: ``(1 - g/ROIC)/(WACC-g) = 1/WACC``
exactly when ``ROIC == WACC``), which is why no separate terminal-growth
parameter is exposed -- one is not needed once the spread reaches zero.

The tool computes the number and stops. It never auto-classifies
plausibility -- ``plausibility_assessment``/``plausibility_verdict`` are
filled in by the analyst, and the stage is incomplete without them.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict

#: Bisection search range floor, and the margin kept below ROIC.
#:
#: Unlike v2's original single-stage closed-form perpetuity
#: (``NOPAT*(1-g/ROIC)/(WACC-g)``), this two-stage model discounts stage-1
#: cash flows year by year and prices the terminal year at ``NOPAT/WACC``
#: -- g never appears in a ``WACC - g`` denominator anywhere in this
#: formulation, so there is no mathematical singularity at ``g == WACC``,
#: and capping the search there (as an earlier version of this module
#: did) wrongly excluded the entire region above WACC where a
#: high-ROIC company's genuine growth story lives. The real ceiling here
#: is ``ROIC``: at ``g >= roic`` the reinvestment rate (``g / roic``)
#: reaches or exceeds 100%, so ``FCF <= 0`` in every explicit year --
#: not invalid, but a qualitatively different regime this bisection
#: range is not meant to search.
GROWTH_SEARCH_FLOOR: float = -0.10
GROWTH_SEARCH_ROIC_MARGIN: float = 0.001

#: A solved growth rate within this distance of ROIC is treated as
#: unstable -- reinvestment approaches or exceeds 100% of NOPAT, an
#: economically extreme assumption and a common sign of a
#: poorly-conditioned root close to the search boundary.
UNSTABLE_BAND: float = 0.01

#: If continuing value exceeds this share of total value, the valuation
#: is primarily an assumption about the far future, not an analysis of
#: the business today.
TERMINAL_VALUE_DOMINANT_THRESHOLD: float = 0.75

MIN_YEARS_LISTED: int = 5

_BISECTION_MAX_ITERATIONS: int = 100
_BISECTION_TOLERANCE: float = 1e-9


class NoSolutionError(Exception):
    """Raised when no growth rate in the search range reproduces ``enterprise_value``."""


class UnstableSolutionError(Exception):
    """Raised when the solved growth rate lands within ``UNSTABLE_BAND`` of WACC."""


class SteadyStateViolationError(Exception):
    """Raised when the steady-state assumptions this model requires don't hold:
    negative ROIC, non-positive NOPAT, or under ``MIN_YEARS_LISTED`` years listed."""


class TwoStageExpectations(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    implied_growth_at_wacc_low: float
    implied_growth_at_wacc_point: float
    implied_growth_at_wacc_high: float
    continuing_value_share_of_total: float
    terminal_value_dominant: bool
    historical_revenue_cagr_3y: Optional[float]
    historical_revenue_cagr_5y: Optional[float]
    historical_nopat_cagr_3y: Optional[float]
    historical_nopat_cagr_5y: Optional[float]
    explicit_years: int
    #: HUMAN INPUT REQUIRED. Stage incomplete without these.
    plausibility_assessment: Optional[str] = None
    plausibility_verdict: Optional[Literal["LOW", "FAIR", "HEROIC"]] = None


def two_stage_implied_growth(
    symbol: str,
    enterprise_value: float,
    nopat: float,
    roic: float,
    wacc_low: float,
    wacc_point: float,
    wacc_high: float,
    years_listed: int,
    explicit_years: int = 10,
    historical_revenue_cagr_3y: Optional[float] = None,
    historical_revenue_cagr_5y: Optional[float] = None,
    historical_nopat_cagr_3y: Optional[float] = None,
    historical_nopat_cagr_5y: Optional[float] = None,
) -> TwoStageExpectations:
    """Solve implied stage-1 growth at each of the three WACC-band points.

    Raises ``SteadyStateViolationError`` if ``roic <= 0``, ``nopat <= 0``,
    or ``years_listed < MIN_YEARS_LISTED`` -- the perpetuity form assumes
    steady state, and companies far from it produce implied numbers that
    describe a fiction. Raises ``NoSolutionError`` if no root exists in
    the search range for a given WACC, or ``UnstableSolutionError`` if
    the solved growth lands within 1pp of that WACC.
    """
    if roic <= 0:
        raise SteadyStateViolationError(f"{symbol}: ROIC must be positive, got {roic}")
    if nopat <= 0:
        raise SteadyStateViolationError(f"{symbol}: NOPAT must be positive, got {nopat}")
    if years_listed < MIN_YEARS_LISTED:
        raise SteadyStateViolationError(
            f"{symbol}: only {years_listed} years listed, need >= {MIN_YEARS_LISTED}"
        )
    if enterprise_value <= 0:
        raise ValueError(f"{symbol}: enterprise_value must be positive")

    g_low = _solve_growth(symbol, enterprise_value, nopat, roic, wacc_low, explicit_years)
    g_point = _solve_growth(symbol, enterprise_value, nopat, roic, wacc_point, explicit_years)
    g_high = _solve_growth(symbol, enterprise_value, nopat, roic, wacc_high, explicit_years)

    _, continuing_pv_point = _two_stage_value(nopat, g_point, roic, wacc_point, explicit_years)
    # by construction, g_point solves total_pv == enterprise_value
    continuing_share = continuing_pv_point / enterprise_value

    return TwoStageExpectations(
        symbol=symbol,
        implied_growth_at_wacc_low=g_low,
        implied_growth_at_wacc_point=g_point,
        implied_growth_at_wacc_high=g_high,
        continuing_value_share_of_total=continuing_share,
        terminal_value_dominant=continuing_share > TERMINAL_VALUE_DOMINANT_THRESHOLD,
        historical_revenue_cagr_3y=historical_revenue_cagr_3y,
        historical_revenue_cagr_5y=historical_revenue_cagr_5y,
        historical_nopat_cagr_3y=historical_nopat_cagr_3y,
        historical_nopat_cagr_5y=historical_nopat_cagr_5y,
        explicit_years=explicit_years,
    )


def _two_stage_value(
    nopat_0: float, g: float, roic: float, wacc: float, explicit_years: int
) -> tuple[float, float]:
    """Returns ``(total_pv, continuing_value_pv)`` for one growth-rate guess."""
    reinvestment_rate = g / roic
    nopat_t = nopat_0
    pv_stage1 = 0.0
    for t in range(1, explicit_years + 1):
        nopat_t = nopat_t * (1 + g)
        fcf_t = nopat_t * (1 - reinvestment_rate)
        pv_stage1 += fcf_t / (1 + wacc) ** t

    nopat_terminal = nopat_t * (1 + g)
    continuing_value_at_n = nopat_terminal / wacc
    pv_continuing = continuing_value_at_n / (1 + wacc) ** explicit_years

    return pv_stage1 + pv_continuing, pv_continuing


def _solve_growth(
    symbol: str,
    enterprise_value: float,
    nopat: float,
    roic: float,
    wacc: float,
    explicit_years: int,
) -> float:
    lower = GROWTH_SEARCH_FLOOR
    upper = roic - GROWTH_SEARCH_ROIC_MARGIN
    if upper <= lower:
        raise NoSolutionError(
            f"{symbol}: search range is empty (wacc={wacc}, roic={roic})"
        )

    def f(g: float) -> float:
        total_pv, _ = _two_stage_value(nopat, g, roic, wacc, explicit_years)
        return total_pv - enterprise_value

    f_lower, f_upper = f(lower), f(upper)
    if f_lower == 0:
        root = lower
    elif f_upper == 0:
        root = upper
    elif (f_lower > 0) == (f_upper > 0):
        raise NoSolutionError(
            f"{symbol}: no growth rate in [{lower}, {upper}] reproduces "
            f"enterprise_value={enterprise_value} at wacc={wacc}"
        )
    else:
        root = _bisect(f, lower, upper)

    if abs(roic - root) < UNSTABLE_BAND:
        raise UnstableSolutionError(
            f"{symbol}: solved growth {root:.4f} is within {UNSTABLE_BAND:.2f} of "
            f"roic={roic:.4f} -- implies reinvesting at/above 100% of NOPAT"
        )
    return root


def _bisect(f: Callable[[float], float], lower: float, upper: float) -> float:
    f_lower = f(lower)
    for _ in range(_BISECTION_MAX_ITERATIONS):
        mid = (lower + upper) / 2.0
        f_mid = f(mid)
        if abs(f_mid) < _BISECTION_TOLERANCE or (upper - lower) / 2.0 < _BISECTION_TOLERANCE:
            return mid
        if (f_mid > 0) == (f_lower > 0):
            lower, f_lower = mid, f_mid
        else:
            upper = mid
    return (lower + upper) / 2.0
