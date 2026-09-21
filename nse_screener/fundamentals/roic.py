"""ROIC decomposition and economic profit.

McKinsey's key value driver: value comes from growth and ROIC relative
to cost of capital. Growth creates value only when ROIC exceeds WACC;
growth at an inadequate ROIC destroys value. The decomposition into
margin and turnover matters on its own: two firms with identical ROIC
built from margin vs. turns are different businesses with different
vulnerabilities.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from nse_screener.fundamentals.reorganize import ReorganizedStatements
from nse_screener.fundamentals.wacc import WACCResult

#: Tolerance, in percentage points, for the identity
#: ``roic == operating_margin * capital_turnover * (1 - tax_rate)``.
#: A violation beyond this means the reorganization feeding this
#: function is internally inconsistent -- not a rounding issue.
DECOMPOSITION_TOLERANCE_PP: float = 0.5


class ROICAnalysis(BaseModel):
    """ROIC, its margin/turnover decomposition, and value creation vs. WACC.

    Multi-year history of each component is out of scope for now -- see
    the historical-depth finding in the project notes: yfinance does not
    reliably supply more than ~4 consecutive annual periods, well short
    of what a meaningful multi-year trend needs.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    roic: float
    operating_margin: float
    capital_turnover: float
    economic_spread: float
    economic_profit: float
    value_creating: bool
    spread_at_wacc_low: float
    spread_at_wacc_high: float
    spread_sign_uncertain: bool


class DecompositionError(Exception):
    """Raised when roic != operating_margin * capital_turnover * (1 - tax_rate)."""

    def __init__(self, symbol: str, roic: float, decomposed: float, diff_pp: float) -> None:
        self.symbol = symbol
        self.roic = roic
        self.decomposed = decomposed
        self.diff_pp = diff_pp
        super().__init__(
            f"{symbol}: roic ({roic:.4f}) does not match "
            f"margin*turnover*(1-tax) ({decomposed:.4f}), off by {diff_pp:.2f}pp"
        )


def roic_analysis(r: ReorganizedStatements, wacc: WACCResult) -> ROICAnalysis:
    """Decompose ROIC and compare it to the WACC band.

    ``spread_sign_uncertain`` is set when the economic spread's sign
    flips between ``wacc.wacc_low`` and ``wacc.wacc_high`` -- a company
    whose value creation depends on which end of the WACC band you use
    has not demonstrated value creation. Raises ``DecompositionError`` if
    the margin*turnover identity fails to hold, which signals an
    inconsistency in the ``ReorganizedStatements`` passed in, not a
    normal-range computation.
    """
    if r.average_invested_capital == 0:
        raise ValueError(f"{r.symbol}: average_invested_capital is zero, ROIC undefined")
    if r.revenue <= 0:
        raise ValueError(f"{r.symbol}: revenue must be positive to compute margin/turnover")

    roic = r.nopat / r.average_invested_capital
    operating_margin = r.ebita / r.revenue
    capital_turnover = r.revenue / r.average_invested_capital

    decomposed = operating_margin * capital_turnover * (1 - r.effective_tax_rate)
    diff_pp = abs(roic - decomposed) * 100
    if diff_pp > DECOMPOSITION_TOLERANCE_PP:
        raise DecompositionError(r.symbol, roic, decomposed, diff_pp)

    economic_spread = roic - wacc.wacc_point
    economic_profit = economic_spread * r.average_invested_capital

    spread_at_wacc_low = roic - wacc.wacc_low
    spread_at_wacc_high = roic - wacc.wacc_high
    spread_sign_uncertain = (spread_at_wacc_low > 0) != (spread_at_wacc_high > 0)

    return ROICAnalysis(
        symbol=r.symbol,
        roic=roic,
        operating_margin=operating_margin,
        capital_turnover=capital_turnover,
        economic_spread=economic_spread,
        economic_profit=economic_profit,
        value_creating=economic_spread > 0,
        spread_at_wacc_low=spread_at_wacc_low,
        spread_at_wacc_high=spread_at_wacc_high,
        spread_sign_uncertain=spread_sign_uncertain,
    )
