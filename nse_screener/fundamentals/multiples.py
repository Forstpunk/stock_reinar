"""Valuation multiples -- currently absent entirely from v1/v2.

Every multiple here is meant to be read two ways: against the sector
(via ``fundamentals.peers.peer_statistics``) and against the subject's
own history. "Cheap versus peers" and "cheap versus its own past" are
different claims -- this module computes the numbers, it does not blend
them into one verdict.

``total_payout_yield`` closes the Williams gap: dividend-discount
reasoning structurally undervalues modern firms because buybacks now
rival dividends as the payout channel. Total payout is the correct
modern reading of "what this will actually pay me".

Own-history (5-year) comparison is out of scope for the same reason
``roic.py``'s and ``distress.py``'s multi-year features are: yfinance
does not reliably supply more than ~4 consecutive annual periods.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from nse_screener.fundamentals.peers import PeerStats
from nse_screener.fundamentals.reorganize import ReorganizedStatements


class MultiplesInputs(BaseModel):
    """Line items ``compute_multiples`` needs beyond ``ReorganizedStatements``."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    net_income: float
    book_value: float
    ebitda: float
    ebit: float
    revenue: float
    dividends_paid: float
    net_buybacks: float
    shares_outstanding: float


class Multiples(BaseModel):
    """Every multiple that can be undefined is paired with a ``*_reason`` field,
    populated only when the value is ``None`` -- never a negative multiple
    standing in for "undefined"."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    enterprise_value: float
    market_cap: float

    pe: Optional[float]
    pe_reason: Optional[str]
    pb: Optional[float]
    pb_reason: Optional[str]
    ev_ebitda: Optional[float]
    ev_ebitda_reason: Optional[str]
    ev_ebit: Optional[float]
    ev_ebit_reason: Optional[str]
    ev_sales: Optional[float]
    ev_sales_reason: Optional[str]
    ev_invested_capital: Optional[float]
    ev_invested_capital_reason: Optional[str]

    fcf_yield: float
    dividend_yield: float
    payout_ratio: Optional[float]
    payout_ratio_reason: Optional[str]
    total_payout_yield: float


class PremiumDecomposition(BaseModel):
    """The premium/discount to sector on one multiple, alongside the
    fundamentals that might (or might not) justify it. Presentation only
    -- this never concludes justified/unjustified, that is a judgment call
    for Palepu step 4."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    multiple_name: str
    subject_value: float
    sector_median: float
    premium_to_sector_median_pct: float
    roic_percentile_within_sector: float
    operating_margin_percentile_within_sector: float


def compute_multiples(
    inputs: MultiplesInputs,
    r: ReorganizedStatements,
    market_cap: float,
    net_debt: float,
) -> Multiples:
    """Compute every multiple, returning ``None`` with a stated reason where undefined.

    ``fcf_yield`` and ``total_payout_yield`` can legitimately be negative
    or low (a cash-burning business, a non-payer) -- that is a real
    signal, not an undefined case, so they are always computed rather
    than nulled out.
    """
    if market_cap <= 0:
        raise ValueError(f"{inputs.symbol}: market_cap must be positive")

    enterprise_value = market_cap + net_debt

    pe, pe_reason = _ratio_or_reason(market_cap, inputs.net_income, "net income is not positive")
    pb, pb_reason = _ratio_or_reason(market_cap, inputs.book_value, "book value is not positive")
    ev_ebitda, ev_ebitda_reason = _ratio_or_reason(
        enterprise_value, inputs.ebitda, "EBITDA is not positive"
    )
    ev_ebit, ev_ebit_reason = _ratio_or_reason(
        enterprise_value, inputs.ebit, "EBIT is not positive"
    )
    ev_sales, ev_sales_reason = _ratio_or_reason(
        enterprise_value, inputs.revenue, "revenue is not positive"
    )
    ev_ic, ev_ic_reason = _ratio_or_reason(
        enterprise_value, r.invested_capital, "invested capital is not positive"
    )
    payout_ratio, payout_reason = _ratio_or_reason(
        inputs.dividends_paid, inputs.net_income, "net income is not positive"
    )

    return Multiples(
        symbol=inputs.symbol,
        enterprise_value=enterprise_value,
        market_cap=market_cap,
        pe=pe,
        pe_reason=pe_reason,
        pb=pb,
        pb_reason=pb_reason,
        ev_ebitda=ev_ebitda,
        ev_ebitda_reason=ev_ebitda_reason,
        ev_ebit=ev_ebit,
        ev_ebit_reason=ev_ebit_reason,
        ev_sales=ev_sales,
        ev_sales_reason=ev_sales_reason,
        ev_invested_capital=ev_ic,
        ev_invested_capital_reason=ev_ic_reason,
        fcf_yield=r.free_cash_flow / market_cap,
        dividend_yield=inputs.dividends_paid / market_cap,
        payout_ratio=payout_ratio,
        payout_ratio_reason=payout_reason,
        total_payout_yield=(inputs.dividends_paid + inputs.net_buybacks) / market_cap,
    )


def _ratio_or_reason(
    numerator: float, denominator: float, reason_if_non_positive: str
) -> tuple[Optional[float], Optional[str]]:
    if denominator <= 0:
        return None, reason_if_non_positive
    return numerator / denominator, None


def premium_decomposition(
    multiple_name: str,
    multiple_stats: PeerStats,
    roic_stats: PeerStats,
    operating_margin_stats: PeerStats,
) -> PremiumDecomposition:
    """Join a multiple's sector premium with the ROIC/margin percentiles that
    might explain it. All three ``PeerStats`` must be for the same symbol."""
    symbols = {multiple_stats.symbol, roic_stats.symbol, operating_margin_stats.symbol}
    if len(symbols) != 1:
        raise ValueError(f"mismatched symbols across peer stats: {symbols}")
    if multiple_stats.sector_median == 0:
        raise ValueError(f"{multiple_stats.symbol}: sector median is zero, premium undefined")

    premium_pct = (multiple_stats.metric_value / multiple_stats.sector_median - 1.0) * 100

    return PremiumDecomposition(
        symbol=multiple_stats.symbol,
        multiple_name=multiple_name,
        subject_value=multiple_stats.metric_value,
        sector_median=multiple_stats.sector_median,
        premium_to_sector_median_pct=premium_pct,
        roic_percentile_within_sector=roic_stats.percentile_within_sector,
        operating_margin_percentile_within_sector=operating_margin_stats.percentile_within_sector,
    )
