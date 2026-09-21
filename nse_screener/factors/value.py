"""Value (cheapness) factors: earnings yield and book-to-price.

The classic "is this stock cheap relative to its own fundamentals"
factor -- Basu (1977) documented the P/E anomaly; Fama & French
(1992, 1993) established book-to-market as a primary cross-sectional
factor alongside size, later formalised as one leg of their three- and
five-factor models. Decades of replication across markets make this one
of the best-evidenced factors in equity research, on par with momentum
and profitability.

Expressed here as yields (earnings/price, book/price) rather than
ratios (P/E, P/B) deliberately: a yield stays well-defined and orders
sensibly for loss-making companies (a negative earnings yield is a
legible signal -- "expensive relative to negative earnings" -- whereas
a negative P/E inverts the usual "lower is cheaper" reading and is a
common source of screening bugs).

This research is US/developed-market derived and applied here to India
without local replication -- treat as a reasonable prior, not a proven
local edge, same caveat as the quality factors in ``quality.py``.
"""

from __future__ import annotations

from nse_screener.data import FundamentalData


def earnings_yield(f: FundamentalData, close_price: float) -> float:
    """Net income divided by market capitalisation (inverse of P/E).

    ``close_price`` is the current (or as-of) share price -- not stored
    on ``FundamentalData``, since it is a live market quantity, not a
    fundamental one.
    """
    market_cap = _market_cap(f, close_price)
    return f.net_income / market_cap


def book_to_price(f: FundamentalData, close_price: float) -> float:
    """Total equity divided by market capitalisation (inverse of P/B)."""
    market_cap = _market_cap(f, close_price)
    return f.total_equity / market_cap


def _market_cap(f: FundamentalData, close_price: float) -> float:
    if close_price <= 0:
        raise ValueError(f"{f.symbol}: close_price must be positive, got {close_price}")
    market_cap = close_price * f.shares_outstanding
    if market_cap <= 0:
        raise ValueError(f"{f.symbol}: market cap must be positive, got {market_cap}")
    return market_cap
