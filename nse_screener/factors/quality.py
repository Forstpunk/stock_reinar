"""Quality factors: gross profitability and return on equity.

Novy-Marx: gross profits-to-assets predicts the cross-section about as
strongly as book-to-market; the most profitable quintile beat the least
profitable by ~4%/year (1963-2010), replicated across 19 developed
markets and in emerging markets (~5.1%/yr long high-ROE / short
low-ROE). This research is US/developed-market derived and applied here
to India without local replication -- treat as a reasonable prior, not
a proven local edge.
"""

from __future__ import annotations

from nse_screener.data import FundamentalData


def gross_profitability(f: FundamentalData) -> float:
    """Gross profit divided by total assets, for the current period only."""
    if f.total_assets <= 0:
        raise ValueError(f"{f.symbol}: total_assets must be positive, got {f.total_assets}")
    return f.gross_profit / f.total_assets


def return_on_equity(f: FundamentalData) -> float:
    """Net income divided by average equity across the current and prior period.

    Requires ``f.prior`` to be set (as returned by ``data.fetch_fundamentals``).
    """
    if f.prior is None:
        raise ValueError(f"{f.symbol}: return_on_equity requires the prior period (f.prior is None)")
    avg_equity = (f.total_equity + f.prior.total_equity) / 2.0
    if avg_equity <= 0:
        raise ValueError(f"{f.symbol}: average equity must be positive, got {avg_equity}")
    return f.net_income / avg_equity
