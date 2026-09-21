"""Reorganized financial statements -- the prerequisite for honest ROIC.

McKinsey's core insight: reported statements mix operating and
non-operating items, so raw figures produce misleading returns. This
module separates them before anything downstream (``wacc.py``,
``roic.py``, ``multiples.py``, v2's ``valuation.py``) computes a return
or a value off the result.

``RawOperatingFinancials`` is this module's own input model, deliberately
separate from ``data.FundamentalData``: it needs line items (operating
income, PPE, tax detail, a cash/current-liability split) that
``FundamentalData`` does not carry. Wiring a live yfinance fetch for
these fields is a distinct, separate task -- this module only defines
what it needs and computes correctly once it has it.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict

#: The 2%-of-revenue operating-cash threshold is a convention (a common
#: rule of thumb for what a business needs on hand to run operations),
#: not a measured fact about any specific company. Cash held above this
#: is treated as a non-operating asset (excess cash) and excluded from
#: operating working capital.
EXCESS_CASH_REVENUE_FRACTION: float = 0.02

#: Effective tax rates outside this band usually reflect one-off items
#: (a tax credit, a settlement, a prior-year true-up) rather than the
#: company's normal run-rate tax burden, which would make NOPAT
#: unreliable if used unadjusted. The rate is clamped into this band and
#: the clamp is flagged, never applied silently.
TAX_RATE_MIN: float = 0.10
TAX_RATE_MAX: float = 0.50


class RawOperatingFinancials(BaseModel):
    """Line items ``reorganize()`` needs for one annual period.

    ``prior`` carries the immediately preceding period (with its own
    ``prior`` left unset), the same two-period pattern as
    ``data.FundamentalData``.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    period_end: date
    revenue: float
    operating_income: float
    amortization_of_intangibles: float
    tax_expense: float
    pretax_income: float
    current_assets: float
    current_liabilities: float
    cash_and_equivalents: float
    short_term_debt: float
    net_ppe: float
    net_other_operating_assets: float
    prior: Optional["RawOperatingFinancials"] = None


RawOperatingFinancials.model_rebuild()


class ReorganizedStatements(BaseModel):
    """Operating-only view of one period, plus the two-period averages ROIC needs."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    period_end: date
    revenue: float
    ebita: float
    effective_tax_rate: float
    tax_rate_clamped: bool
    operating_tax: float
    nopat: float
    operating_working_capital: float
    excess_cash: float
    invested_capital: float
    average_invested_capital: float
    increase_in_invested_capital: float
    free_cash_flow: float


class ReorganizationError(Exception):
    """Raised when ``reorganize()`` cannot compute a reliable result from its input."""

    def __init__(self, symbol: str, reason: str) -> None:
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"{symbol}: {reason}")


def reorganize(f: RawOperatingFinancials) -> ReorganizedStatements:
    """Separate operating from non-operating items and compute NOPAT/invested capital.

    Requires ``f.prior`` (average invested capital and its period-over-
    period increase both need two periods) and a positive
    ``pretax_income`` (the effective tax rate is undefined otherwise) --
    raises ``ReorganizationError`` naming the reason for either failure.
    Never substitutes a default for a value it cannot compute.
    """
    if f.prior is None:
        raise ReorganizationError(
            f.symbol, "prior period required to compute average invested capital"
        )
    if f.pretax_income <= 0:
        raise ReorganizationError(
            f.symbol,
            f"pretax_income must be positive to derive an effective tax rate, "
            f"got {f.pretax_income}",
        )

    raw_tax_rate = f.tax_expense / f.pretax_income
    tax_rate_clamped = not (TAX_RATE_MIN <= raw_tax_rate <= TAX_RATE_MAX)
    effective_tax_rate = min(max(raw_tax_rate, TAX_RATE_MIN), TAX_RATE_MAX)

    ebita = f.operating_income + f.amortization_of_intangibles
    operating_tax = ebita * effective_tax_rate
    nopat = ebita - operating_tax

    owc, excess_cash = _operating_working_capital(f)
    current_ic = owc + f.net_ppe + f.net_other_operating_assets
    prior_ic = _invested_capital(f.prior)
    average_ic = (current_ic + prior_ic) / 2.0
    increase_ic = current_ic - prior_ic

    free_cash_flow = nopat - increase_ic

    return ReorganizedStatements(
        symbol=f.symbol,
        period_end=f.period_end,
        revenue=f.revenue,
        ebita=ebita,
        effective_tax_rate=effective_tax_rate,
        tax_rate_clamped=tax_rate_clamped,
        operating_tax=operating_tax,
        nopat=nopat,
        operating_working_capital=owc,
        excess_cash=excess_cash,
        invested_capital=current_ic,
        average_invested_capital=average_ic,
        increase_in_invested_capital=increase_ic,
        free_cash_flow=free_cash_flow,
    )


def _operating_working_capital(f: RawOperatingFinancials) -> tuple[float, float]:
    """Returns ``(operating_working_capital, excess_cash)`` for one period."""
    excess_cash = max(0.0, f.cash_and_equivalents - EXCESS_CASH_REVENUE_FRACTION * f.revenue)
    owc = (f.current_assets - excess_cash) - (f.current_liabilities - f.short_term_debt)
    return owc, excess_cash


def _invested_capital(f: RawOperatingFinancials) -> float:
    owc, _ = _operating_working_capital(f)
    return owc + f.net_ppe + f.net_other_operating_assets
