"""Forensic accounting screen -- a disqualifier, not a score.

Based on the accrual anomaly: accruals are less persistent than cash
flow, and the market tends to overreact to earnings that are heavy on
accruals relative to cash. Note the literature also reports this anomaly
weakened after roughly 2002 -- treat it as a filter for names that
deserve closer investigation, not as proof of manipulation. A single
flag commonly has an innocent explanation (a one-off receivable, a
planned inventory build ahead of a launch); it is the accumulation of
flags that warrants exclusion from the shortlist.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from nse_screener.data import FundamentalData


class ForensicResult(BaseModel):
    """Forensic screen outcome for one symbol.

    Flags indicate a need for investigation, not proven manipulation.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    cfo_ni_ratio: float
    accrual_ratio: float
    dso_growth_pp: float
    revenue_growth_pp: float
    inventory_growth_pp: float
    cogs_growth_pp: float
    leverage_jump_pct: float
    flags: list[str]
    disqualified: bool


def forensic_screen(
    current: FundamentalData,
    prior: FundamentalData,
    cfo_ni_min: float = 0.8,
    accrual_ratio_max: float = 0.10,
    dso_vs_revenue_growth_pp: float = 15.0,
    inventory_vs_cogs_growth_pp: float = 20.0,
    leverage_jump_fraction: float = 0.50,
    disqualify_flag_count: int = 2,
) -> ForensicResult:
    """Run the five forensic flags and disqualify at ``disqualify_flag_count`` or more."""
    if current.total_assets <= 0:
        raise ValueError(f"{current.symbol}: total_assets must be positive, got {current.total_assets}")

    flags: list[str] = []

    if current.net_income == 0:
        cfo_ni_ratio = math.nan
        flags.append("cfo_ni_ratio: net income is zero, ratio undefined")
    else:
        cfo_ni_ratio = current.operating_cash_flow / current.net_income
        if cfo_ni_ratio < cfo_ni_min:
            flags.append(f"cfo_ni_ratio {cfo_ni_ratio:.2f} < {cfo_ni_min:.2f}")

    accrual_ratio = (current.net_income - current.operating_cash_flow) / current.total_assets
    if accrual_ratio > accrual_ratio_max:
        flags.append(f"accrual_ratio {accrual_ratio:.2f} > {accrual_ratio_max:.2f}")

    current_dso = _days_outstanding(current.accounts_receivable, current.total_revenue)
    prior_dso = _days_outstanding(prior.accounts_receivable, prior.total_revenue)
    dso_growth_pp = _growth_pp(current_dso, prior_dso)
    revenue_growth_pp = _growth_pp(current.total_revenue, prior.total_revenue)
    if math.isnan(dso_growth_pp) or dso_growth_pp - revenue_growth_pp > dso_vs_revenue_growth_pp:
        flags.append(
            f"dso_growth {dso_growth_pp:.1f}pp exceeds revenue_growth "
            f"{revenue_growth_pp:.1f}pp by more than {dso_vs_revenue_growth_pp:.1f}pp"
        )

    inventory_growth_pp = _growth_pp(current.inventory, prior.inventory)
    cogs_growth_pp = _growth_pp(current.cost_of_goods_sold, prior.cost_of_goods_sold)
    if (
        math.isnan(inventory_growth_pp)
        or inventory_growth_pp - cogs_growth_pp > inventory_vs_cogs_growth_pp
    ):
        flags.append(
            f"inventory_growth {inventory_growth_pp:.1f}pp exceeds cogs_growth "
            f"{cogs_growth_pp:.1f}pp by more than {inventory_vs_cogs_growth_pp:.1f}pp"
        )

    current_de = _debt_to_equity(current.total_debt, current.total_equity)
    prior_de = _debt_to_equity(prior.total_debt, prior.total_equity)
    if math.isnan(current_de) or math.isnan(prior_de) or prior_de == 0:
        leverage_jump_pct = math.nan
        flags.append("leverage_jump: debt-to-equity undefined in current or prior period")
    else:
        leverage_jump_pct = (current_de / prior_de - 1.0) * 100
        if current_de / prior_de - 1.0 > leverage_jump_fraction:
            flags.append(f"leverage_jump {leverage_jump_pct:.1f}% exceeds {leverage_jump_fraction:.0%}")

    disqualified = len(flags) >= disqualify_flag_count

    return ForensicResult(
        symbol=current.symbol,
        cfo_ni_ratio=cfo_ni_ratio,
        accrual_ratio=accrual_ratio,
        dso_growth_pp=dso_growth_pp,
        revenue_growth_pp=revenue_growth_pp,
        inventory_growth_pp=inventory_growth_pp,
        cogs_growth_pp=cogs_growth_pp,
        leverage_jump_pct=leverage_jump_pct,
        flags=flags,
        disqualified=disqualified,
    )


def _days_outstanding(receivable: float, revenue: float) -> float:
    if revenue == 0:
        return math.nan
    return receivable / revenue * 365.0


def _growth_pp(current: float, prior: float) -> float:
    if prior == 0:
        return math.nan
    return (current / prior - 1.0) * 100


def _debt_to_equity(debt: float, equity: float) -> float:
    if equity == 0:
        return math.nan
    return debt / equity
