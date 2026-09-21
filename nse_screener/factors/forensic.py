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

#: A balance-sheet line below this share of total assets cannot support a
#: meaningful year-on-year growth comparison: a few crore of movement on a
#: lakh-crore balance sheet produces a large percentage that means nothing.
#: (TCS carries ~0.02% of assets as inventory; TATASTEEL ~15%, where the
#: check is genuinely informative.)
INVENTORY_MATERIALITY_FRACTION: float = 0.02
RECEIVABLE_MATERIALITY_FRACTION: float = 0.02


class ForensicResult(BaseModel):
    """Forensic screen outcome for one symbol.

    Flags indicate a need for investigation, not proven manipulation.
    ``undefined_checks`` lists checks that could not be evaluated, either
    because a ratio was undefined (zero denominator: no debt, zero net
    income) or because the line item was too small or wrongly signed to
    carry meaning (inventory or receivables below a materiality floor,
    negative equity). They are reported for transparency only -- they are
    NOT evidence and never count towards disqualification.
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
    undefined_checks: list[str]
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
    """Run the five forensic flags and disqualify at ``disqualify_flag_count`` or more.

    Only genuine threshold breaches are recorded in ``flags`` and counted
    towards disqualification. A check whose ratio is undefined (zero
    denominator -- e.g. a debt-free or inventory-free business) is recorded
    in ``undefined_checks`` for transparency, but is an *inability to
    evaluate*, not evidence of a problem, and never contributes to the
    disqualification count.
    """
    if current.total_assets <= 0:
        raise ValueError(
            f"{current.symbol}: total_assets must be positive, got {current.total_assets}"
        )
    if prior.total_assets <= 0:
        raise ValueError(
            f"{prior.symbol}: prior total_assets must be positive, got {prior.total_assets}"
        )

    flags: list[str] = []
    undefined_checks: list[str] = []

    if current.net_income == 0:
        cfo_ni_ratio = math.nan
        undefined_checks.append("cfo_ni_ratio: net income is zero, ratio undefined")
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
    receivables_meaningful, receivables_reason = _comparison_is_meaningful(
        "dso_growth",
        "receivables",
        current.accounts_receivable,
        prior.accounts_receivable,
        current.total_assets,
        prior.total_assets,
        RECEIVABLE_MATERIALITY_FRACTION,
    )
    if not receivables_meaningful:
        undefined_checks.append(str(receivables_reason))
    elif math.isnan(dso_growth_pp) or math.isnan(revenue_growth_pp):
        undefined_checks.append(
            "dso_growth: revenue or receivables zero in a period, growth undefined"
        )
    elif dso_growth_pp - revenue_growth_pp > dso_vs_revenue_growth_pp:
        flags.append(
            f"dso_growth {dso_growth_pp:.1f}pp exceeds revenue_growth "
            f"{revenue_growth_pp:.1f}pp by more than {dso_vs_revenue_growth_pp:.1f}pp"
        )

    inventory_growth_pp = _growth_pp(current.inventory, prior.inventory)
    cogs_growth_pp = _growth_pp(current.cost_of_goods_sold, prior.cost_of_goods_sold)
    inventory_meaningful, inventory_reason = _comparison_is_meaningful(
        "inventory_growth",
        "inventory",
        current.inventory,
        prior.inventory,
        current.total_assets,
        prior.total_assets,
        INVENTORY_MATERIALITY_FRACTION,
    )
    if not inventory_meaningful:
        undefined_checks.append(str(inventory_reason))
    elif math.isnan(inventory_growth_pp) or math.isnan(cogs_growth_pp):
        undefined_checks.append(
            "inventory_growth: inventory or COGS zero in prior period, growth undefined"
        )
    elif inventory_growth_pp - cogs_growth_pp > inventory_vs_cogs_growth_pp:
        flags.append(
            f"inventory_growth {inventory_growth_pp:.1f}pp exceeds cogs_growth "
            f"{cogs_growth_pp:.1f}pp by more than {inventory_vs_cogs_growth_pp:.1f}pp"
        )

    current_de = _debt_to_equity(current.total_debt, current.total_equity)
    prior_de = _debt_to_equity(prior.total_debt, prior.total_equity)
    # Materiality fraction 0: equity is the ratio's denominator, so the only
    # question is its sign -- a period of negative equity is not comparable.
    equity_meaningful, equity_reason = _comparison_is_meaningful(
        "leverage_jump",
        "equity",
        current.total_equity,
        prior.total_equity,
        current.total_assets,
        prior.total_assets,
        materiality_fraction=0.0,
        require_positive=True,
    )
    if not equity_meaningful:
        leverage_jump_pct = math.nan
        undefined_checks.append(str(equity_reason))
    elif math.isnan(current_de) or math.isnan(prior_de) or prior_de == 0:
        leverage_jump_pct = math.nan
        undefined_checks.append(
            "leverage_jump: debt-to-equity undefined in current or prior period"
        )
    else:
        leverage_jump_pct = (current_de / prior_de - 1.0) * 100
        if current_de / prior_de - 1.0 > leverage_jump_fraction:
            flags.append(
                f"leverage_jump {leverage_jump_pct:.1f}% exceeds {leverage_jump_fraction:.0%}"
            )

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
        undefined_checks=undefined_checks,
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
    # Negative equity makes the ratio uninterpretable, not merely negative:
    # a D/E of -2.0 says nothing about leverage relative to a D/E of +0.5.
    if equity <= 0:
        return math.nan
    return debt / equity


def _comparison_is_meaningful(
    check: str,
    quantity: str,
    current_value: float,
    prior_value: float,
    current_scale: float,
    prior_scale: float,
    materiality_fraction: float,
    require_positive: bool = False,
) -> tuple[bool, str | None]:
    """Return (is_meaningful, reason_if_not) for a year-on-year comparison.

    A ratio built from inputs that are too small, or wrongly signed, to
    carry meaning is arithmetically valid and economically empty -- it must
    be reported as an inability to evaluate, never as a flag. Both periods
    are tested: a comparison is only as sound as its weaker input. With
    ``require_positive`` the sign is tested first, so a negative line is
    reported as such rather than as "below the materiality floor".
    """
    for period, value, scale in (
        ("current", current_value, current_scale),
        ("prior", prior_value, prior_scale),
    ):
        if require_positive and value <= 0:
            return False, (
                f"{check}: {period} period {quantity} {value:.1f} is not positive, "
                "comparison undefined"
            )
        share = value / scale
        if share < materiality_fraction:
            return False, (
                f"{check}: {period} period {quantity} is {share:.2%} of total assets, "
                f"below the {materiality_fraction:.0%} materiality floor -- immaterial, "
                "comparison skipped"
            )
    return True, None
