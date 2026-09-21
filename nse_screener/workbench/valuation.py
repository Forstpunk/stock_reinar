"""Margin of safety / value as a range.

Graham's discipline: intrinsic value is a range, precise only enough to
identify a clear discrepancy. This module never produces a point
estimate -- if asked for one, the answer is the range.

Print alongside every range: "A value range is only as good as its
inputs. Wide ranges mean low confidence, not a wide opportunity."
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

#: Growth assumptions are judgment, not something this module generates.
MIN_SCENARIOS: int = 3

VALUE_RANGE_CAVEAT: str = (
    "A value range is only as good as its inputs. Wide ranges mean low "
    "confidence, not a wide opportunity."
)


class ValueRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    per_share_values: dict[str, float]
    low: float
    high: float
    current_price: float
    price_position: Literal["BELOW_RANGE", "INSIDE_RANGE", "ABOVE_RANGE"]
    discount_to_low: Optional[float]
    #: HUMAN INPUT REQUIRED. Per Graham's own definition, an investment
    #: operation promises safety of principal and a satisfactory return
    #: *on thorough analysis* -- this tool cannot certify that happened.
    #: If the analyst cannot write the justification, the position is a
    #: SPECULATION (permitted, but sized accordingly by sizing.py).
    graham_label: Optional[Literal["INVESTMENT", "SPECULATION"]] = None
    label_justification: Optional[str] = None


def value_range(
    symbol: str,
    nopat: float,
    roic: float,
    wacc: float,
    scenarios: dict[str, float],
    net_debt: float,
    shares: float,
    current_price: float,
) -> ValueRange:
    """Compute a per-share value for each analyst-supplied growth scenario.

    ``scenarios`` maps a name to a growth rate and must have at least
    ``MIN_SCENARIOS`` entries (conservative/base/optimistic, or similar)
    -- growth assumptions are never generated automatically, that would
    defeat the purpose of asking for judgment here. ``price_position`` is
    mechanical; ``graham_label``/``label_justification`` are not and are
    left ``None`` pending the analyst.
    """
    if len(scenarios) < MIN_SCENARIOS:
        raise ValueError(
            f"{symbol}: need at least {MIN_SCENARIOS} scenarios, got {len(scenarios)}"
        )
    if shares <= 0:
        raise ValueError(f"{symbol}: shares must be positive")
    if roic <= 0:
        raise ValueError(f"{symbol}: roic must be positive")

    per_share_values: dict[str, float] = {}
    for name, g in scenarios.items():
        if g >= roic:
            raise ValueError(f"{symbol}: scenario {name!r} growth {g} >= roic {roic}")
        if g >= wacc:
            raise ValueError(f"{symbol}: scenario {name!r} growth {g} >= wacc {wacc}")
        enterprise_value = nopat * (1 - g / roic) / (wacc - g)
        equity_value = enterprise_value - net_debt
        per_share_values[name] = equity_value / shares

    low = min(per_share_values.values())
    high = max(per_share_values.values())

    if current_price < low:
        price_position: Literal["BELOW_RANGE", "INSIDE_RANGE", "ABOVE_RANGE"] = "BELOW_RANGE"
        discount_to_low: Optional[float] = (low - current_price) / low
    elif current_price > high:
        price_position = "ABOVE_RANGE"
        discount_to_low = None
    else:
        price_position = "INSIDE_RANGE"
        discount_to_low = None

    return ValueRange(
        symbol=symbol,
        per_share_values=per_share_values,
        low=low,
        high=high,
        current_price=current_price,
        price_position=price_position,
        discount_to_low=discount_to_low,
    )


def apply_graham_label(
    range_result: ValueRange,
    label: Literal["INVESTMENT", "SPECULATION"],
    justification: Optional[str] = None,
) -> ValueRange:
    """Attach the analyst's Graham label. An ``INVESTMENT`` label requires a
    non-empty ``justification`` -- if the analyst cannot write one, the
    label must be ``SPECULATION`` instead, not an unjustified INVESTMENT."""
    if label == "INVESTMENT" and not (justification and justification.strip()):
        raise ValueError(
            "an INVESTMENT label requires a written justification; "
            "use SPECULATION if none can be given"
        )
    return range_result.model_copy(
        update={"graham_label": label, "label_justification": justification}
    )
