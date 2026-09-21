"""Position sizing and risk -- fully computable, no human gate.

Grimes's framing: risk management is the tool through which an edge is
applied. It is not itself an edge.
"""

from __future__ import annotations

import math
import random
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from nse_screener.gate import GateVerdict

#: Fraction of capital a single position is capped at, independent of
#: stop-loss risk -- concentration risk is not covered by risk_pct alone.
POSITION_VALUE_CAP_FRACTION: float = 0.20

#: A SPECULATION-labelled position (no Graham-style justification for
#: "thorough analysis" behind it) is permitted but must be sized smaller
#: -- the label has a consequence, or it is decoration.
SPECULATION_RISK_MULTIPLIER: float = 0.5

_GATE_MULTIPLIER: dict[str, float] = {"HEALTHY": 1.0, "NEUTRAL": 0.5, "HOSTILE": 0.0}

MAX_RISK_PCT: float = 0.02


class PositionPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry_price: float
    stop_price: float
    stop_distance: float
    atr_multiple: float
    capital: float
    risk_pct_requested: float
    risk_pct_effective: float
    risk_amount: float
    gate_multiplier: float
    speculation_multiplier: float
    shares: int
    position_value: float
    position_value_capped: bool
    permitted: bool
    reason: Optional[str]
    #: Reference levels for journaling (entry + N * stop_distance), NOT
    #: predictions of where the price will go.
    r_multiple_targets: dict[int, float]


class RiskParameterError(Exception):
    """Raised when ``risk_pct`` exceeds ``MAX_RISK_PCT`` -- fail fast rather
    than allow account-threatening sizing."""


def calculate_position(
    capital: float,
    risk_pct: float,
    entry_price: float,
    atr_14: float,
    gate_verdict: GateVerdict,
    graham_label: Literal["INVESTMENT", "SPECULATION"],
    atr_multiple: float = 2.0,
) -> PositionPlan:
    """Size a position from ATR-based risk, scaled by market regime and Graham label.

    ``stop_distance`` is always an ATR multiple, never a fixed
    percentage -- the tested failure mode this system explicitly avoids.
    Raises ``RiskParameterError`` if ``risk_pct`` exceeds 2%, before any
    scaling is applied.
    """
    if risk_pct > MAX_RISK_PCT:
        raise RiskParameterError(
            f"risk_pct {risk_pct:.2%} exceeds the {MAX_RISK_PCT:.0%} maximum"
        )
    if capital <= 0:
        raise ValueError("capital must be positive")
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if atr_14 <= 0:
        raise ValueError("atr_14 must be positive")
    if atr_multiple <= 0:
        raise ValueError("atr_multiple must be positive")

    stop_distance = atr_14 * atr_multiple
    stop_price = entry_price - stop_distance

    gate_multiplier = _GATE_MULTIPLIER[gate_verdict.verdict.value]
    speculation_multiplier = (
        SPECULATION_RISK_MULTIPLIER if graham_label == "SPECULATION" else 1.0
    )
    risk_pct_effective = risk_pct * gate_multiplier * speculation_multiplier

    risk_amount = capital * risk_pct_effective
    shares = math.floor(risk_amount / stop_distance)
    position_value = shares * entry_price

    cap_value = capital * POSITION_VALUE_CAP_FRACTION
    position_value_capped = position_value > cap_value
    if position_value_capped:
        shares = math.floor(cap_value / entry_price)
        position_value = shares * entry_price

    permitted = gate_verdict.verdict.value != "HOSTILE"
    reason = None if permitted else "market gate hostile"

    r_multiple_targets = {n: entry_price + stop_distance * n for n in (1, 2, 3)}

    return PositionPlan(
        entry_price=entry_price,
        stop_price=stop_price,
        stop_distance=stop_distance,
        atr_multiple=atr_multiple,
        capital=capital,
        risk_pct_requested=risk_pct,
        risk_pct_effective=risk_pct_effective,
        risk_amount=risk_amount,
        gate_multiplier=gate_multiplier,
        speculation_multiplier=speculation_multiplier,
        shares=shares,
        position_value=position_value,
        position_value_capped=position_value_capped,
        permitted=permitted,
        reason=reason,
        r_multiple_targets=r_multiple_targets,
    )


class TradeRecord(BaseModel):
    """One closed trade's realized outcome in R-multiples (risk units),
    e.g. the realized (exit - entry) / stop_distance for a long."""

    model_config = ConfigDict(frozen=True)

    r_multiple: float


class ExpectancyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    expectancy_r: float
    win_rate: float
    loss_rate: float
    avg_win_r: float
    avg_loss_r: float
    sample_size: int
    ci_low: float
    ci_high: float


class InsufficientSampleError(Exception):
    """Raised below the minimum trade count -- expectancy on small samples is noise."""

    def __init__(self, sample_size: int, minimum: int) -> None:
        self.sample_size = sample_size
        self.minimum = minimum
        super().__init__(f"only {sample_size} trades, need >= {minimum} for expectancy")


def expectancy(
    trades: list[TradeRecord],
    min_sample: int = 30,
    bootstrap_iterations: int = 10_000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> ExpectancyResult:
    """``(win_rate * avg_win) - (loss_rate * avg_loss)``, in R-multiples.

    Raises ``InsufficientSampleError`` below ``min_sample`` trades. The
    confidence interval is a percentile bootstrap over the trade sample
    -- ``seed`` is exposed for reproducible tests, left unset (system
    randomness) in normal use.
    """
    n = len(trades)
    if n < min_sample:
        raise InsufficientSampleError(n, min_sample)

    r_values = [t.r_multiple for t in trades]
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r <= 0]
    win_rate = len(wins) / n
    loss_rate = len(losses) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    expectancy_r = win_rate * avg_win - loss_rate * avg_loss

    rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(bootstrap_iterations):
        resample = [r_values[rng.randrange(n)] for _ in range(n)]
        bootstrap_means.append(sum(resample) / n)
    bootstrap_means.sort()

    alpha = 1 - confidence
    low_index = int((alpha / 2) * bootstrap_iterations)
    high_index = min(int((1 - alpha / 2) * bootstrap_iterations), bootstrap_iterations - 1)

    return ExpectancyResult(
        expectancy_r=expectancy_r,
        win_rate=win_rate,
        loss_rate=loss_rate,
        avg_win_r=avg_win,
        avg_loss_r=avg_loss,
        sample_size=n,
        ci_low=bootstrap_means[low_index],
        ci_high=bootstrap_means[high_index],
    )
