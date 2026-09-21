"""Cost of capital -- removes the most fragile manual input in v2.

v2's reverse-DCF originally required the analyst to type in a WACC by
hand. This module computes it from real beta and a stated risk-free
rate / equity risk premium instead, and -- critically -- returns a band,
not a point estimate.

WACC is an estimate with a wide error band. Beta is unstable, ERP is
contested, and a +-1pp WACC error moves a perpetuity valuation
substantially. Downstream callers should propagate ``wacc_low``/
``wacc_high`` into valuation rather than using ``wacc_point`` alone.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

#: Blume (1975): raw betas mean-revert toward 1.0 over time; a raw
#: regression beta is a biased predictor of *future* beta on its own.
BLUME_WEIGHT_RAW: float = 0.67
BLUME_WEIGHT_MARKET: float = 0.33

#: ~80 weekly observations is roughly 1.5 years -- short of the
#: recommended 2-year window but the floor below which a beta estimate
#: is treated as too noisy to use at all.
MIN_WEEKLY_OBSERVATIONS: int = 80

BETA_SENSITIVITY: float = 0.2
ERP_SENSITIVITY: float = 0.01


class WACCInputs(BaseModel):
    """Raw items ``compute_wacc`` needs beyond price history and its rate parameters.

    ``effective_tax_rate`` is expected to come from
    ``reorganize.ReorganizedStatements`` for the same period, not a raw
    fetch -- WACC should use the same operating tax rate ROIC does.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    market_cap: float
    total_debt: float
    prior_total_debt: float
    interest_expense: float
    effective_tax_rate: float


class WACCResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    raw_beta: float
    adjusted_beta: float
    cost_of_equity: float
    cost_of_debt: float
    after_tax_cost_of_debt: float
    wacc_low: float
    wacc_point: float
    wacc_high: float
    unlevered: bool
    weekly_observations: int
    risk_free_rate: float
    equity_risk_premium: float


class InsufficientReturnHistoryError(Exception):
    """Raised when fewer than ``MIN_WEEKLY_OBSERVATIONS`` aligned weekly returns exist."""

    def __init__(self, symbol: str, observations: int, minimum: int) -> None:
        self.symbol = symbol
        self.observations = observations
        self.minimum = minimum
        super().__init__(
            f"{symbol}: only {observations} weekly return observations, need >= {minimum}"
        )


def compute_wacc(
    inputs: WACCInputs,
    price_history: pd.Series,
    index_history: pd.Series,
    risk_free_rate: float,
    equity_risk_premium: float,
) -> WACCResult:
    """Compute WACC from real beta, with a sensitivity band around the point estimate.

    ``price_history``/``index_history`` should cover roughly 2 years of
    daily or weekly closes -- resampled to weekly internally, since daily
    returns on Indian mid-caps are noisy and induce downward beta bias
    from non-synchronous trading. ``risk_free_rate`` and
    ``equity_risk_premium`` are required with no defaults: sourcing them
    is the analyst's decision, not this function's.

    Raises ``InsufficientReturnHistoryError`` below
    ``MIN_WEEKLY_OBSERVATIONS`` aligned weekly observations, or
    ``ValueError`` if the index has zero return variance (beta
    undefined). If ``total_debt`` is zero, WACC equals the cost of
    equity and ``unlevered=True``.
    """
    weekly_stock = _weekly_returns(price_history)
    weekly_index = _weekly_returns(index_history)
    aligned = pd.DataFrame({"stock": weekly_stock, "index": weekly_index}).dropna()

    if len(aligned) < MIN_WEEKLY_OBSERVATIONS:
        raise InsufficientReturnHistoryError(
            inputs.symbol, len(aligned), MIN_WEEKLY_OBSERVATIONS
        )

    variance = float(aligned["index"].var())
    if variance <= 0:
        raise ValueError(f"{inputs.symbol}: index return variance is zero, beta is undefined")
    raw_beta = float(aligned["stock"].cov(aligned["index"])) / variance
    adjusted_beta = BLUME_WEIGHT_RAW * raw_beta + BLUME_WEIGHT_MARKET * 1.0

    average_total_debt = (inputs.total_debt + inputs.prior_total_debt) / 2.0
    if average_total_debt <= 0:
        cost_of_debt = 0.0
        after_tax_kd = 0.0
    else:
        cost_of_debt = inputs.interest_expense / average_total_debt
        after_tax_kd = cost_of_debt * (1 - inputs.effective_tax_rate)

    def wacc_at(beta: float, erp: float) -> float:
        cost_of_equity = risk_free_rate + beta * erp
        if inputs.total_debt <= 0:
            return cost_of_equity
        equity, debt = inputs.market_cap, inputs.total_debt
        return (equity / (debt + equity)) * cost_of_equity + (
            debt / (debt + equity)
        ) * after_tax_kd

    cost_of_equity_point = risk_free_rate + adjusted_beta * equity_risk_premium
    wacc_point = wacc_at(adjusted_beta, equity_risk_premium)
    band = (
        wacc_at(adjusted_beta - BETA_SENSITIVITY, equity_risk_premium - ERP_SENSITIVITY),
        wacc_at(adjusted_beta + BETA_SENSITIVITY, equity_risk_premium + ERP_SENSITIVITY),
    )

    return WACCResult(
        symbol=inputs.symbol,
        raw_beta=raw_beta,
        adjusted_beta=adjusted_beta,
        cost_of_equity=cost_of_equity_point,
        cost_of_debt=cost_of_debt,
        after_tax_cost_of_debt=after_tax_kd,
        wacc_low=min(band),
        wacc_point=wacc_point,
        wacc_high=max(band),
        unlevered=inputs.total_debt <= 0,
        weekly_observations=len(aligned),
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
    )


def _weekly_returns(price_history: pd.Series) -> pd.Series:
    weekly_close = price_history.sort_index().resample("W").last().dropna()
    return weekly_close.pct_change().dropna()
