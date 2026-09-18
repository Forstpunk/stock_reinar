"""Piotroski F-Score.

Indian evidence: a one-point F-Score improvement associates with ~4.93%
higher one-year market-adjusted return; a Nifty-100 study (2007-2024)
found high-F-Score firms delivered stronger returns and cushioned
drawdowns, with ROA and accruals the dominant drivers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from nse_screener.data import FundamentalData


class FScoreResult(BaseModel):
    """Piotroski F-Score with the per-signal breakdown exposed.

    A 7 built from profitability signals is a different company from a 7
    built from leverage signals -- callers should look at the category
    subtotals, not just ``total_score``.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str

    roa_positive: bool
    cfo_positive: bool
    roa_improved: bool
    accruals_quality: bool

    leverage_decreased: bool
    current_ratio_improved: bool
    no_share_issuance: bool

    gross_margin_improved: bool
    asset_turnover_improved: bool

    profitability_score: int
    leverage_liquidity_score: int
    efficiency_score: int
    total_score: int


def piotroski_f_score(current: FundamentalData, prior: FundamentalData) -> FScoreResult:
    """Compute the nine-signal Piotroski F-Score from two annual periods."""
    if current.total_assets <= 0 or prior.total_assets <= 0:
        raise ValueError(f"{current.symbol}: total_assets must be positive in both periods")
    if current.current_liabilities <= 0 or prior.current_liabilities <= 0:
        raise ValueError(f"{current.symbol}: current_liabilities must be positive in both periods")
    if current.total_revenue <= 0 or prior.total_revenue <= 0:
        raise ValueError(f"{current.symbol}: total_revenue must be positive in both periods")

    roa_current = current.net_income / current.total_assets
    roa_prior = prior.net_income / prior.total_assets
    roa_positive = roa_current > 0
    cfo_positive = current.operating_cash_flow > 0
    roa_improved = roa_current > roa_prior
    accruals_quality = current.operating_cash_flow > current.net_income

    leverage_current = current.total_debt / current.total_assets
    leverage_prior = prior.total_debt / prior.total_assets
    leverage_decreased = leverage_current < leverage_prior

    current_ratio_now = current.current_assets / current.current_liabilities
    current_ratio_prior = prior.current_assets / prior.current_liabilities
    current_ratio_improved = current_ratio_now > current_ratio_prior

    no_share_issuance = current.shares_outstanding <= prior.shares_outstanding

    gross_margin_now = current.gross_profit / current.total_revenue
    gross_margin_prior = prior.gross_profit / prior.total_revenue
    gross_margin_improved = gross_margin_now > gross_margin_prior

    asset_turnover_now = current.total_revenue / current.total_assets
    asset_turnover_prior = prior.total_revenue / prior.total_assets
    asset_turnover_improved = asset_turnover_now > asset_turnover_prior

    profitability_score = sum([roa_positive, cfo_positive, roa_improved, accruals_quality])
    leverage_liquidity_score = sum([leverage_decreased, current_ratio_improved, no_share_issuance])
    efficiency_score = sum([gross_margin_improved, asset_turnover_improved])

    return FScoreResult(
        symbol=current.symbol,
        roa_positive=roa_positive,
        cfo_positive=cfo_positive,
        roa_improved=roa_improved,
        accruals_quality=accruals_quality,
        leverage_decreased=leverage_decreased,
        current_ratio_improved=current_ratio_improved,
        no_share_issuance=no_share_issuance,
        gross_margin_improved=gross_margin_improved,
        asset_turnover_improved=asset_turnover_improved,
        profitability_score=profitability_score,
        leverage_liquidity_score=leverage_liquidity_score,
        efficiency_score=efficiency_score,
        total_score=profitability_score + leverage_liquidity_score + efficiency_score,
    )
