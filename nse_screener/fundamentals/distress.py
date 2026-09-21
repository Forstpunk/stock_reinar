"""Distress scoring: Altman Z-Score, plus the liquidity/coverage family Tracy's
ratio taxonomy calls for and v1/v2 don't otherwise compute.

Rationale for running this alongside M-Score: in the Toshiba case, the
Beneish model failed to detect the manipulation while an Altman Z-score
did flag distress. Two models with different failure modes are better
than one -- this is not a replacement for ``workbench/mscore.py``, it is
a deliberately independent second opinion.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from nse_screener.fundamentals.peers import NotApplicableForSector, SectorInfo

#: Altman (1968), original coefficients for manufacturing/non-financial firms.
_W_X1: float = 1.2
_W_X2: float = 1.4
_W_X3: float = 3.3
_W_X4: float = 0.6
_W_X5: float = 1.0

SAFE_THRESHOLD: float = 2.99
DISTRESS_THRESHOLD: float = 1.81


class DistressInputs(BaseModel):
    """Line items for one period. Single-period -- the Z-score itself needs
    no history, though a trajectory would (out of scope, see roic.py's
    module docstring for why)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    working_capital: float
    total_assets: float
    retained_earnings: float
    ebit: float
    market_cap: float
    total_liabilities: float
    revenue: float
    interest_expense: float
    net_debt: float
    ebitda: float
    current_assets: float
    current_liabilities: float
    inventory: float
    cash_and_equivalents: float


class DistressResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    z_score: float
    verdict: Literal["SAFE", "GREY", "DISTRESS"]
    x1_working_capital_ratio: float
    x2_retained_earnings_ratio: float
    x3_ebit_ratio: float
    x4_equity_to_liabilities: float
    x5_asset_turnover: float
    interest_coverage: Optional[float]
    net_debt_to_ebitda: float
    current_ratio: float
    acid_test_ratio: float
    cash_ratio: float


def classify_z_band(z_score: float) -> Literal["SAFE", "GREY", "DISTRESS"]:
    """SAFE strictly above 2.99; GREY from 1.81 up to and including 2.99;
    DISTRESS strictly below 1.81 -- matches 'Z > 2.99 safe, 1.81-2.99 grey,
    < 1.81 distress' with the boundary values themselves falling in GREY."""
    if z_score > SAFE_THRESHOLD:
        return "SAFE"
    if z_score >= DISTRESS_THRESHOLD:
        return "GREY"
    return "DISTRESS"


def altman_z_score(f: DistressInputs, sector_info: SectorInfo) -> DistressResult:
    """Compute the Altman Z-Score plus a plain liquidity/coverage family.

    Raises ``NotApplicableForSector`` for financials -- the original
    model was not built for them (deposits as liabilities, provisioning,
    capital adequacy make the ratios meaningless), matching
    ``peers.check_applicable``'s ``altman_z_score`` entry.
    """
    if sector_info.requires_specialist_analysis:
        raise NotApplicableForSector(f.symbol, "altman_z_score", sector_info.sector)
    if f.total_assets <= 0:
        raise ValueError(f"{f.symbol}: total_assets must be positive")
    if f.total_liabilities <= 0:
        raise ValueError(f"{f.symbol}: total_liabilities must be positive")
    if f.current_liabilities <= 0:
        raise ValueError(f"{f.symbol}: current_liabilities must be positive")
    if f.ebitda == 0:
        raise ValueError(f"{f.symbol}: ebitda is zero, net_debt_to_ebitda undefined")

    x1 = f.working_capital / f.total_assets
    x2 = f.retained_earnings / f.total_assets
    x3 = f.ebit / f.total_assets
    x4 = f.market_cap / f.total_liabilities
    x5 = f.revenue / f.total_assets
    z = _W_X1 * x1 + _W_X2 * x2 + _W_X3 * x3 + _W_X4 * x4 + _W_X5 * x5

    interest_coverage = f.ebit / f.interest_expense if f.interest_expense > 0 else None

    return DistressResult(
        symbol=f.symbol,
        z_score=z,
        verdict=classify_z_band(z),
        x1_working_capital_ratio=x1,
        x2_retained_earnings_ratio=x2,
        x3_ebit_ratio=x3,
        x4_equity_to_liabilities=x4,
        x5_asset_turnover=x5,
        interest_coverage=interest_coverage,
        net_debt_to_ebitda=f.net_debt / f.ebitda,
        current_ratio=f.current_assets / f.current_liabilities,
        acid_test_ratio=(f.current_assets - f.inventory) / f.current_liabilities,
        cash_ratio=f.cash_and_equivalents / f.current_liabilities,
    )
