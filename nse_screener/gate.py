"""Market regime gate.

Identical stock-selection setups have shown materially different forward
edge depending on the broad market regime (+0.32pp in healthy regimes,
-0.86pp in hostile ones in the source backtests), and Siegel's long-run
200-day SMA work supports trend gating as a drawdown-reduction tool. This
module computes that regime and is a mandatory input to the report: when
the verdict is HOSTILE, the shortlist must be labelled a watchlist, never
a buy list.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
from pydantic import BaseModel, ConfigDict


class Verdict(str, Enum):
    HEALTHY = "HEALTHY"
    NEUTRAL = "NEUTRAL"
    HOSTILE = "HOSTILE"


class GateVerdict(BaseModel):
    """Result of one market-gate evaluation."""

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    index_close: float
    index_ma50: float
    index_ma200: float
    index_vs_50dma_pct: float
    index_vs_200dma_pct: float
    dma200_slope_21d_pct: float
    breadth_trend_aligned_pct: float
    breadth_above_ma200_pct: float
    breadth_near_52w_high_pct: float
    exposure_guidance: str


@dataclass(frozen=True)
class _Breadth:
    trend_aligned_pct: float
    above_ma200_pct: float
    near_52w_high_pct: float


def evaluate_market_gate(
    index_close: pd.Series,
    universe_closes: pd.DataFrame,
    ma_short: int = 50,
    ma_long: int = 200,
    slope_lookback: int = 21,
    breadth_high_proximity: float = 0.15,
) -> GateVerdict:
    """Evaluate the market regime from the index and breadth of the universe.

    ``index_close`` must be sorted-orderable and have at least
    ``ma_long + slope_lookback`` sessions. Verdict logic is exactly:

        HEALTHY  if index > MA200 and dma200_slope_21d > 0
        NEUTRAL  if index > MA200 and dma200_slope_21d <= 0
        HOSTILE  otherwise
    """
    index_close = index_close.sort_index()
    if len(index_close) < ma_long + slope_lookback:
        raise ValueError(
            f"index_close needs at least {ma_long + slope_lookback} sessions, "
            f"got {len(index_close)}"
        )

    ma50 = index_close.rolling(ma_short).mean()
    ma200 = index_close.rolling(ma_long).mean()

    last_close = float(index_close.iloc[-1])
    last_ma50 = float(ma50.iloc[-1])
    last_ma200 = float(ma200.iloc[-1])
    if pd.isna(last_ma50) or pd.isna(last_ma200):
        raise ValueError("insufficient history to compute moving averages")

    prior_ma200 = ma200.iloc[-1 - slope_lookback]
    if pd.isna(prior_ma200):
        raise ValueError("insufficient history to compute 200dma slope")
    slope = last_ma200 / float(prior_ma200) - 1.0

    index_vs_50 = last_close / last_ma50 - 1.0
    index_vs_200 = last_close / last_ma200 - 1.0

    if last_close > last_ma200 and slope > 0:
        verdict = Verdict.HEALTHY
        guidance = "Full position sizing per risk rules"
    elif last_close > last_ma200 and slope <= 0:
        verdict = Verdict.NEUTRAL
        guidance = "Half position sizing"
    else:
        verdict = Verdict.HOSTILE
        guidance = "Watchlist only -- no new positions indicated"

    breadth = _compute_breadth(
        universe_closes, ma_short, ma_long, slope_lookback, breadth_high_proximity
    )

    return GateVerdict(
        verdict=verdict,
        index_close=last_close,
        index_ma50=last_ma50,
        index_ma200=last_ma200,
        index_vs_50dma_pct=index_vs_50,
        index_vs_200dma_pct=index_vs_200,
        dma200_slope_21d_pct=slope,
        breadth_trend_aligned_pct=breadth.trend_aligned_pct,
        breadth_above_ma200_pct=breadth.above_ma200_pct,
        breadth_near_52w_high_pct=breadth.near_52w_high_pct,
        exposure_guidance=guidance,
    )


def _compute_breadth(
    universe_closes: pd.DataFrame,
    ma_short: int,
    ma_long: int,
    slope_lookback: int,
    high_proximity: float,
) -> _Breadth:
    if universe_closes.empty:
        raise ValueError("universe_closes must not be empty")

    trend_aligned = 0
    above_200 = 0
    near_high = 0
    counted = 0

    for symbol in universe_closes.columns:
        closes = universe_closes[symbol].dropna().sort_index()
        if len(closes) < ma_long + slope_lookback:
            continue
        counted += 1

        ma50 = closes.rolling(ma_short).mean()
        ma200 = closes.rolling(ma_long).mean()
        last_close = closes.iloc[-1]
        last_ma50 = ma50.iloc[-1]
        last_ma200 = ma200.iloc[-1]
        prior_ma200 = ma200.iloc[-1 - slope_lookback]

        if (
            last_close > last_ma50 > last_ma200
            and pd.notna(prior_ma200)
            and last_ma200 > prior_ma200
        ):
            trend_aligned += 1
        if last_close > last_ma200:
            above_200 += 1

        fifty_two_week_high = closes.tail(252).max()
        if last_close >= fifty_two_week_high * (1 - high_proximity):
            near_high += 1

    if counted == 0:
        raise ValueError(
            "no symbols in universe_closes had enough history for breadth calc"
        )

    return _Breadth(
        trend_aligned_pct=trend_aligned / counted,
        above_ma200_pct=above_200 / counted,
        near_52w_high_pct=near_high / counted,
    )
