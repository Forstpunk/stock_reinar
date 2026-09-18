"""Relative strength (momentum) -- the primary factor.

Momentum is the best-evidenced cross-sectional equity factor available:
documented across the US, Europe, Japan and India since Jegadeesh &
Titman (1993). In-sample NSE measurement backing this module: the top
relative-strength decile forward-returned +2.40% vs +1.44% for the
mid-decile over the next 20 sessions, monotonic across deciles,
n=457,160 observations.
"""

from __future__ import annotations

import pandas as pd


def weighted_relative_strength(
    closes: pd.DataFrame,
    quarter_sessions: int = 63,
    weights: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
    min_sessions: int = 252,
) -> pd.Series:
    """Quarterly-weighted relative strength, one raw score per symbol.

    The percentile rank of this score is meaningful only relative to the
    other symbols supplied in the same ``closes`` frame -- it is NOT an
    absolute market rating, and percentile ranking is deliberately left
    to ``ranking.py`` so it can be computed after the liquidity filter is
    applied. Momentum returns are also negatively skewed and prone to
    occasional severe crashes; a high score here is not a smooth ride.

    ``closes`` must have one column per symbol, indexed by date, with no
    gaps left unhandled by the caller (see ``data.validate_price_data``).
    ``min_sessions`` is a floor only: the formula itself needs
    ``4 * quarter_sessions + 1`` sessions (the oldest term looks back four
    full quarters), so the effective requirement is the larger of the two.
    Raises ``ValueError`` naming the symbol if it has fewer sessions than
    that, or if the score cannot be computed without NaN propagation.
    """
    if weights[0] < weights[1] or weights[0] < weights[2] or weights[0] < weights[3]:
        raise ValueError("the most recent quarter must be weighted at least as heavily as the others")

    w1, w2, w3, w4 = weights
    required = max(min_sessions, 4 * quarter_sessions + 1)
    scores: dict[str, float] = {}

    for symbol in closes.columns:
        series = closes[symbol].dropna().sort_index()
        if len(series) < required:
            raise ValueError(
                f"{symbol}: requires at least {required} sessions of "
                f"history for relative strength, got {len(series)}"
            )

        c = series
        r1 = c / c.shift(quarter_sessions) - 1.0
        r2 = c.shift(quarter_sessions) / c.shift(2 * quarter_sessions) - 1.0
        r3 = c.shift(2 * quarter_sessions) / c.shift(3 * quarter_sessions) - 1.0
        r4 = c.shift(3 * quarter_sessions) / c.shift(4 * quarter_sessions) - 1.0
        rs = w1 * r1 + w2 * r2 + w3 * r3 + w4 * r4

        score = rs.iloc[-1]
        if pd.isna(score):
            raise ValueError(
                f"{symbol}: insufficient history to compute a non-NaN "
                f"relative-strength score (need more than "
                f"{4 * quarter_sessions} sessions)"
            )
        scores[symbol] = float(score)

    return pd.Series(scores, name="weighted_rs")
