from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_screener.factors.momentum import weighted_relative_strength


def _price_path(session_count: int, daily_return: float, start: float = 100.0) -> pd.Series:
    dates = pd.bdate_range("2020-01-01", periods=session_count)
    prices = start * (1 + daily_return) ** np.arange(session_count)
    return pd.Series(prices, index=dates)


def test_weighted_relative_strength_matches_formula():
    session_count = 260
    closes = pd.DataFrame({"AAA": _price_path(session_count, 0.001)})

    result = weighted_relative_strength(closes)

    c = closes["AAA"]
    r1 = c.iloc[-1] / c.iloc[-1 - 63] - 1.0
    r2 = c.iloc[-1 - 63] / c.iloc[-1 - 126] - 1.0
    r3 = c.iloc[-1 - 126] / c.iloc[-1 - 189] - 1.0
    r4 = c.iloc[-1 - 189] / c.iloc[-1 - 252] - 1.0
    expected = 0.4 * r1 + 0.2 * r2 + 0.2 * r3 + 0.2 * r4

    assert result["AAA"] == pytest.approx(expected)


def test_weighted_relative_strength_ranks_faster_mover_higher():
    session_count = 260
    closes = pd.DataFrame(
        {
            "SLOW": _price_path(session_count, 0.0005),
            "FAST": _price_path(session_count, 0.003),
        }
    )

    result = weighted_relative_strength(closes)

    assert result["FAST"] > result["SLOW"]


def test_weighted_relative_strength_raises_below_min_sessions():
    closes = pd.DataFrame({"AAA": _price_path(100, 0.001)})
    with pytest.raises(ValueError):
        weighted_relative_strength(closes)


def test_weighted_relative_strength_raises_when_score_would_be_nan():
    # Exactly 252 sessions passes the raw session-count gate but cannot
    # produce a non-NaN r4 term (that needs more than 252 sessions), so
    # this must raise rather than silently emit NaN.
    closes = pd.DataFrame({"AAA": _price_path(252, 0.001)})
    with pytest.raises(ValueError):
        weighted_relative_strength(closes)
