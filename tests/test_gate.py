from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_screener.gate import Verdict, evaluate_market_gate


def _rising_series(session_count: int, start: float, daily_return: float) -> pd.Series:
    dates = pd.bdate_range("2019-01-01", periods=session_count)
    prices = start * (1 + daily_return) ** np.arange(session_count)
    return pd.Series(prices, index=dates)


def _flat_series(session_count: int, value: float) -> pd.Series:
    dates = pd.bdate_range("2019-01-01", periods=session_count)
    return pd.Series(np.full(session_count, value), index=dates)


def _universe(index_close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"AAA": index_close})


def test_healthy_when_steadily_rising():
    index_close = _rising_series(300, 100.0, 0.002)
    verdict = evaluate_market_gate(index_close, _universe(index_close))
    assert verdict.verdict == Verdict.HEALTHY
    assert verdict.index_vs_200dma_pct > 0
    assert verdict.dma200_slope_21d_pct > 0


def test_hostile_when_steadily_declining():
    index_close = _rising_series(300, 100.0, -0.002)
    verdict = evaluate_market_gate(index_close, _universe(index_close))
    assert verdict.verdict == Verdict.HOSTILE


def test_neutral_when_above_200dma_but_slope_has_rolled_over():
    # 229 flat sessions establish MA200 == 100, then a 20-session dip
    # (pulling the trailing 200-session average down) followed by a small
    # one-day pop that leaves today's close above the now-softer MA200,
    # while the MA200 itself is lower than it was 21 sessions ago.
    flat = np.full(229, 100.0)
    dipped = np.full(20, 100.0 * (1 - 0.02))
    pop = np.array([100.0 * (1 + 0.01)])
    prices = np.concatenate([flat, dipped, pop])
    dates = pd.bdate_range("2019-01-01", periods=len(prices))
    index_close = pd.Series(prices, index=dates)

    verdict = evaluate_market_gate(index_close, _universe(index_close))

    assert verdict.verdict == Verdict.NEUTRAL
    assert verdict.index_vs_200dma_pct > 0
    assert verdict.dma200_slope_21d_pct <= 0


def test_hostile_at_exact_200dma_boundary():
    # A perfectly flat series has close == MA200 exactly and slope == 0.
    # The verdict logic uses a strict ">" against MA200, so this must be
    # HOSTILE, not HEALTHY or NEUTRAL.
    index_close = _flat_series(230, 100.0)
    verdict = evaluate_market_gate(index_close, _universe(index_close))
    assert verdict.index_vs_200dma_pct == 0.0
    assert verdict.dma200_slope_21d_pct == 0.0
    assert verdict.verdict == Verdict.HOSTILE


def test_raises_on_insufficient_history():
    index_close = _flat_series(100, 100.0)
    with pytest.raises(ValueError):
        evaluate_market_gate(index_close, _universe(index_close))
