from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from nse_screener.tracker import TrackedEntry, TrackedRun, compute_evaluations, load_runs, log_run


def _series(session_count: int, start: float, daily_return: float) -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=session_count)
    prices = start * (1 + daily_return) ** np.arange(session_count)
    return pd.Series(prices, index=dates)


def _run(entry_close: float = 100.0, index_close: float = 20000.0) -> TrackedRun:
    return TrackedRun(
        run_date=date(2024, 1, 2),
        gate_verdict="HEALTHY",
        index_close_at_run=index_close,
        holding_horizon_sessions=63,
        entries=[TrackedEntry(symbol="AAA", rank=1, composite_score=0.9, entry_close=entry_close)],
    )


def test_compute_evaluations_matched_benchmark_arithmetic():
    run = _run(entry_close=100.0, index_close=20000.0)
    prices = {
        "AAA": _series(70, 100.0, 0.002),  # stock rises ~0.2%/session
        "^NSEI": _series(70, 20000.0, 0.0005),  # index rises ~0.05%/session
    }

    results = compute_evaluations(run, prices, index_symbol="^NSEI", min_sessions=63)

    assert len(results) == 1
    result = results[0]

    expected_stock_return = (prices["AAA"].iloc[63] / 100.0 - 1.0) * 100
    expected_index_return = (prices["^NSEI"].iloc[63] / 20000.0 - 1.0) * 100
    assert result.stock_return_pct == pytest.approx(expected_stock_return)
    assert result.index_return_pct == pytest.approx(expected_index_return)
    assert result.excess_return_pct == pytest.approx(expected_stock_return - expected_index_return)
    # the faster-rising stock should show positive excess return vs the index
    assert result.excess_return_pct > 0


def test_compute_evaluations_empty_when_not_matured():
    run = _run()
    prices = {
        "AAA": _series(30, 100.0, 0.002),  # only 30 sessions, below the 63 threshold
        "^NSEI": _series(30, 20000.0, 0.0005),
    }
    assert compute_evaluations(run, prices, index_symbol="^NSEI", min_sessions=63) == []


def test_compute_evaluations_skips_symbol_missing_from_prices():
    run = _run()
    prices = {"^NSEI": _series(70, 20000.0, 0.0005)}  # AAA never fetched (e.g. delisted)
    assert compute_evaluations(run, prices, index_symbol="^NSEI", min_sessions=63) == []


def test_log_run_and_load_runs_roundtrip(tmp_path):
    log_path = tmp_path / "tracker.jsonl"
    run_a = _run(entry_close=100.0)
    run_b = _run(entry_close=200.0)

    log_run(run_a, log_path)
    log_run(run_b, log_path)

    loaded = load_runs(log_path)
    assert len(loaded) == 2
    assert loaded[0].entries[0].entry_close == 100.0
    assert loaded[1].entries[0].entry_close == 200.0


def test_load_runs_missing_file_returns_empty(tmp_path):
    assert load_runs(tmp_path / "does_not_exist.jsonl") == []
