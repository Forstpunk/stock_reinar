from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_screener.data import DataIntegrityError, validate_price_data


def _frame(dates: pd.DatetimeIndex, start: float = 100.0) -> pd.DataFrame:
    close = start + np.arange(len(dates), dtype=float)
    return pd.DataFrame({"Close": close}, index=dates)


def test_symbol_with_too_few_sessions_is_rejected():
    index_dates = pd.bdate_range("2023-01-01", periods=320)
    index_frame = _frame(index_dates)
    thin_dates = index_dates[-100:]
    frames = {"GOOD": _frame(index_dates), "THIN": _frame(thin_dates)}

    report = validate_price_data(frames, index_frame, min_sessions=300)

    assert "THIN" not in report.accepted
    assert "GOOD" in report.accepted
    assert "sessions" in report.rejected["THIN"]


def test_symbol_missing_more_than_2pct_of_calendar_is_rejected():
    index_dates = pd.bdate_range("2023-01-01", periods=320)
    index_frame = _frame(index_dates)

    # Drop 10 of the most recent 252 reference sessions (10/252 ~= 4%,
    # above the 2% threshold) while keeping the total row count well
    # above min_sessions so this is purely a calendar-alignment failure.
    reference_window = index_dates[-252:]
    dropped = reference_window[:10]
    symbol_dates = index_dates.difference(dropped)
    frames = {"GOOD": _frame(index_dates), "GAPPY": _frame(symbol_dates)}

    report = validate_price_data(frames, index_frame, min_sessions=300)

    assert "GAPPY" not in report.accepted
    assert "GOOD" in report.accepted
    assert "calendar" in report.rejected["GAPPY"]


def test_symbol_within_2pct_calendar_gap_is_accepted():
    index_dates = pd.bdate_range("2023-01-01", periods=320)
    index_frame = _frame(index_dates)

    reference_window = index_dates[-252:]
    dropped = reference_window[:3]  # 3/252 ~= 1.2%, under the 2% threshold
    symbol_dates = index_dates.difference(dropped)
    frames = {"OK": _frame(symbol_dates)}

    report = validate_price_data(frames, index_frame, min_sessions=300)

    assert "OK" in report.accepted


def test_duplicate_index_raises_immediately():
    index_dates = pd.bdate_range("2023-01-01", periods=320)
    dup_dates = index_dates.insert(0, index_dates[0])
    index_frame = _frame(dup_dates)
    frames = {"AAA": _frame(index_dates)}

    with pytest.raises(DataIntegrityError):
        validate_price_data(frames, index_frame, min_sessions=300)


def test_all_symbols_rejected_raises():
    index_dates = pd.bdate_range("2023-01-01", periods=320)
    index_frame = _frame(index_dates)
    frames = {"THIN": _frame(index_dates[-50:])}

    with pytest.raises(DataIntegrityError):
        validate_price_data(frames, index_frame, min_sessions=300)


def test_non_positive_price_is_rejected():
    index_dates = pd.bdate_range("2023-01-01", periods=320)
    index_frame = _frame(index_dates)
    bad_frame = _frame(index_dates)
    bad_frame.loc[bad_frame.index[-1], "Close"] = 0.0
    frames = {"GOOD": _frame(index_dates), "ZERO": bad_frame}

    report = validate_price_data(frames, index_frame, min_sessions=300)

    assert "ZERO" not in report.accepted
    assert "GOOD" in report.accepted
    assert "non-positive" in report.rejected["ZERO"]


def test_impossible_move_is_flagged_not_rejected():
    index_dates = pd.bdate_range("2023-01-01", periods=320)
    index_frame = _frame(index_dates)
    spike_frame = _frame(index_dates)
    spike_frame.loc[spike_frame.index[-1], "Close"] = spike_frame["Close"].iloc[-2] * 3
    frames = {"SPIKE": spike_frame}

    report = validate_price_data(frames, index_frame, min_sessions=300)

    assert "SPIKE" in report.accepted
    assert len(report.flagged) == 1
    assert report.flagged[0][0] == "SPIKE"
