from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from nse_screener.experimental.validation_log import (
    HORIZONS,
    DetectionLogEntry,
    log_detection,
    score_open_detections,
)


def _flat_series(days: int, start: str, value: float) -> pd.Series:
    dates = pd.bdate_range(start, periods=days)
    return pd.Series([value] * days, index=dates)


def _rising_series(days: int, start: str, start_value: float, daily_return: float) -> pd.Series:
    import numpy as np

    dates = pd.bdate_range(start, periods=days)
    return pd.Series(start_value * (1 + daily_return) ** np.arange(days), index=dates)


def test_score_open_detections_computes_edge_vs_control(tmp_path):
    log_path = tmp_path / "detections.jsonl"
    entry = DetectionLogEntry(
        symbol="AAA",
        detection_date=date(2024, 1, 2),
        price_at_detection=100.0,
        gate_verdict_at_detection="HEALTHY",
        pattern_metadata={"bust_type": "single"},
    )
    log_detection(entry, log_path)

    # AAA rises strongly after detection; BBB (the control) stays flat
    price_data = {
        "AAA": _rising_series(200, "2024-01-02", 100.0, 0.005),
        "BBB": _flat_series(200, "2024-01-02", 100.0),
    }

    report = score_open_detections(log_path, price_data, seed=42)

    assert report.total_detections == 1
    shortest_horizon = min(HORIZONS)
    result = report.horizons[shortest_horizon]
    assert result.matured_count == 1
    assert result.detection_return_mean_pct > 0
    assert result.control_return_mean_pct == pytest.approx(0.0)
    assert result.edge_vs_control_pp > 0


def test_score_open_detections_excludes_symbols_without_enough_forward_data(tmp_path):
    log_path = tmp_path / "detections.jsonl"
    entry = DetectionLogEntry(
        symbol="AAA",
        detection_date=date(2024, 1, 2),
        price_at_detection=100.0,
        gate_verdict_at_detection="HEALTHY",
        pattern_metadata={},
    )
    log_detection(entry, log_path)

    # only 10 days of forward data -- nowhere near the shortest horizon (21)
    price_data = {
        "AAA": _flat_series(10, "2024-01-02", 100.0),
        "BBB": _flat_series(10, "2024-01-02", 100.0),
    }

    report = score_open_detections(log_path, price_data, seed=1)
    for result in report.horizons.values():
        assert result.matured_count == 0


def test_summary_line_reports_counts():
    from nse_screener.experimental.validation_log import HorizonResult, ValidationReport

    report = ValidationReport(
        total_detections=5,
        horizons={
            21: HorizonResult(
                horizon_sessions=21, matured_count=2, detection_return_mean_pct=1.0,
                detection_return_median_pct=1.0, detection_win_rate=0.5,
                control_return_mean_pct=0.5, edge_vs_control_pp=0.5,
            )
        },
    )
    line = report.summary_line()
    assert "5 detections logged" in line
    assert "2 matured" in line
    assert "30" in line
