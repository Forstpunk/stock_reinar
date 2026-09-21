from __future__ import annotations

import types

import pandas as pd
import pytest

from nse_screener.experimental.busted import NotValidatedError, detect_busted_patterns
from nse_screener.experimental.validation_log import (
    DetectionLogEntry,
    load_detections,
    log_detection,
)


def _series(values: list[float]) -> pd.Series:
    dates = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=dates)


# pattern window (indices 0-9): min=98, max=102
_PATTERN_WINDOW = [100.0, 101.0, 99.0, 102.0, 98.0, 100.0, 101.0, 99.0, 100.0, 101.0]
# breakout at index 10 (95 < 98), adverse dip to 94 (within 5% of 98), recovers
# and confirms above 102 at index 17 (sessions_elapsed = 7)
_BREAKOUT_AND_CONFIRM = [95.0, 94.0, 95.0, 96.0, 97.0, 99.0, 101.0, 103.0]


def test_detects_single_bust():
    tail = [103.0] * 22  # stays well above pattern_low for the rest of the series
    closes = _series(_PATTERN_WINDOW + _BREAKOUT_AND_CONFIRM + tail)

    patterns = detect_busted_patterns(
        "AAA", closes, lookback=20, pattern_window=10, bust_threshold=0.05
    )

    assert len(patterns) == 1
    p = patterns[0]
    assert p.pattern_low == pytest.approx(98.0)
    assert p.pattern_high == pytest.approx(102.0)
    assert p.bust_confirmed is True
    assert p.sessions_elapsed == 7
    assert p.bust_type == "single"


def test_detects_double_bust():
    # same as the single-bust series through confirmation, then a second
    # reversal back below pattern_low (96 < 98) two sessions after confirmation
    tail = [103.0, 96.0] + [103.0] * 20
    closes = _series(_PATTERN_WINDOW + _BREAKOUT_AND_CONFIRM + tail)

    patterns = detect_busted_patterns(
        "AAA", closes, lookback=20, pattern_window=10, bust_threshold=0.05
    )

    assert len(patterns) == 1
    assert patterns[0].bust_type == "double"


def test_no_pattern_when_breakout_keeps_falling():
    # breaks out and keeps dropping well past the bust_threshold -- not a bust
    tail = [80.0] * 20
    closes = _series(_PATTERN_WINDOW + [95.0, 90.0, 85.0] + tail)

    patterns = detect_busted_patterns(
        "AAA", closes, lookback=20, pattern_window=10, bust_threshold=0.05
    )
    assert patterns == []


def test_no_pattern_when_never_confirmed_within_lookback():
    # dips and stays within bust_threshold but never closes back above pattern_high
    tail = [96.0] * 20
    closes = _series(_PATTERN_WINDOW + [95.0] + tail)

    patterns = detect_busted_patterns(
        "AAA", closes, lookback=20, pattern_window=10, bust_threshold=0.05
    )
    assert patterns == []


def test_not_validated_error_when_called_from_outside_experimental_or_tests():
    fake_module = types.ModuleType("nse_screener.ranking")
    code = compile(
        "from nse_screener.experimental.busted import detect_busted_patterns\n"
        "import pandas as pd\n"
        "detect_busted_patterns('AAA', pd.Series([1.0] * 50, "
        "index=pd.bdate_range('2024-01-01', periods=50)))\n",
        "<fake_caller>",
        "exec",
    )
    with pytest.raises(NotValidatedError):
        exec(code, fake_module.__dict__)


def test_detections_are_written_to_the_log(tmp_path):
    log_path = tmp_path / "detections.jsonl"
    entry = DetectionLogEntry(
        symbol="AAA",
        detection_date=pd.Timestamp("2024-02-01").date(),
        price_at_detection=103.0,
        gate_verdict_at_detection="HEALTHY",
        pattern_metadata={"bust_type": "single", "sessions_elapsed": 7},
    )
    log_detection(entry, log_path)

    loaded = load_detections(log_path)
    assert len(loaded) == 1
    assert loaded[0].symbol == "AAA"
    assert loaded[0].pattern_metadata["bust_type"] == "single"
