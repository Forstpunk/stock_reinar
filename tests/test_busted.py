from __future__ import annotations

import importlib.machinery
import types

import pandas as pd
import pytest

from nse_screener.experimental import busted
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


# -- The quarantine must hold against every non-sanctioned caller --------------

_CALL_SOURCE = (
    "from nse_screener.experimental.busted import detect_busted_patterns\n"
    "import pandas as pd\n"
    "detect_busted_patterns('AAA', pd.Series([1.0] * 50, "
    "index=pd.bdate_range('2024-01-01', periods=50)))\n"
)


def _exec_as(module_name: str, spec_name: str | None = None) -> None:
    """Run the quarantined call from a namespace impersonating ``module_name``.

    ``spec_name`` mirrors ``__spec__.name`` as set by ``python -m``; a script
    run directly has ``__spec__ = None`` and ``__name__ == "__main__"``.
    """
    namespace: dict[str, object] = {"__name__": module_name, "__spec__": None}
    if spec_name is not None:
        namespace["__spec__"] = importlib.machinery.ModuleSpec(spec_name, None)
    exec(compile(_CALL_SOURCE, f"<{module_name}>", "exec"), namespace)


def test_script_run_directly_as_main_is_not_exempt():
    # A script run as ``python some_script.py`` is the accidental-production
    # use the quarantine exists to catch. It must raise.
    with pytest.raises(NotValidatedError):
        _exec_as("__main__")


def test_main_is_not_in_the_allowlist():
    assert "__main__" not in busted._ALLOWED_CALLER_PREFIXES


def test_internal_experimental_hop_does_not_launder_an_outside_caller():
    # A wrapper inside nse_screener.experimental must not make the caller of
    # that wrapper look like an experimental caller: the guard walks past
    # internal frames to the nearest outside caller.
    wrapper_ns: dict[str, object] = {"__name__": "nse_screener.experimental.wrapper"}
    exec(
        compile(
            "from nse_screener.experimental.busted import detect_busted_patterns\n"
            "import pandas as pd\n"
            "def run():\n"
            "    return detect_busted_patterns('AAA', pd.Series([1.0] * 50, "
            "index=pd.bdate_range('2024-01-01', periods=50)))\n",
            "<experimental_wrapper>",
            "exec",
        ),
        wrapper_ns,
    )
    outside_ns: dict[str, object] = {"__name__": "nse_screener.ranking", "run": wrapper_ns["run"]}
    with pytest.raises(NotValidatedError):
        exec(compile("run()\n", "<outside_caller>", "exec"), outside_ns)


def test_call_from_within_experimental_package_succeeds():
    # An experimental frame is an internal hop; the nearest outside caller
    # here is this test module, which is sanctioned.
    _exec_as("nse_screener.experimental.some_module")


def test_call_from_test_suite_succeeds():
    _exec_as(__name__)


def test_workbench_experimental_cli_is_sanctioned():
    # ``python -m nse_screener.workbench experimental-scan`` runs with
    # __name__ == "__main__" but __spec__.name naming the workbench: the
    # labelled EXPERIMENTAL command that feeds the validation log.
    _exec_as("__main__", spec_name="nse_screener.workbench.__main__")


# -- Regression: detection_date must be the confirmation date, not the ---------
# -- breakout date, or forward-return scoring re-measures an already-known -----
# -- recovery (lookahead bias). See _build_detection_entry's docstring. --------


def test_logged_detection_date_is_confirmation_not_breakout():
    from nse_screener.workbench.__main__ import _build_detection_entry

    tail = [103.0] * 22
    closes = _series(_PATTERN_WINDOW + _BREAKOUT_AND_CONFIRM + tail)
    patterns = detect_busted_patterns(
        "AAA", closes, lookback=20, pattern_window=10, bust_threshold=0.05
    )
    p = patterns[0]
    assert p.breakout_date != p.bust_confirmation_date  # the fixture must exercise this

    entry = _build_detection_entry("AAA", closes, p, "HEALTHY")

    assert entry.detection_date == p.bust_confirmation_date
    assert entry.detection_date != p.breakout_date
    assert entry.price_at_detection == pytest.approx(
        float(closes.loc[pd.Timestamp(p.bust_confirmation_date)])
    )
