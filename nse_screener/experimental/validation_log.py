"""Forward-validation log for experimental/busted.py detections.

Every detection is appended here; ``score_open_detections`` later checks
what actually happened, against a matched random-entry control -- that
comparison is the entire point of this log. Without it, the log is an
anecdote collection.

Print with every report: "N detections logged, M matured. Edge vs random
control: X pp. A minimum of 30 matured detections is required before
this result means anything."
"""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

#: Forward horizons, in sessions, a detection is scored at.
HORIZONS: tuple[int, ...] = (21, 63, 126)

#: Matured detections needed before edge-vs-control means anything.
MIN_MEANINGFUL_SAMPLE: int = 30


class DetectionLogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    detection_date: date
    price_at_detection: float
    gate_verdict_at_detection: str
    pattern_metadata: dict[str, object]


def log_detection(entry: DetectionLogEntry, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")


def load_detections(log_path: Path) -> list[DetectionLogEntry]:
    if not log_path.exists():
        return []
    entries: list[DetectionLogEntry] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(DetectionLogEntry.model_validate_json(line))
    return entries


class HorizonResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon_sessions: int
    matured_count: int
    detection_return_mean_pct: float
    detection_return_median_pct: float
    detection_win_rate: float
    control_return_mean_pct: float
    edge_vs_control_pp: float


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_detections: int
    horizons: dict[int, HorizonResult]

    def summary_line(self) -> str:
        matured = max((h.matured_count for h in self.horizons.values()), default=0)
        return (
            f"{self.total_detections} detections logged, {matured} matured. "
            f"A minimum of {MIN_MEANINGFUL_SAMPLE} matured detections is required "
            "before this result means anything."
        )


def score_open_detections(
    log_path: Path,
    price_data: dict[str, pd.Series],
    seed: Optional[int] = None,
) -> ValidationReport:
    """Score every logged detection at each horizon against a matched random control.

    ``price_data`` maps symbol to a Close-price Series covering (at
    least) the detection date through the longest horizon. A detection
    "matures" at a horizon only if its symbol's price data extends that
    far past the detection date -- otherwise it is excluded from that
    horizon's stats, not padded or estimated.
    """
    detections = load_detections(log_path)
    rng = random.Random(seed)
    horizons: dict[int, HorizonResult] = {}

    for horizon in HORIZONS:
        detection_returns: list[float] = []
        control_returns: list[float] = []

        for entry in detections:
            detection_return = _forward_return(
                price_data, entry.symbol, entry.detection_date, horizon
            )
            if detection_return is None:
                continue

            control_candidates = [s for s in price_data if s != entry.symbol]
            control_return = None
            attempts = list(control_candidates)
            rng.shuffle(attempts)
            for control_symbol in attempts:
                control_return = _forward_return(
                    price_data, control_symbol, entry.detection_date, horizon
                )
                if control_return is not None:
                    break
            if control_return is None:
                continue

            detection_returns.append(detection_return)
            control_returns.append(control_return)

        if detection_returns:
            sorted_returns = sorted(detection_returns)
            mean_r = sum(detection_returns) / len(detection_returns)
            median_r = sorted_returns[len(sorted_returns) // 2]
            win_rate = sum(1 for r in detection_returns if r > 0) / len(detection_returns)
        else:
            mean_r = median_r = win_rate = 0.0
        control_mean = sum(control_returns) / len(control_returns) if control_returns else 0.0

        horizons[horizon] = HorizonResult(
            horizon_sessions=horizon,
            matured_count=len(detection_returns),
            detection_return_mean_pct=mean_r,
            detection_return_median_pct=median_r,
            detection_win_rate=win_rate,
            control_return_mean_pct=control_mean,
            edge_vs_control_pp=mean_r - control_mean,
        )

    return ValidationReport(total_detections=len(detections), horizons=horizons)


def _forward_return(
    price_data: dict[str, pd.Series], symbol: str, entry_date: date, horizon: int
) -> Optional[float]:
    series = price_data.get(symbol)
    if series is None:
        return None
    series = series.sort_index()
    entry_pos = int(series.index.searchsorted(pd.Timestamp(entry_date)))
    if entry_pos >= len(series):
        return None
    exit_pos = entry_pos + horizon
    if exit_pos >= len(series):
        return None
    entry_price = float(series.iloc[entry_pos])
    exit_price = float(series.iloc[exit_pos])
    if entry_price <= 0:
        return None
    return (exit_price / entry_price - 1.0) * 100
