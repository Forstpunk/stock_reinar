"""Busted-pattern detection -- QUARANTINED. Read this before using anything here.

Busted-pattern outperformance is the most cross-validated claim in the
technical source material -- Bulkowski measured single busted patterns
beating non-busted counterparts in over 90% of bull-market contests
(40% vs 28% average rise after bear-market down-breakouts), and Grimes
reaches the same conclusion independently from market structure. **But
it has never been tested on NSE data.** A closely related specification
(O'Neil breakout) tested on 1,253 NSE trades produced zero edge versus
random entry. Until busted patterns are validated on Indian data, they
are a hypothesis.

Honest labelling: what this module detects is a 30-session high/low
breakout-and-reversal proxy, NOT Bulkowski's named chart patterns (cup
with handle, double bottom, head-and-shoulders). His per-pattern
statistics do not transfer to this proxy -- never cite his percentages
for a detection made by this function.

Hard rules, enforced by this module, not just documented:
1. Never contributes to the v1 composite score, at any weight.
2. Never appears in the Stage 1 shortlist.
3. Output belongs only in a report section headed
   "EXPERIMENTAL -- UNVALIDATED ON NSE".
4. Every detection should be written to the forward-validation log
   (``validation_log.py``).
5. This module raises ``NotValidatedError`` if anything outside
   ``nse_screener.experimental``, the test suite, or the workbench's
   labelled ``experimental-scan`` command calls into it -- see
   ``_require_experimental_caller``. ``__main__`` is deliberately NOT
   exempt: a script run directly is exactly the accidental-production-use
   case this guards against. The workbench qualifies by its module spec
   (``python -m nse_screener.workbench``), not by being ``__main__``.
"""

from __future__ import annotations

import inspect
from datetime import date
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict


class NotValidatedError(Exception):
    """Raised when code outside ``nse_screener.experimental`` (or tests) calls
    into this module -- it is quarantined by design, not by convention."""


#: Module-name prefixes sanctioned to call into this module. ``__main__`` is
#: deliberately absent: a script run directly must raise. The workbench is
#: listed by its spec name, which ``python -m nse_screener.workbench`` sets
#: even though that frame's ``__name__`` is ``"__main__"``.
_ALLOWED_CALLER_PREFIXES: tuple[str, ...] = (
    "tests.",
    "test_",
    "nse_screener.workbench.__main__",
)


def _frame_module_name(frame_info: inspect.FrameInfo) -> str:
    """The module a frame belongs to, by spec name where one exists.

    ``__spec__.name`` is set for imported modules and for ``python -m``
    entry points; it is ``None`` for a script run directly, whose
    ``__name__`` is ``"__main__"`` -- so that case falls through to the
    name and is refused by the allowlist.
    """
    spec = frame_info.frame.f_globals.get("__spec__")
    if spec is not None and getattr(spec, "name", None):
        return str(spec.name)
    return str(frame_info.frame.f_globals.get("__name__", ""))


def _require_experimental_caller(public_function_name: str) -> None:
    """Raise unless the nearest caller outside this package is sanctioned.

    Walks the stack outward past ``nse_screener.experimental`` frames so
    that an internal wrapper cannot launder an outside caller, and decides
    on the first frame that is not experimental. ``__main__`` is NOT exempt
    -- see ``_ALLOWED_CALLER_PREFIXES``.
    """
    for frame_info in inspect.stack()[1:]:
        module = _frame_module_name(frame_info)
        if module.startswith("nse_screener.experimental"):
            continue  # internal hop, keep walking outward
        if module.startswith(_ALLOWED_CALLER_PREFIXES):
            return
        raise NotValidatedError(
            f"{public_function_name} may only be called from nse_screener.experimental, "
            f"the test suite or the workbench experimental-scan command -- called from "
            f"{module!r}. This module is unvalidated on NSE data and quarantined by "
            "design; see the module docstring."
        )
    raise NotValidatedError(
        f"{public_function_name}: no legitimate caller found in the stack."
    )


class BustedPattern(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    breakout_date: date
    pattern_high: float
    pattern_low: float
    max_adverse_move_pct: float
    bust_confirmed: bool
    bust_confirmation_date: Optional[date]
    sessions_elapsed: Optional[int]
    bust_type: Optional[Literal["single", "double"]]


def detect_busted_patterns(
    symbol: str,
    closes: pd.Series,
    lookback: int = 45,
    pattern_window: int = 30,
    bust_threshold: float = 0.05,
) -> list[BustedPattern]:
    """Detect single/double busted downward breakouts on ``closes``.

    For each session whose close falls below the minimum close of the
    prior ``pattern_window`` sessions (a downward breakout): if the
    subsequent minimum close (scanned up to ``lookback`` sessions ahead)
    stays within ``bust_threshold`` of that broken support level, and
    price later closes back above the pre-breakout ``pattern_window``'s
    maximum, the breakout is recorded as a confirmed (single) bust. If
    price reverses a second time -- closing back below ``pattern_low``
    within a further ``lookback`` sessions after confirmation -- it is
    reclassified ``"double"``.

    This is a mechanical proxy, not Bulkowski's named patterns -- see
    the module docstring.
    """
    _require_experimental_caller("detect_busted_patterns")

    closes = closes.sort_index()
    patterns: list[BustedPattern] = []
    n = len(closes)

    # A manual index (not a for-loop) so that once a breakout candidate is
    # evaluated -- confirmed or not -- the scan jumps past it, rather than
    # re-triggering on the same continued decline as a "new" breakout once
    # today's low enters tomorrow's own trailing window.
    i = pattern_window
    while i < n:
        window = closes.iloc[i - pattern_window : i]
        pattern_low = float(window.min())
        pattern_high = float(window.max())

        if closes.iloc[i] >= pattern_low:
            i += 1
            continue  # no downward breakout here

        breakout_date = closes.index[i].date()
        scan_end = min(i + lookback, n)
        forward = closes.iloc[i:scan_end]

        min_subsequent = float(forward.min())
        max_adverse_move_pct = (pattern_low - min_subsequent) / pattern_low * 100
        if max_adverse_move_pct > bust_threshold * 100:
            i = scan_end  # the move kept going, not a bust -- skip past it
            continue

        confirmation_idx = None
        for j in range(i, scan_end):
            if closes.iloc[j] > pattern_high:
                confirmation_idx = j
                break

        if confirmation_idx is None:
            i = scan_end  # never recovered within lookback -- skip past it
            continue

        bust_confirmation_date = closes.index[confirmation_idx].date()
        sessions_elapsed = confirmation_idx - i

        double_scan_end = min(confirmation_idx + lookback, n)
        bust_type: Literal["single", "double"] = "single"
        for k in range(confirmation_idx, double_scan_end):
            if closes.iloc[k] < pattern_low:
                bust_type = "double"
                break

        patterns.append(
            BustedPattern(
                symbol=symbol,
                breakout_date=breakout_date,
                pattern_high=pattern_high,
                pattern_low=pattern_low,
                max_adverse_move_pct=max_adverse_move_pct,
                bust_confirmed=True,
                bust_confirmation_date=bust_confirmation_date,
                sessions_elapsed=sessions_elapsed,
                bust_type=bust_type,
            )
        )
        i = confirmation_idx + 1  # past the whole confirmed pattern

    return patterns
