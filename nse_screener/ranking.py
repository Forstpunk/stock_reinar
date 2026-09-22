"""Eligibility filtering, composite scoring, and shortlist construction."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

from nse_screener.config import ScreenerConfig
from nse_screener.data import FundamentalData
from nse_screener.factors.financial import FScoreResult, piotroski_f_score
from nse_screener.factors.forensic import ForensicResult, forensic_screen
from nse_screener.factors.momentum import weighted_relative_strength
from nse_screener.factors.quality import gross_profitability, return_on_equity
from nse_screener.factors.value import book_to_price, earnings_yield


class ShortlistEntry(BaseModel):
    """One shortlisted symbol with every component that fed its composite score."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    composite_score: float
    # What ``composite_score`` actually is. ``"composite"`` is the weighted
    # blend of RS, F-Score, gross profitability and ROE percentiles;
    # ``"rs_only"`` (the --skip-fundamentals path) is the RS percentile alone.
    # Scores with different bases are not comparable.
    score_basis: Literal["composite", "rs_only"]
    rs_percentile: float
    raw_weighted_rs: float
    avg_traded_value_crore: float
    atr_pct: float
    distance_from_52w_high_pct: float
    holding_horizon_sessions: int

    # Fundamentals-derived fields. ``None`` only in --skip-fundamentals runs,
    # which is why this is exposed rather than defaulted to a fake zero.
    f_score_percentile: Optional[float] = None
    gross_profitability_percentile: Optional[float] = None
    roe_percentile: Optional[float] = None
    gross_profitability: Optional[float] = None
    roe: Optional[float] = None
    f_score: Optional[FScoreResult] = None
    forensic_flags: list[str] = []

    # Value (cheapness) factor. ``value_percentile`` is the average of the
    # earnings-yield and book-to-price percentiles, blended into one weight
    # slot (``config.weight_value``) rather than two, for composite
    # simplicity -- see ``factors/value.py`` for the evidence basis.
    earnings_yield: Optional[float] = None
    book_to_price: Optional[float] = None
    value_percentile: Optional[float] = None


class Shortlist(BaseModel):
    """Result of one ranking run.

    Never padded -- ``entries`` may be shorter than ``requested_top_n``.
    """

    model_config = ConfigDict(frozen=True)

    entries: list[ShortlistEntry]
    requested_top_n: int
    qualifying_count: int
    excluded_count: dict[str, int]
    disqualified: dict[str, list[str]]
    fundamentals_included: bool = True
    shortfall_note: Optional[str] = None


def build_shortlist(
    price_frames: dict[str, pd.DataFrame],
    fundamentals: dict[str, FundamentalData],
    config: ScreenerConfig,
    top_n: int = 5,
) -> Shortlist:
    """Filter, score, and rank a validated universe into a top-N shortlist.

    ``price_frames`` must contain only symbols that already passed
    ``data.validate_price_data`` (i.e. the ``accepted`` list). ``fundamentals``
    maps symbol to a ``FundamentalData`` with ``.prior`` set; symbols absent
    from this dict, or whose factors cannot be computed, are excluded and
    counted in ``excluded_count`` -- never silently dropped.

    Every entry carries ``score_basis="composite"``.
    """
    excluded_count: dict[str, int] = defaultdict(int)
    disqualified: dict[str, list[str]] = {}
    forensic_by_symbol: dict[str, ForensicResult] = {}

    eligible_symbols = _filter_eligible(
        price_frames,
        config,
        require_fundamentals=True,
        fundamentals=fundamentals,
        excluded_count=excluded_count,
        disqualified=disqualified,
        forensic_by_symbol=forensic_by_symbol,
    )
    if not eligible_symbols:
        return _finalise([], top_n, excluded_count, disqualified, fundamentals_included=True)

    closes = pd.DataFrame({symbol: price_frames[symbol]["Close"] for symbol in eligible_symbols})
    rs_values = _compute_rs(eligible_symbols, closes, config, excluded_count)
    eligible_symbols = [s for s in eligible_symbols if s in rs_values]
    if not eligible_symbols:
        return _finalise([], top_n, excluded_count, disqualified, fundamentals_included=True)
    rs_series = pd.Series(rs_values, name="weighted_rs")

    f_scores: dict[str, FScoreResult] = {}
    gp_values: dict[str, float] = {}
    roe_values: dict[str, float] = {}
    ey_values: dict[str, float] = {}
    btp_values: dict[str, float] = {}
    for symbol in eligible_symbols:
        fdata = fundamentals[symbol]
        last_close = float(price_frames[symbol]["Close"].iloc[-1])
        # Each factor can legitimately raise on a real, non-erroneous
        # financial state (e.g. negative book equity at a heavily leveraged
        # infra name) -- one such symbol must not crash a multi-hundred-name
        # screen. Assigned atomically per symbol so the five dicts below
        # never end up with a partially-computed entry for a failed symbol.
        try:
            f_score = piotroski_f_score(fdata, fdata.prior)  # type: ignore[arg-type]
            gp = gross_profitability(fdata)
            roe = return_on_equity(fdata)
            ey = earnings_yield(fdata, last_close)
            btp = book_to_price(fdata, last_close)
        except ValueError:
            excluded_count["factor_computation_failed"] += 1
            continue
        f_scores[symbol] = f_score
        gp_values[symbol] = gp
        roe_values[symbol] = roe
        ey_values[symbol] = ey
        btp_values[symbol] = btp

    eligible_symbols = [s for s in eligible_symbols if s in f_scores]
    if not eligible_symbols:
        return _finalise([], top_n, excluded_count, disqualified, fundamentals_included=True)

    f_score_series = pd.Series({s: f_scores[s].total_score for s in eligible_symbols})
    gp_series = pd.Series(gp_values)
    roe_series = pd.Series(roe_values)
    ey_series = pd.Series(ey_values)
    btp_series = pd.Series(btp_values)

    rs_pct = rs_series.rank(pct=True)
    f_score_pct = f_score_series.rank(pct=True)
    gp_pct = gp_series.rank(pct=True)
    roe_pct = roe_series.rank(pct=True)
    ey_pct = ey_series.rank(pct=True)
    btp_pct = btp_series.rank(pct=True)
    value_pct = (ey_pct + btp_pct) / 2.0

    # WEIGHTS ARE UNVALIDATED. They reflect evidence strength of each factor,
    # NOT an optimised blend. Do not tune these against historical returns
    # without a proper out-of-sample split -- in-sample optimisation of
    # these weights will produce a curve-fitted result that fails live.
    # Changing them requires a fresh backtest with a matched random-entry
    # benchmark. See the comment on the weight_* fields in config.py for
    # why weight_value was set where it was.
    composite = (
        config.weight_rs * rs_pct
        + config.weight_f_score * f_score_pct
        + config.weight_value * value_pct
        + config.weight_gross_profitability * gp_pct
        + config.weight_roe * roe_pct
    )

    entries = [
        _build_entry(
            symbol,
            price_frames[symbol],
            config,
            composite_score=float(composite[symbol]),
            score_basis="composite",
            rs_percentile=float(rs_pct[symbol]),
            raw_weighted_rs=float(rs_series[symbol]),
            f_score_percentile=float(f_score_pct[symbol]),
            gross_profitability_percentile=float(gp_pct[symbol]),
            roe_percentile=float(roe_pct[symbol]),
            gross_profitability=float(gp_series[symbol]),
            roe=float(roe_series[symbol]),
            f_score=f_scores[symbol],
            forensic_flags=forensic_by_symbol[symbol].flags,
            earnings_yield=float(ey_series[symbol]),
            book_to_price=float(btp_series[symbol]),
            value_percentile=float(value_pct[symbol]),
        )
        for symbol in eligible_symbols
    ]
    return _finalise(entries, top_n, excluded_count, disqualified, fundamentals_included=True)


def build_momentum_only_shortlist(
    price_frames: dict[str, pd.DataFrame],
    config: ScreenerConfig,
    top_n: int = 5,
) -> Shortlist:
    """Rank purely on relative strength and liquidity, no fundamentals required.

    This is the ``--skip-fundamentals`` path. It cannot run the forensic
    screen (which needs fundamentals) or the quality/F-Score components of
    the composite, so every entry carries ``score_basis="rs_only"`` and its
    ``composite_score`` is the RS percentile alone.
    """
    excluded_count: dict[str, int] = defaultdict(int)

    eligible_symbols = _filter_eligible(
        price_frames,
        config,
        require_fundamentals=False,
        fundamentals={},
        excluded_count=excluded_count,
        disqualified={},
        forensic_by_symbol={},
    )
    if not eligible_symbols:
        return _finalise([], top_n, excluded_count, {}, fundamentals_included=False)

    closes = pd.DataFrame({symbol: price_frames[symbol]["Close"] for symbol in eligible_symbols})
    rs_values = _compute_rs(eligible_symbols, closes, config, excluded_count)
    eligible_symbols = [s for s in eligible_symbols if s in rs_values]
    if not eligible_symbols:
        return _finalise([], top_n, excluded_count, {}, fundamentals_included=False)
    rs_series = pd.Series(rs_values, name="weighted_rs")
    rs_pct = rs_series.rank(pct=True)

    entries = [
        _build_entry(
            symbol,
            price_frames[symbol],
            config,
            composite_score=float(rs_pct[symbol]),
            score_basis="rs_only",
            rs_percentile=float(rs_pct[symbol]),
            raw_weighted_rs=float(rs_series[symbol]),
        )
        for symbol in eligible_symbols
    ]
    return _finalise(entries, top_n, excluded_count, {}, fundamentals_included=False)


def _filter_eligible(
    price_frames: dict[str, pd.DataFrame],
    config: ScreenerConfig,
    require_fundamentals: bool,
    fundamentals: dict[str, FundamentalData],
    excluded_count: dict[str, int],
    disqualified: dict[str, list[str]],
    forensic_by_symbol: dict[str, ForensicResult],
) -> list[str]:
    """Apply the per-symbol eligibility filters, recording every exclusion.

    ``excluded_count``, ``disqualified`` and ``forensic_by_symbol`` are
    filled in place. With ``require_fundamentals=False`` the fundamentals
    and forensic checks are skipped entirely.
    """
    eligible_symbols: list[str] = []

    for symbol, frame in price_frames.items():
        if len(frame) < config.eligibility_min_sessions:
            excluded_count["insufficient_price_history"] += 1
            continue

        liquidity = _average_traded_value_crore(frame, config.liquidity_window_sessions)
        if liquidity < config.min_avg_traded_value_crore:
            excluded_count["insufficient_liquidity"] += 1
            continue

        if require_fundamentals:
            fdata = fundamentals.get(symbol)
            if fdata is None or fdata.prior is None:
                excluded_count["fundamentals_unavailable"] += 1
                continue

            forensic = forensic_screen(
                fdata,
                fdata.prior,
                cfo_ni_min=config.forensic_cfo_ni_min,
                accrual_ratio_max=config.forensic_accrual_ratio_max,
                dso_vs_revenue_growth_pp=config.forensic_dso_vs_revenue_growth_pp,
                inventory_vs_cogs_growth_pp=config.forensic_inventory_vs_cogs_growth_pp,
                leverage_jump_fraction=config.forensic_leverage_jump_fraction,
                disqualify_flag_count=config.forensic_disqualify_flag_count,
            )
            forensic_by_symbol[symbol] = forensic
            if forensic.disqualified:
                excluded_count["forensically_disqualified"] += 1
                disqualified[symbol] = forensic.flags
                continue

        eligible_symbols.append(symbol)

    return eligible_symbols


def _compute_rs(
    eligible_symbols: list[str],
    closes: pd.DataFrame,
    config: ScreenerConfig,
    excluded_count: dict[str, int],
) -> dict[str, float]:
    """Raw weighted relative strength per symbol, in ``eligible_symbols`` order.

    Computed one symbol at a time so that a symbol whose momentum cannot be
    computed (``weighted_relative_strength`` raises ``ValueError``) is
    counted under ``momentum_uncomputable`` and omitted, rather than
    aborting the whole run.
    """
    rs_values: dict[str, float] = {}
    for symbol in eligible_symbols:
        try:
            rs_values[symbol] = float(
                weighted_relative_strength(
                    closes[[symbol]],
                    quarter_sessions=config.momentum_quarter_sessions,
                    weights=config.momentum_weights,
                    min_sessions=config.momentum_min_sessions,
                ).iloc[0]
            )
        except ValueError:
            excluded_count["momentum_uncomputable"] += 1
    return rs_values


def _build_entry(
    symbol: str,
    frame: pd.DataFrame,
    config: ScreenerConfig,
    *,
    composite_score: float,
    score_basis: Literal["composite", "rs_only"],
    rs_percentile: float,
    raw_weighted_rs: float,
    f_score_percentile: Optional[float] = None,
    gross_profitability_percentile: Optional[float] = None,
    roe_percentile: Optional[float] = None,
    gross_profitability: Optional[float] = None,
    roe: Optional[float] = None,
    f_score: Optional[FScoreResult] = None,
    forensic_flags: Optional[list[str]] = None,
    earnings_yield: Optional[float] = None,
    book_to_price: Optional[float] = None,
    value_percentile: Optional[float] = None,
) -> ShortlistEntry:
    """Assemble one entry, deriving the price-based descriptive fields from ``frame``."""
    return ShortlistEntry(
        symbol=symbol,
        composite_score=composite_score,
        score_basis=score_basis,
        rs_percentile=rs_percentile,
        f_score_percentile=f_score_percentile,
        gross_profitability_percentile=gross_profitability_percentile,
        roe_percentile=roe_percentile,
        raw_weighted_rs=raw_weighted_rs,
        gross_profitability=gross_profitability,
        roe=roe,
        f_score=f_score,
        forensic_flags=forensic_flags if forensic_flags is not None else [],
        earnings_yield=earnings_yield,
        book_to_price=book_to_price,
        value_percentile=value_percentile,
        avg_traded_value_crore=float(
            _average_traded_value_crore(frame, config.liquidity_window_sessions)
        ),
        atr_pct=float(_atr_percent(frame)),
        distance_from_52w_high_pct=float(_distance_from_52w_high_pct(frame)),
        holding_horizon_sessions=config.holding_horizon_sessions,
    )


def _finalise(
    entries: list[ShortlistEntry],
    top_n: int,
    excluded_count: dict[str, int],
    disqualified: dict[str, list[str]],
    *,
    fundamentals_included: bool,
) -> Shortlist:
    """Sort by score, truncate to ``top_n`` and attach the shortfall note (never padded)."""
    if not entries:
        return Shortlist(
            entries=[],
            requested_top_n=top_n,
            qualifying_count=0,
            excluded_count=dict(excluded_count),
            disqualified=disqualified,
            fundamentals_included=fundamentals_included,
            shortfall_note="no symbols passed eligibility filters",
        )

    ranked = sorted(entries, key=lambda entry: entry.composite_score, reverse=True)
    top_entries = ranked[:top_n]

    shortfall_note = None
    if len(top_entries) < top_n:
        shortfall_note = (
            f"only {len(top_entries)} of {top_n} requested names qualified "
            "after eligibility filtering -- list is not padded"
        )

    return Shortlist(
        entries=top_entries,
        requested_top_n=top_n,
        qualifying_count=len(entries),
        excluded_count=dict(excluded_count),
        disqualified=disqualified,
        fundamentals_included=fundamentals_included,
        shortfall_note=shortfall_note,
    )


def _average_traded_value_crore(frame: pd.DataFrame, window: int) -> float:
    traded_value = frame["Close"] * frame["Volume"]
    avg = traded_value.rolling(window).mean().iloc[-1]
    if pd.isna(avg):
        raise ValueError("insufficient history to compute average traded value")
    return float(avg) / 1e7


def _atr_percent(frame: pd.DataFrame, window: int = 14) -> float:
    high = frame["High"]
    low = frame["Low"]
    close = frame["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(window).mean().iloc[-1]
    if pd.isna(atr):
        raise ValueError("insufficient history to compute ATR")
    return float(atr) / float(close.iloc[-1]) * 100


def _distance_from_52w_high_pct(frame: pd.DataFrame, window: int = 252) -> float:
    close = frame["Close"]
    high_52w = close.tail(window).max()
    return (float(close.iloc[-1]) / float(high_52w) - 1.0) * 100
