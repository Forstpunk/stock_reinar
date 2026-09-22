"""Point-in-time factor backtests against real NSE history.

Answers, with real numbers instead of literature citations alone: does
each v1 factor actually predict forward returns on NSE, using data this
project already has? Two different rigor levels, by necessity -- never
blur them into one number:

Momentum is pure price data with no restatement risk, so it can be
walked forward across many historical dates with full point-in-time
correctness: truncate the close-price frame at each as-of date and call
the exact ``weighted_relative_strength`` the live screener runs. The
factor computation and the forward-return measurement never share
information.

The fundamentals factors (F-Score, gross profitability, ROE, value)
cannot be backtested that way: yfinance has no point-in-time
fundamentals vintage store, only ~4 stored annual periods per symbol,
each "as currently known" rather than "as it stood on date X". The best
available proxy: treat each symbol's successive annual filings as their
own (current, prior) pairs (the same pairing and fiscal-gap check
``data.fetch_fundamentals`` uses, just walked across all valid pairs
instead of only the newest one), with ``REPORTING_LAG_DAYS`` added to
the "current" period's fiscal year-end as a conservative stand-in for
when that filing actually became public. This gives roughly 3
observations per symbol rather than continuous coverage. This ceiling
is structural, not a bug: it cannot be raised without a vintage
fundamentals data source, unlike momentum's as-of range (see
``backtest_momentum``, which uses the full available price history).

Both reuse the exact factor functions the live screener runs and the
forward-return lookup already validated in
``experimental.validation_log`` -- no new return-computation logic to
risk a second lookahead bug like the one found and fixed in
``experimental/busted.py``.

Statistical honesty layer (added after a review of the first version's
output found real numbers reported without the context needed to
interpret them):

- Overlapping forward-return windows from closely-spaced as-of dates
  are not independent observations. ``effective_n`` estimates the
  independent sample size from as-of date spacing vs. the horizon, and
  ``overlap_warning`` fires below 30. Treat ``n_observations`` as the
  row count, never as statistical power -- ``effective_n`` is the
  number that matters.
- Every headline statistic (rank IC, Q5-Q1 spread) carries a bootstrap
  95% CI, resampled by *as-of date*, not by row -- rows sharing an
  as-of date share a market regime and are not independent draws, so
  resampling them individually would understate the interval
  substantially, the same class of error ``effective_n`` corrects for.
  A CI spanning zero means "unknown", not "zero" -- do not read a wide
  interval as a null result.
- A large Q5-Q1 spread can be driven by a single quintile-boundary
  outlier rather than genuine ordering. ``spearman_quintile_monotonicity``
  and ``largest_single_step_share`` make that visible instead of letting
  spread alone stand in for evidence of quality.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, ConfigDict

from nse_screener.config import ScreenerConfig
from nse_screener.data import (
    _ANNUAL_GAP_DAYS_MAX,
    _ANNUAL_GAP_DAYS_MIN,
    FundamentalData,
    InsufficientFundamentalsError,
    _extract_period,
    _is_unreported_placeholder,
    fetch_price_history,
)
from nse_screener.experimental.validation_log import _forward_return
from nse_screener.factors.financial import piotroski_f_score
from nse_screener.factors.momentum import weighted_relative_strength
from nse_screener.factors.quality import gross_profitability, return_on_equity
from nse_screener.factors.value import book_to_price, earnings_yield

#: SEBI LODR Regulation 33 requires annual results within 60 days of
#: fiscal year-end; this adds a margin above that ceiling since it is a
#: deadline, not an average filing date, and understating the lag risks
#: exactly the lookahead bias this module exists to avoid.
REPORTING_LAG_DAYS: int = 75

HORIZONS: tuple[int, ...] = (21, 63, 126)
N_QUINTILES: int = 5
MIN_OBSERVATIONS_FOR_RANK_IC: int = 10

#: Below this many independent as-of dates, statistics are noise-dominated.
MIN_EFFECTIVE_N: int = 30

#: Below this many distinct as-of dates, a bootstrap CI cannot be trusted
#: to reflect between-date variation at all (too few clusters to resample).
MIN_AS_OF_DATES_FOR_CI: int = 3

#: Bootstrap resample count and the RNG seed that makes results reproducible.
BOOTSTRAP_RESAMPLES: int = 2000
BOOTSTRAP_SEED: int = 42

#: Trading days per calendar year, used only to approximate as-of spacing
#: in sessions when no exact trading calendar is available (the
#: fundamentals backtest's as-of dates are not aligned to one shared
#: calendar the way momentum's are -- see _mean_spacing_sessions).
_TRADING_DAYS_PER_CALENDAR_YEAR: float = 252.0 / 365.0

#: The fundamentals backtest's as-of-date count is bounded by real annual
#: filing counts (~3-4 per symbol) and cannot be raised without a vintage
#: fundamentals data source -- unlike momentum, which now uses the full
#: available price history. Surfaced in FundamentalsBacktestRun.note so
#: the asymmetry with momentum's much larger as-of count is not mistaken
#: for an oversight.
FUNDAMENTALS_ASOF_NOTE: str = (
    "Fundamentals as-of dates are bounded by real annual filing counts "
    "(~3 per symbol) and are NOT extended the way momentum's are -- "
    "yfinance keeps no fundamentals vintage archive, so there is no "
    "additional historical data to walk forward through. This is a "
    "structural ceiling, not an oversight."
)


class QuintileStat(BaseModel):
    model_config = ConfigDict(frozen=True)

    quintile: int  # 1 = lowest factor percentile, N_QUINTILES = highest
    mean_forward_return_pct: float
    n: int


class HorizonResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon_sessions: int
    n_observations: int

    # -- Sample-size honesty (Task 1) --------------------------------------
    n_as_of_dates: int
    mean_as_of_spacing_sessions: Optional[float]
    #: Independent observation count, scaled down from n_as_of_dates by how
    #: much consecutive forward-return windows overlap. This, not
    #: n_observations, is the number that determines whether a result means
    #: anything.
    effective_n: int
    overlap_warning: bool

    # -- Point estimates ----------------------------------------------------
    #: Spearman rank correlation between factor percentile and forward
    #: return, computed via rank+Pearson (no scipy dependency). None
    #: below MIN_OBSERVATIONS_FOR_RANK_IC.
    rank_ic: Optional[float]
    rank_ic_ci_95: Optional[tuple[float, float]]
    quintiles: list[QuintileStat]
    q5_minus_q1_pp: Optional[float]
    q5_minus_q1_ci_95: Optional[tuple[float, float]]
    #: True when the Q5-Q1 CI includes zero -- "unknown", not "no edge".
    spans_zero: Optional[bool]

    # -- Monotonicity (Task 3) ----------------------------------------------
    spearman_quintile_monotonicity: Optional[float]
    is_monotonic: Optional[bool]
    #: Biggest adjacent-quintile step divided by the total Q5-Q1 spread.
    #: Near 1.0 means one quintile boundary drives the entire result --
    #: the signal to distrust a large spread as broad-based ordering.
    largest_single_step_share: Optional[float]


class FactorBacktestReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    factor_name: str
    n_observations_total: int
    n_as_of_dates: int
    first_as_of_date: Optional[date]
    last_as_of_date: Optional[date]
    horizons: dict[int, HorizonResult]


def _rank_ic(factor_pct: pd.Series, forward_returns: pd.Series) -> Optional[float]:
    df = pd.DataFrame({"pct": factor_pct, "ret": forward_returns}).dropna()
    if len(df) < MIN_OBSERVATIONS_FOR_RANK_IC:
        return None
    return float(df["pct"].rank().corr(df["ret"].rank()))


def _quintile_stats(factor_pct: pd.Series, forward_returns: pd.Series) -> list[QuintileStat]:
    df = pd.DataFrame({"pct": factor_pct, "ret": forward_returns}).dropna()
    if len(df) < N_QUINTILES:
        return []
    df["quintile"] = pd.qcut(df["pct"], N_QUINTILES, labels=False, duplicates="drop") + 1
    stats = [
        QuintileStat(
            quintile=int(q),  # type: ignore[arg-type]
            mean_forward_return_pct=float(group["ret"].mean()),
            n=len(group),
        )
        for q, group in df.groupby("quintile")
    ]
    return sorted(stats, key=lambda s: s.quintile)


def _monotonicity_stats(
    quintiles: list[QuintileStat],
) -> tuple[Optional[float], Optional[bool], Optional[float]]:
    """Spearman rho of quintile index vs. mean return, whether every step is
    non-decreasing, and the share of the total spread the single largest
    adjacent step accounts for. None for all three below 2 quintiles."""
    if len(quintiles) < 2:
        return None, None, None
    qs = sorted(quintiles, key=lambda q: q.quintile)
    idx = pd.Series([float(q.quintile) for q in qs])
    rets = pd.Series([q.mean_forward_return_pct for q in qs])

    spearman = float(idx.rank().corr(rets.rank()))
    is_monotonic = all(rets[i] <= rets[i + 1] for i in range(len(rets) - 1))

    total_spread = rets.iloc[-1] - rets.iloc[0]
    diffs = [abs(rets[i + 1] - rets[i]) for i in range(len(rets) - 1)]
    largest_step_share = max(diffs) / abs(total_spread) if total_spread != 0 else None

    return spearman, is_monotonic, largest_step_share


def _mean_spacing_sessions(
    as_of_dates: pd.Series, session_calendar: Optional[pd.DatetimeIndex]
) -> Optional[float]:
    """Mean gap between consecutive distinct as-of dates, in trading sessions.

    Exact when ``session_calendar`` is supplied (momentum: as-of dates are
    literal positions in the live screener's own price index). Otherwise
    approximated from calendar-day gaps via a trading-day ratio -- the
    fundamentals backtest pools as-of dates across many symbols with no
    single shared trading calendar to look positions up in, so an exact
    session count is not available there; this is documented, not hidden.
    """
    unique_sorted = sorted(pd.Timestamp(d) for d in set(as_of_dates))
    if len(unique_sorted) < 2:
        return None
    gaps: list[float]
    if session_calendar is not None:
        positions = [int(session_calendar.searchsorted(d)) for d in unique_sorted]
        gaps = [float(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
    else:
        gaps = [
            (unique_sorted[i + 1] - unique_sorted[i]).days * _TRADING_DAYS_PER_CALENDAR_YEAR
            for i in range(len(unique_sorted) - 1)
        ]
    return float(sum(gaps) / len(gaps))


def _independence_factor(mean_spacing_sessions: float, horizon: int) -> float:
    """Fraction of each forward-return window that does not overlap the
    next one, given the mean gap between as-of dates. 1.0 once spacing
    reaches or exceeds the horizon (windows no longer overlap at all)."""
    return min(1.0, mean_spacing_sessions / horizon)


def _effective_n(n_as_of_dates: int, mean_spacing_sessions: Optional[float], horizon: int) -> int:
    """Independent observation count once overlapping forward-return windows
    are accounted for.

    Forward windows from as-of dates closer together than the horizon
    overlap almost completely and are not independent observations --
    scale the as-of date count by the fraction of each window that does
    not overlap the next one.
    """
    if n_as_of_dates <= 0:
        return 0
    if mean_spacing_sessions is None:
        # A single as-of date: nothing to overlap against, but also no
        # basis to claim independence beyond that one date.
        return 1
    factor = _independence_factor(mean_spacing_sessions, horizon)
    return max(1, int(round(n_as_of_dates * factor)))


def _rank_ic_statistic(df: pd.DataFrame, col: str) -> float:
    result = df["factor_pct"].rank().corr(df[col].rank())
    if pd.isna(result):
        raise ValueError("rank IC undefined for this resample (no variance)")
    return float(result)


def _q5_minus_q1_statistic(df: pd.DataFrame, col: str) -> float:
    d = df.copy()
    d["quintile"] = pd.qcut(d["factor_pct"], N_QUINTILES, labels=False, duplicates="drop") + 1
    means = d.groupby("quintile")[col].mean()
    if 1 not in means.index or N_QUINTILES not in means.index:
        raise ValueError("resample did not produce both extreme quintiles")
    return float(means[N_QUINTILES] - means[1])


def _bootstrap_by_as_of_date(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[tuple[float, float]]:
    """Cluster bootstrap: resample whole as-of dates with replacement.

    Resampling individual rows would treat stocks measured on the same day
    as independent draws. They are not -- they share a market regime --
    and doing so would understate the confidence interval substantially,
    the same error effective_n corrects for at the point-estimate level.
    Returns None when fewer than MIN_AS_OF_DATES_FOR_CI distinct as-of
    dates exist, or when too few resamples produced a valid statistic to
    form a reliable interval, rather than emit a fabricated one.
    """
    unique_dates = df["as_of"].unique()
    if len(unique_dates) < MIN_AS_OF_DATES_FOR_CI:
        return None

    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    # Precompute each date's row positions once; every resample then costs
    # one numpy concat + one iloc lookup instead of pd.concat-ing many small
    # per-date frames 2000 times over.
    positions_by_date = {d: g.index.to_numpy() for d, g in df.groupby("as_of")}
    date_list = list(unique_dates)

    stats: list[float] = []
    for _ in range(n_resamples):
        sampled_dates = rng.choice(date_list, size=len(date_list), replace=True)
        positions = np.concatenate([positions_by_date[d] for d in sampled_dates])
        resampled = df.iloc[positions]
        try:
            stats.append(statistic(resampled))
        except (ValueError, ZeroDivisionError):
            continue

    if len(stats) < n_resamples // 2:
        return None
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _build_report(
    factor_name: str,
    df: pd.DataFrame,
    horizons: tuple[int, ...],
    session_calendar: Optional[pd.DatetimeIndex] = None,
) -> FactorBacktestReport:
    """``df`` must have columns ``as_of``, ``symbol``, ``factor_pct``, and
    one ``fwd_<horizon>`` column per horizon -- percentile ranking is the
    caller's responsibility, since momentum and the fundamentals factors
    rank cross-sectionally in different, appropriately-documented ways.

    ``session_calendar``, when supplied, gives exact trading-session
    spacing between as-of dates (see ``_mean_spacing_sessions``).
    """
    n_as_of_dates = int(df["as_of"].nunique()) if not df.empty else 0
    first_as_of = pd.Timestamp(df["as_of"].min()).date() if not df.empty else None
    last_as_of = pd.Timestamp(df["as_of"].max()).date() if not df.empty else None

    horizon_results: dict[int, HorizonResult] = {}
    for h in horizons:
        col = f"fwd_{h}"
        sub = df.dropna(subset=[col]) if not df.empty and col in df.columns else pd.DataFrame()
        if sub.empty:
            horizon_results[h] = HorizonResult(
                horizon_sessions=h,
                n_observations=0,
                n_as_of_dates=0,
                mean_as_of_spacing_sessions=None,
                effective_n=0,
                overlap_warning=True,
                rank_ic=None,
                rank_ic_ci_95=None,
                quintiles=[],
                q5_minus_q1_pp=None,
                q5_minus_q1_ci_95=None,
                spans_zero=None,
                spearman_quintile_monotonicity=None,
                is_monotonic=None,
                largest_single_step_share=None,
            )
            continue

        h_n_as_of_dates = int(sub["as_of"].nunique())
        spacing = _mean_spacing_sessions(sub["as_of"], session_calendar)
        effective_n = _effective_n(h_n_as_of_dates, spacing, h)
        quintiles = _quintile_stats(sub["factor_pct"], sub[col])
        spearman_mono, is_monotonic, largest_step_share = _monotonicity_stats(quintiles)

        q5_minus_q1: Optional[float] = None
        if len(quintiles) >= N_QUINTILES:
            by_q = {q.quintile: q.mean_forward_return_pct for q in quintiles}
            if 1 in by_q and N_QUINTILES in by_q:
                q5_minus_q1 = by_q[N_QUINTILES] - by_q[1]

        rank_ic_ci = _bootstrap_by_as_of_date(sub, lambda d: _rank_ic_statistic(d, col))
        q5_q1_ci = _bootstrap_by_as_of_date(sub, lambda d: _q5_minus_q1_statistic(d, col))
        spans_zero = (q5_q1_ci[0] <= 0 <= q5_q1_ci[1]) if q5_q1_ci is not None else None

        horizon_results[h] = HorizonResult(
            horizon_sessions=h,
            n_observations=len(sub),
            n_as_of_dates=h_n_as_of_dates,
            mean_as_of_spacing_sessions=spacing,
            effective_n=effective_n,
            overlap_warning=effective_n < MIN_EFFECTIVE_N,
            rank_ic=_rank_ic(sub["factor_pct"], sub[col]),
            rank_ic_ci_95=rank_ic_ci,
            quintiles=quintiles,
            q5_minus_q1_pp=q5_minus_q1,
            q5_minus_q1_ci_95=q5_q1_ci,
            spans_zero=spans_zero,
            spearman_quintile_monotonicity=spearman_mono,
            is_monotonic=is_monotonic,
            largest_single_step_share=largest_step_share,
        )

    return FactorBacktestReport(
        factor_name=factor_name,
        n_observations_total=len(df),
        n_as_of_dates=n_as_of_dates,
        first_as_of_date=first_as_of,
        last_as_of_date=last_as_of,
        horizons=horizon_results,
    )


def power_warnings(reports: dict[str, FactorBacktestReport], min_dates: int = 30) -> list[str]:
    """One message per factor whose as-of date count is still below
    ``min_dates`` -- named so a reader does not mistake a flat result for a
    broken factor when the real cause is insufficient sample."""
    return [
        f"{name}: only {report.n_as_of_dates} as-of dates (< {min_dates}); "
        "results likely reflect insufficient sample, not a broken factor"
        for name, report in reports.items()
        if report.n_as_of_dates < min_dates
    ]


def backtest_momentum(
    price_frames: dict[str, pd.DataFrame],
    config: ScreenerConfig,
    as_of_step_sessions: int = 21,
    horizons: tuple[int, ...] = HORIZONS,
) -> FactorBacktestReport:
    """Walk-forward, point-in-time backtest of ``weighted_relative_strength``.

    At each as-of date, only price history up to and including that date
    is used to compute the RS score (the live screener's own function,
    unmodified); forward returns are looked up from that date onward,
    genuinely unknown at computation time. Uses the full history in
    ``price_frames`` -- pass a long period (e.g. ``fetch_price_history(...,
    period="max")``) to maximise the as-of date count; unlike the
    fundamentals backtest, this ceiling is a data-fetch choice, not a
    structural limit.
    """
    closes = pd.DataFrame({s: f["Close"] for s, f in price_frames.items()}).sort_index()
    price_data = {s: f["Close"] for s, f in price_frames.items()}
    # Matches ranking.py's own call exactly: momentum_min_sessions (252),
    # not the unrelated general eligibility floor.
    required = max(config.momentum_min_sessions, 4 * 63 + 1)
    max_horizon = max(horizons)
    dates = closes.index

    rows: list[dict[str, object]] = []
    i = required
    while i + max_horizon < len(dates):
        as_of_ts = dates[i]
        window = closes.iloc[: i + 1]
        valid_cols = [s for s in window.columns if window[s].dropna().shape[0] >= required]
        if len(valid_cols) >= 10:
            try:
                rs = weighted_relative_strength(
                    window[valid_cols], min_sessions=config.momentum_min_sessions
                )
            except ValueError:
                rs = None
            if rs is not None:
                rs_pct = rs.rank(pct=True)
                for symbol in valid_cols:
                    row: dict[str, object] = {
                        "as_of": as_of_ts,
                        "symbol": symbol,
                        "factor_pct": float(rs_pct[symbol]),
                    }
                    for h in horizons:
                        row[f"fwd_{h}"] = _forward_return(price_data, symbol, as_of_ts.date(), h)
                    rows.append(row)
        i += as_of_step_sessions

    return _build_report(
        "weighted_relative_strength",
        pd.DataFrame(rows),
        horizons,
        session_calendar=dates,  # type: ignore[arg-type]
    )


def _fetch_statements(
    symbol: str,
) -> tuple[Optional[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], Optional[str]]:
    """Returns ``(statements, failure_reason)`` -- reason is None on success.

    yfinance failures span genuinely expected cases (a micro-cap with no
    filed statements) and unexpected ones (a network error, a malformed
    response); both must be counted and named, never silently merged
    into "no data", or a real, fixable problem could hide behind an
    excluded-symbol count indistinguishable from ordinary data gaps.
    """
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        income, balance, cashflow = ticker.income_stmt, ticker.balance_sheet, ticker.cashflow
    except Exception as exc:
        return None, f"fetch_error:{type(exc).__name__}"
    if income is None or income.empty:
        return None, "income_stmt_unavailable"
    if balance is None or balance.empty:
        return None, "balance_sheet_unavailable"
    if cashflow is None or cashflow.empty:
        return None, "cashflow_unavailable"
    return (income, balance, cashflow), None


class FundamentalsVintage(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    fdata: FundamentalData
    as_of: date


def _fundamentals_vintages(
    symbol: str, income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame
) -> list[FundamentalsVintage]:
    """Every valid (current, prior) consecutive annual-period pair for one
    symbol, each tagged with its reporting-lag-adjusted as-of date."""
    # yfinance statement columns are period-end Timestamps; stubs type them as str.
    common_periods: list[pd.Timestamp] = sorted(
        set(income.columns) & set(balance.columns) & set(cashflow.columns),  # type: ignore[arg-type]
        reverse=True,
    )
    reported_periods = [p for p in common_periods if not _is_unreported_placeholder(income, p)]

    vintages: list[FundamentalsVintage] = []
    for idx in range(len(reported_periods) - 1):
        current_period, prior_period = reported_periods[idx], reported_periods[idx + 1]
        gap_days = (current_period - prior_period).days
        if not (_ANNUAL_GAP_DAYS_MIN <= gap_days <= _ANNUAL_GAP_DAYS_MAX):
            continue
        try:
            current = _extract_period(symbol, current_period, income, balance, cashflow)
            prior = _extract_period(symbol, prior_period, income, balance, cashflow)
        except InsufficientFundamentalsError:
            continue
        as_of = current_period.date() + timedelta(days=REPORTING_LAG_DAYS)
        vintages.append(
            FundamentalsVintage(fdata=current.model_copy(update={"prior": prior}), as_of=as_of)
        )
    return vintages


class FundamentalsBacktestRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    reports: dict[str, FactorBacktestReport]
    #: symbol -> reason, for every symbol _fetch_statements could not use.
    #: Never merged into one "no data" bucket -- see _fetch_statements.
    fetch_failures: dict[str, str]
    fetch_failure_counts: dict[str, int]
    #: Explains why this backtest's as-of date count is not extended the
    #: way momentum's is -- see FUNDAMENTALS_ASOF_NOTE.
    note: str = FUNDAMENTALS_ASOF_NOTE


def backtest_fundamentals_factors(
    symbols: list[str],
    price_frames: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] = HORIZONS,
    max_workers: int = 16,
) -> FundamentalsBacktestRun:
    """Point-in-time backtest of F-Score, gross profitability, ROE and value.

    Weaker than ``backtest_momentum`` by necessity -- see the module
    docstring. Percentile ranking is pooled across all (symbol, vintage)
    observations globally, not grouped by exact calendar date: with only
    ~3 vintages per symbol and Indian fiscal year-ends clustering tightly
    around March 31, a per-date cross-section would mostly be empty.
    """
    price_data = {s: f["Close"] for s, f in price_frames.items()}

    f_score_rows: list[dict[str, object]] = []
    gp_rows: list[dict[str, object]] = []
    roe_rows: list[dict[str, object]] = []
    ey_rows: list[dict[str, object]] = []
    btp_rows: list[dict[str, object]] = []
    fetch_failures: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_statements, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            statements, failure_reason = future.result()
            if statements is None:
                fetch_failures[symbol] = failure_reason or "unknown"
                continue
            income, balance, cashflow = statements
            close_series = price_data.get(symbol)
            if close_series is None:
                fetch_failures[symbol] = "no_price_history_supplied"
                continue

            vintages = _fundamentals_vintages(symbol, income, balance, cashflow)
            if not vintages:
                # A common, legitimate case (e.g. banks/NBFCs: no classified
                # current/non-current balance sheet, so every vintage fails
                # InsufficientFundamentalsError) -- still counted, not silent.
                fetch_failures[symbol] = "no_valid_vintage"
                continue
            for vintage in vintages:
                fdata = vintage.fdata
                as_of = vintage.as_of
                future_idx = close_series.index[close_series.index >= pd.Timestamp(as_of)]
                if len(future_idx) == 0:
                    continue
                price_at_asof = float(close_series.loc[future_idx[0]])

                def _fwd_row() -> dict[str, object]:
                    row: dict[str, object] = {"as_of": as_of, "symbol": symbol}
                    for h in horizons:
                        row[f"fwd_{h}"] = _forward_return(price_data, symbol, as_of, h)
                    return row

                try:
                    f_score = piotroski_f_score(fdata, fdata.prior)  # type: ignore[arg-type]
                    f_score_rows.append({**_fwd_row(), "value": f_score.total_score})
                except ValueError:
                    pass
                try:
                    gp_rows.append({**_fwd_row(), "value": gross_profitability(fdata)})
                except ValueError:
                    pass
                try:
                    roe_rows.append({**_fwd_row(), "value": return_on_equity(fdata)})
                except ValueError:
                    pass
                try:
                    ey_rows.append({**_fwd_row(), "value": earnings_yield(fdata, price_at_asof)})
                except ValueError:
                    pass
                try:
                    btp_rows.append({**_fwd_row(), "value": book_to_price(fdata, price_at_asof)})
                except ValueError:
                    pass

    def _to_report(name: str, rows: list[dict[str, object]]) -> FactorBacktestReport:
        df = pd.DataFrame(rows)
        if not df.empty:
            df["factor_pct"] = df["value"].rank(pct=True)
        return _build_report(name, df, horizons)

    reports = {
        "f_score": _to_report("piotroski_f_score", f_score_rows),
        "gross_profitability": _to_report("gross_profitability", gp_rows),
        "roe": _to_report("return_on_equity", roe_rows),
        "earnings_yield": _to_report("earnings_yield", ey_rows),
        "book_to_price": _to_report("book_to_price", btp_rows),
    }
    failure_counts: dict[str, int] = {}
    for reason in fetch_failures.values():
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    return FundamentalsBacktestRun(
        reports=reports, fetch_failures=fetch_failures, fetch_failure_counts=failure_counts
    )


# -- ROE leverage diagnostic (Task 5) -----------------------------------------
#
# ROE reports a positive rank IC at every horizon but a negative, worsening
# Q5-Q1 spread -- a contradiction that a non-monotonic relationship explains:
# ROE is mechanically inflated by leverage, so the top quintile is plausibly
# split between genuinely high-return businesses and merely high-debt ones.
# This is diagnostic only -- it does not change the factor or its weight.


class LeverageGroupResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    group: str  # "high_leverage" or "low_leverage"
    n: int
    mean_forward_return_pct: dict[int, float]  # horizon -> mean return
    mean_debt_to_equity: float
    mean_roic: Optional[float]  # None when ROIC was not computable for any member


class ROEDiagnosticReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizons: tuple[int, ...]
    high_leverage: LeverageGroupResult
    low_leverage: LeverageGroupResult
    #: symbol -> reason, for top-quintile members excluded from the split
    #: (e.g. non-positive equity makes debt-to-equity undefined).
    excluded: dict[str, str]
    #: ROIC is computed from each symbol's current (most recent) filing via
    #: fetch_live_financials, not the historical vintage the ROE split used
    #: -- a live cross-reference, not a per-vintage match. Building
    #: historical ROIC per vintage would need the same extra-field
    #: extraction live_data.py already duplicates data.py for, across
    #: every vintage; out of scope for a diagnostic.
    roic_note: str = (
        "ROIC is each symbol's current filing via fetch_live_financials, "
        "not the historical vintage used for the ROE split -- a live "
        "cross-reference, not a per-vintage match."
    )


def diagnose_roe_leverage(
    symbols: list[str],
    price_frames: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] = HORIZONS,
    fetch_roic: bool = True,
) -> ROEDiagnosticReport:
    """Split the ROE top quintile by debt-to-equity and compare forward
    returns for the high- and low-leverage halves, plus each symbol's
    current ROIC where computable -- see the module-level note above."""
    price_data = {s: f["Close"] for s, f in price_frames.items()}
    rows: list[dict[str, object]] = []
    excluded: dict[str, str] = {}

    # Rebuild the raw (symbol, as_of, roe, d/e) rows the same way
    # backtest_fundamentals_factors did, since it does not expose them.
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_fetch_statements, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            statements, _ = future.result()
            if statements is None:
                continue
            income, balance, cashflow = statements
            close_series = price_data.get(symbol)
            if close_series is None:
                continue
            for vintage in _fundamentals_vintages(symbol, income, balance, cashflow):
                fdata = vintage.fdata
                as_of = vintage.as_of
                try:
                    roe = return_on_equity(fdata)
                except ValueError:
                    continue
                if fdata.total_equity <= 0:
                    excluded[symbol] = "equity_non_positive"
                    continue
                row: dict[str, object] = {
                    "symbol": symbol,
                    "as_of": as_of,
                    "roe": roe,
                    "debt_to_equity": fdata.total_debt / fdata.total_equity,
                }
                for h in horizons:
                    row[f"fwd_{h}"] = _forward_return(price_data, symbol, as_of, h)
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no ROE observations available to diagnose")

    df["roe_pct"] = df["roe"].rank(pct=True)
    top_quintile = df[df["roe_pct"] >= 0.8].copy()
    if top_quintile.empty:
        raise ValueError("top ROE quintile is empty -- insufficient observations")

    de_median = top_quintile["debt_to_equity"].median()
    top_quintile["leverage_group"] = np.where(
        top_quintile["debt_to_equity"] >= de_median, "high_leverage", "low_leverage"
    )

    roic_by_symbol: dict[str, Optional[float]] = {}
    if fetch_roic:
        roic_by_symbol = _fetch_current_roic(list(top_quintile["symbol"].unique()))

    def _group_result(group_name: str) -> LeverageGroupResult:
        group_df = top_quintile[top_quintile["leverage_group"] == group_name]
        returns = {h: float(group_df[f"fwd_{h}"].dropna().mean()) for h in horizons}
        roic_values: list[float] = [
            value
            for s in group_df["symbol"].unique()
            if (value := roic_by_symbol.get(s)) is not None
        ]
        mean_roic = float(np.mean(roic_values)) if roic_values else None
        return LeverageGroupResult(
            group=group_name,
            n=len(group_df),
            mean_forward_return_pct=returns,
            mean_debt_to_equity=float(group_df["debt_to_equity"].mean()),
            mean_roic=mean_roic,
        )

    return ROEDiagnosticReport(
        horizons=horizons,
        high_leverage=_group_result("high_leverage"),
        low_leverage=_group_result("low_leverage"),
        excluded=excluded,
    )


def _fetch_current_roic(symbols: list[str]) -> dict[str, Optional[float]]:
    """Each symbol's current ROIC via the live workbench pipeline, best
    effort -- a symbol failing anywhere in fetch/reorganize/WACC/ROIC is
    recorded as None (not computable), never silently dropped from the
    result the caller sees, since the caller reports mean_roic as None
    when no member of a group has a value."""
    from nse_screener.fundamentals.live_data import LiveDataError, fetch_live_financials
    from nse_screener.fundamentals.reorganize import ReorganizationError, reorganize
    from nse_screener.fundamentals.roic import DecompositionError, roic_analysis
    from nse_screener.fundamentals.wacc import (
        InsufficientReturnHistoryError,
        WACCInputs,
        compute_wacc,
    )

    results: dict[str, Optional[float]] = {}
    for symbol in symbols:
        try:
            price_frames, _ = fetch_price_history([symbol], period="2y")
            index_series = price_frames["^NSEI"]["Close"]
            price_series = price_frames[symbol]["Close"]
            lf = fetch_live_financials(symbol)
            stmts = reorganize(lf.raw_operating)
            wacc_inputs = WACCInputs(
                symbol=symbol,
                market_cap=lf.close_price * lf.shares_outstanding,
                total_debt=lf.total_debt,
                prior_total_debt=lf.prior_total_debt,
                interest_expense=lf.interest_expense,
                effective_tax_rate=stmts.effective_tax_rate,
            )
            wacc = compute_wacc(
                wacc_inputs, price_series, index_series, risk_free_rate=0.07,
                equity_risk_premium=0.06,
            )
            roic = roic_analysis(stmts, wacc)
            results[symbol] = roic.roic
        except (
            LiveDataError,
            ReorganizationError,
            DecompositionError,
            InsufficientReturnHistoryError,
            ValueError,
            KeyError,
        ):
            results[symbol] = None
    return results


def _run_cli(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="nse_screener.factor_backtest")
    parser.add_argument("--universe", default="data/nifty500_universe.txt")
    parser.add_argument("--diagnose-roe", action="store_true")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in Path(args.universe).read_text().splitlines() if s.strip()]
    price_frames, _ = fetch_price_history(symbols, period="max")
    price_frames.pop("^NSEI", None)

    if args.diagnose_roe:
        report = diagnose_roe_leverage(symbols, price_frames)
        run_date = date.today().isoformat()
        out_path = Path(args.out_dir) / f"roe_diagnostic_{run_date}.json"
        out_path.write_text(report.model_dump_json(indent=2))
        print(f"ROE leverage diagnostic written to {out_path}")
        return

    config = ScreenerConfig()
    momentum_report = backtest_momentum(price_frames, config)
    fundamentals_run = backtest_fundamentals_factors(symbols, price_frames)

    warnings = power_warnings({"momentum": momentum_report, **fundamentals_run.reports})
    out = {
        "run_date": date.today().isoformat(),
        "universe_size": len(symbols),
        "momentum": json.loads(momentum_report.model_dump_json()),
        "fundamentals": {
            k: json.loads(v.model_dump_json()) for k, v in fundamentals_run.reports.items()
        },
        "fundamentals_note": fundamentals_run.note,
        "fetch_failure_counts": fundamentals_run.fetch_failure_counts,
        "power_warning": warnings,
    }
    run_date = date.today().isoformat()
    out_path = Path(args.out_dir) / f"factor_backtest_{run_date}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Backtest written to {out_path}")


if __name__ == "__main__":
    _run_cli()
