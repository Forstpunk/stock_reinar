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
observations per symbol rather than continuous coverage -- report it as
a real but much smaller-sample, weaker-resolution result than the
momentum backtest.

Both reuse the exact factor functions the live screener runs and the
forward-return lookup already validated in
``experimental.validation_log`` -- no new return-computation logic to
risk a second lookahead bug like the one found and fixed in
``experimental/busted.py``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Optional

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


class QuintileStat(BaseModel):
    model_config = ConfigDict(frozen=True)

    quintile: int  # 1 = lowest factor percentile, N_QUINTILES = highest
    mean_forward_return_pct: float
    n: int


class HorizonResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon_sessions: int
    n_observations: int
    #: Spearman rank correlation between factor percentile and forward
    #: return, computed via rank+Pearson (no scipy dependency). None
    #: below MIN_OBSERVATIONS_FOR_RANK_IC.
    rank_ic: Optional[float]
    quintiles: list[QuintileStat]


class FactorBacktestReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    factor_name: str
    n_observations_total: int
    n_as_of_dates: int
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


def _build_report(
    factor_name: str, df: pd.DataFrame, horizons: tuple[int, ...]
) -> FactorBacktestReport:
    """``df`` must have columns ``as_of``, ``symbol``, ``factor_pct``, and
    one ``fwd_<horizon>`` column per horizon -- percentile ranking is the
    caller's responsibility, since momentum and the fundamentals factors
    rank cross-sectionally in different, appropriately-documented ways."""
    n_as_of_dates = int(df["as_of"].nunique()) if not df.empty else 0
    horizon_results: dict[int, HorizonResult] = {}
    for h in horizons:
        col = f"fwd_{h}"
        sub = df.dropna(subset=[col]) if not df.empty and col in df.columns else pd.DataFrame()
        horizon_results[h] = HorizonResult(
            horizon_sessions=h,
            n_observations=len(sub),
            rank_ic=_rank_ic(sub["factor_pct"], sub[col]) if not sub.empty else None,
            quintiles=_quintile_stats(sub["factor_pct"], sub[col]) if not sub.empty else [],
        )
    return FactorBacktestReport(
        factor_name=factor_name,
        n_observations_total=len(df),
        n_as_of_dates=n_as_of_dates,
        horizons=horizon_results,
    )


def backtest_momentum(
    price_frames: dict[str, pd.DataFrame],
    config: ScreenerConfig,
    step_sessions: int = 21,
    horizons: tuple[int, ...] = HORIZONS,
) -> FactorBacktestReport:
    """Walk-forward, point-in-time backtest of ``weighted_relative_strength``.

    At each as-of date, only price history up to and including that date
    is used to compute the RS score (the live screener's own function,
    unmodified); forward returns are looked up from that date onward,
    genuinely unknown at computation time.
    """
    closes = pd.DataFrame({s: f["Close"] for s, f in price_frames.items()}).sort_index()
    price_data = {s: f["Close"] for s, f in price_frames.items()}
    required = max(config.min_sessions, 4 * 63 + 1)
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
                    window[valid_cols], min_sessions=config.min_sessions
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
        i += step_sessions

    return _build_report("weighted_relative_strength", pd.DataFrame(rows), horizons)


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
