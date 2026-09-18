"""Data acquisition and integrity validation for NSE price and fundamental data.

``yfinance`` is an unofficial scrape of a public endpoint, not an official
NSE or exchange data feed. It has been observed to silently drop trading
sessions for some symbols on some days (verified 2026-09-17: 9 of 12
sampled NSE tickers had a given session, 3 did not). All price data that
flows through this module is therefore validated before any downstream
factor computation is allowed to touch it -- see ``validate_price_data``.

Fundamental data quality for Indian companies via ``yfinance`` is worse
still: coverage is inconsistent, restatements are not always reflected,
and some mid-cap/small-cap names have incomplete statements entirely.
``fetch_fundamentals`` fails fast (raises) on any missing required field
rather than imputing or defaulting -- a screen built on invented numbers
is worse than no screen. The primary planned upgrade for this module is
to replace it with an adapter over NSE Bhavcopy (prices) and filed
financial statements (fundamentals); see README.md.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, ConfigDict


class DataIntegrityError(Exception):
    """Raised when price data fails a hard integrity check."""


class InsufficientFundamentalsError(Exception):
    """Raised when a symbol's fundamental statements are missing required fields."""

    def __init__(self, symbol: str, missing_fields: list[str]) -> None:
        self.symbol = symbol
        self.missing_fields = missing_fields
        super().__init__(
            f"{symbol}: missing required fundamental fields: {', '.join(missing_fields)}"
        )


class ValidationReport(BaseModel):
    """Outcome of ``validate_price_data`` for one universe of symbols."""

    model_config = ConfigDict(frozen=True)

    accepted: list[str]
    rejected: dict[str, str]
    flagged: list[tuple[str, date, float]]


class FundamentalData(BaseModel):
    """Annual fundamental snapshot for one fiscal period.

    ``prior`` holds the immediately preceding fiscal period's snapshot
    (with its own ``prior`` left unset), so that a single
    ``fetch_fundamentals`` call carries the two years of history required
    by the Piotroski F-Score and forensic screen.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    period_end: date
    total_revenue: float
    gross_profit: float
    net_income: float
    operating_cash_flow: float
    total_assets: float
    total_equity: float
    total_debt: float
    current_assets: float
    current_liabilities: float
    shares_outstanding: float
    accounts_receivable: float
    inventory: float
    prior: Optional["FundamentalData"] = None

    @property
    def cost_of_goods_sold(self) -> float:
        """Derived: revenue less gross profit. Not a separately fetched field."""
        return self.total_revenue - self.gross_profit


FundamentalData.model_rebuild()


def fetch_price_history(
    symbols: list[str], period: str = "3y", index_symbol: str = "^NSEI"
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Download daily OHLCV history for ``symbols`` plus the index.

    Returns ``(frames, unavailable)``: ``frames`` is keyed by the bare
    symbol (no ``.NS`` suffix) for each universe member with data, plus
    one entry keyed by ``index_symbol`` for the market gate; ``unavailable``
    lists symbols yfinance returned zero data for (delisted, renamed, or
    otherwise not resolvable) -- named explicitly rather than silently
    dropped, but not a fatal error, since a stale ticker among many is an
    expected, ordinary occurrence rather than a data-integrity bug.

    The index itself is mandatory: raises ``DataIntegrityError`` if no
    data comes back for it (the market gate cannot run without it), or if
    literally none of the requested symbols returned data.
    """
    if not symbols:
        raise ValueError("symbols must be a non-empty list")

    ns_symbols = [f"{symbol}.NS" for symbol in symbols]
    all_symbols = ns_symbols + [index_symbol]

    raw = yf.download(
        all_symbols,
        period=period,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise DataIntegrityError(
            f"yfinance returned no data for requested symbols: {all_symbols}"
        )

    frames: dict[str, pd.DataFrame] = {}
    unavailable: list[str] = []
    for bare_symbol, ns_symbol in zip(symbols, ns_symbols):
        frame = _extract_ticker_frame(raw, ns_symbol, len(all_symbols))
        if frame is None or frame.dropna(how="all").empty:
            unavailable.append(bare_symbol)
            continue
        frames[bare_symbol] = frame.dropna(how="all")

    index_frame = _extract_ticker_frame(raw, index_symbol, len(all_symbols))
    if index_frame is None or index_frame.dropna(how="all").empty:
        raise DataIntegrityError(
            f"no price data returned for index {index_symbol} -- the market "
            "gate cannot run without it"
        )
    frames[index_symbol] = index_frame.dropna(how="all")

    if len(frames) == 1:  # only the index came back
        raise DataIntegrityError(
            f"no price data returned for any of the {len(symbols)} requested symbols"
        )

    return frames, unavailable


def _extract_ticker_frame(
    raw: pd.DataFrame, symbol: str, n_symbols: int
) -> Optional[pd.DataFrame]:
    if n_symbols == 1:
        return raw
    if symbol not in raw.columns.get_level_values(0):
        return None
    return raw[symbol]


def validate_price_data(
    frames: dict[str, pd.DataFrame],
    index_frame: pd.DataFrame,
    min_sessions: int = 300,
    calendar_alignment_lookback: int = 252,
    max_missing_calendar_fraction: float = 0.02,
    max_recency_gap_days: int = 5,
    impossible_move_abs_return: float = 0.50,
) -> ValidationReport:
    """Validate price data before any factor computation is allowed to run.

    Raises ``DataIntegrityError`` immediately for structural bugs (a
    duplicated index, indicating a bad merge somewhere upstream) or for a
    universe that ends up with zero accepted symbols. Everything else --
    thin history, calendar gaps, stale data, non-positive prices -- is a
    per-symbol soft rejection recorded in the returned report rather than
    a raised exception, so that one bad symbol does not abort the whole
    run. Suspicious single-session moves are flagged, not rejected.
    """
    if index_frame.index.duplicated().any():
        raise DataIntegrityError(
            "duplicate index entries in the reference index frame -- "
            "this indicates a data merge bug upstream"
        )
    for symbol, frame in frames.items():
        if frame.index.duplicated().any():
            raise DataIntegrityError(
                f"duplicate index entries in price frame for {symbol} -- "
                "this indicates a data merge bug upstream"
            )

    rejected: dict[str, str] = {}
    flagged: list[tuple[str, date, float]] = []

    reference_calendar = set(
        index_frame.sort_index().index[-calendar_alignment_lookback:]
    )
    last_index_date = index_frame.sort_index().index[-1]

    for symbol, frame in frames.items():
        frame = frame.sort_index()

        if len(frame) < min_sessions:
            rejected[symbol] = (
                f"only {len(frame)} sessions of history (< {min_sessions} required)"
            )
            continue

        overlap = len(set(frame.index) & reference_calendar)
        missing_fraction = 1.0 - (overlap / len(reference_calendar))
        if missing_fraction > max_missing_calendar_fraction:
            rejected[symbol] = (
                f"missing {missing_fraction:.1%} of reference trading calendar "
                f"in trailing {calendar_alignment_lookback} sessions "
                f"(> {max_missing_calendar_fraction:.1%} threshold)"
            )
            continue

        last_symbol_date = frame.index[-1]
        gap_days = (last_index_date - last_symbol_date).days
        if gap_days > max_recency_gap_days:
            rejected[symbol] = (
                f"last session {last_symbol_date.date()} is {gap_days} calendar "
                f"days behind index last session {last_index_date.date()} "
                f"(> {max_recency_gap_days} day threshold)"
            )
            continue

        if (frame["Close"] <= 0).any():
            rejected[symbol] = "one or more non-positive close prices"
            continue

        abs_returns = frame["Close"].pct_change().abs()
        moves = abs_returns[abs_returns > impossible_move_abs_return]
        for move_date, move_value in moves.items():
            flagged.append((symbol, move_date.date(), float(move_value)))

    accepted = [symbol for symbol in frames if symbol not in rejected]
    if not accepted:
        raise DataIntegrityError(
            f"all {len(frames)} symbols failed integrity validation: {rejected}"
        )

    return ValidationReport(accepted=accepted, rejected=rejected, flagged=flagged)


_INCOME_FIELDS: dict[str, list[str]] = {
    "total_revenue": ["Total Revenue"],
    "gross_profit": ["Gross Profit"],
    "net_income": ["Net Income"],
}
_BALANCE_FIELDS: dict[str, list[str]] = {
    "total_assets": ["Total Assets"],
    "total_equity": [
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
    ],
    "total_debt": ["Total Debt"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
    "accounts_receivable": ["Receivables", "Accounts Receivable"],
    "inventory": ["Inventory"],
}
_CASHFLOW_FIELDS: dict[str, list[str]] = {
    "operating_cash_flow": [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
    ],
}


def fetch_fundamentals(symbol: str) -> FundamentalData:
    """Fetch two annual fiscal periods of fundamentals for ``symbol``.

    Returns the most recent period as a ``FundamentalData``, with the
    prior period attached via ``.prior``. Raises
    ``InsufficientFundamentalsError`` if fewer than two annual periods are
    available, or if either period is missing a required field -- no
    field is ever imputed or defaulted.
    """
    ticker = yf.Ticker(f"{symbol}.NS")
    income = ticker.income_stmt
    balance = ticker.balance_sheet
    cashflow = ticker.cashflow

    if income is None or income.empty:
        raise InsufficientFundamentalsError(symbol, ["income_stmt unavailable"])
    if balance is None or balance.empty:
        raise InsufficientFundamentalsError(symbol, ["balance_sheet unavailable"])
    if cashflow is None or cashflow.empty:
        raise InsufficientFundamentalsError(symbol, ["cashflow unavailable"])

    common_periods = sorted(
        set(income.columns) & set(balance.columns) & set(cashflow.columns),
        reverse=True,
    )
    if len(common_periods) < 2:
        raise InsufficientFundamentalsError(
            symbol, ["fewer than 2 annual periods common to all three statements"]
        )

    current = _extract_period(
        symbol, common_periods[0], income, balance, cashflow
    )
    prior = _extract_period(symbol, common_periods[1], income, balance, cashflow)
    return current.model_copy(update={"prior": prior})


def _extract_period(
    symbol: str,
    period: pd.Timestamp,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> FundamentalData:
    frames_by_field: dict[str, tuple[pd.DataFrame, list[str]]] = {
        **{field: (income, labels) for field, labels in _INCOME_FIELDS.items()},
        **{field: (balance, labels) for field, labels in _BALANCE_FIELDS.items()},
        **{field: (cashflow, labels) for field, labels in _CASHFLOW_FIELDS.items()},
    }

    values: dict[str, float] = {}
    missing: list[str] = []
    for field, (frame, labels) in frames_by_field.items():
        value = _lookup(frame, period, labels)
        if value is None:
            missing.append(field)
        else:
            values[field] = value

    if missing:
        raise InsufficientFundamentalsError(symbol, missing)

    return FundamentalData(symbol=symbol, period_end=period.date(), **values)


def _lookup(
    frame: pd.DataFrame, period: pd.Timestamp, labels: list[str]
) -> Optional[float]:
    if period not in frame.columns:
        return None
    for label in labels:
        if label in frame.index:
            value = frame.loc[label, period]
            if pd.notna(value):
                return float(value)
    return None
