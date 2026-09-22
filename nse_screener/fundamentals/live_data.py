"""Live yfinance adapter for the fields fundamentals/workbench need beyond
``data.FundamentalData``.

Reuses ``data.fetch_fundamentals`` for every field it already covers
(revenue, gross profit, net income, assets, equity, debt, current
assets/liabilities, receivables, inventory, operating cash flow) rather
than re-deriving them independently -- that function already handles
inventory-absent-means-zero, the fiscal-year-gap check, and skipping
unreported placeholder periods. An earlier version of this module
duplicated that extraction logic with a weaker version that had none of
those fixes, which silently cost real coverage (INFY, NESTLEIND) for no
reason: the two statements should never have drifted apart.

Only the fields ``FundamentalData`` doesn't carry are fetched separately
here, from the exact same two periods ``fetch_fundamentals`` already
settled on (never a different period pair -- that would make the
combined record internally inconsistent).

Verified against real yfinance output for RELIANCE.NS and TCS.NS before
writing the field mapping for those extra fields (the same evidence-first
approach as every other data decision in this project).

Three approximations remain, because yfinance does not separately report
the underlying line item for any NSE company observed:

- ``amortization_of_intangibles`` is always 0, so EBITA collapses to
  EBIT exactly (yfinance reports EBIT directly; acquisition-related
  intangible amortization is not broken out separately for Indian
  filers).
- ``net_other_operating_assets`` is always 0 -- invested capital is
  therefore operating working capital plus net PPE only, understating
  invested capital by whatever non-PPE operating assets a company
  carries (a real simplification, not a rounding error).
- ``income_continuing_ops`` falls back to net income when income from
  continuing operations is not reported as a separate line.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, ConfigDict

from nse_screener.data import fetch_fundamentals
from nse_screener.fundamentals.distress import DistressInputs
from nse_screener.fundamentals.multiples import MultiplesInputs
from nse_screener.fundamentals.reorganize import RawOperatingFinancials
from nse_screener.workbench.mscore import MScoreInputs


class LiveDataError(Exception):
    """Raised when a required field cannot be found in yfinance's statements."""

    def __init__(self, symbol: str, reason: str) -> None:
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"{symbol}: {reason}")


class LiveFinancials(BaseModel):
    """Everything the fundamentals/workbench modules need for one symbol,
    fetched once and shared -- avoids refetching the same statements
    per downstream input model."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    close_price: float
    shares_outstanding: float
    raw_operating: RawOperatingFinancials
    mscore_inputs: MScoreInputs
    distress_inputs: DistressInputs
    multiples_inputs: MultiplesInputs
    interest_expense: float
    total_debt: float
    prior_total_debt: float
    net_debt: float


#: Fields not on data.FundamentalData -- fetched directly from yfinance,
#: from the exact periods fetch_fundamentals already selected.
_EXTRA_INCOME_FIELDS: dict[str, list[str]] = {
    "operating_income": ["EBIT", "Operating Income"],
    "ebitda": ["EBITDA"],
    "pretax_income": ["Pretax Income"],
    "tax_expense": ["Tax Provision"],
    "interest_expense": ["Interest Expense"],
    "sga_expense": ["Selling General And Administration"],
    "income_continuing_ops": [
        "Net Income From Continuing Operation Net Minority Interest",
        "Net Income",
    ],
    "depreciation_expense": ["Reconciled Depreciation"],
}
_EXTRA_BALANCE_FIELDS: dict[str, list[str]] = {
    "net_ppe": ["Net PPE"],
    "cash_and_equivalents": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    ],
    "short_term_debt": ["Current Debt", "Current Debt And Capital Lease Obligation"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "retained_earnings": ["Retained Earnings"],
}


def fetch_live_financials(symbol: str) -> LiveFinancials:
    """Fetch and assemble every input model the fundamentals/workbench
    modules need for ``symbol``, in one pass.

    Raises ``LiveDataError`` (wrapping ``InsufficientFundamentalsError``
    from ``data.fetch_fundamentals`` for the fields it covers) naming the
    missing field if any required line item is absent from the two most
    recent common annual periods.
    """
    from nse_screener.data import InsufficientFundamentalsError

    try:
        fdata = fetch_fundamentals(symbol)
    except InsufficientFundamentalsError as exc:
        raise LiveDataError(symbol, str(exc)) from exc
    if fdata.prior is None:
        raise LiveDataError(symbol, "fewer than 2 years of fundamentals available")

    ticker = yf.Ticker(f"{symbol}.NS")
    income, balance, cashflow = ticker.income_stmt, ticker.balance_sheet, ticker.cashflow
    if income is None or income.empty:
        raise LiveDataError(symbol, "income_stmt unavailable")
    if balance is None or balance.empty:
        raise LiveDataError(symbol, "balance_sheet unavailable")
    if cashflow is None or cashflow.empty:
        raise LiveDataError(symbol, "cashflow unavailable")

    current_period = _match_period(symbol, fdata.period_end, income)
    prior_period = _match_period(symbol, fdata.prior.period_end, income)
    current_extra = _extract_extra_period(symbol, current_period, income, balance, cashflow)
    prior_extra = _extract_extra_period(symbol, prior_period, income, balance, cashflow)

    history = ticker.history(period="5d")
    if history is None or history.empty:
        raise LiveDataError(symbol, "no recent price history for close/market cap")
    close_price = float(history["Close"].iloc[-1])
    market_cap = close_price * fdata.shares_outstanding
    net_debt = fdata.total_debt - current_extra["cash_and_equivalents"]

    raw_operating = RawOperatingFinancials(
        symbol=symbol,
        period_end=fdata.period_end,
        revenue=fdata.total_revenue,
        operating_income=current_extra["operating_income"],
        amortization_of_intangibles=0.0,
        tax_expense=current_extra["tax_expense"],
        pretax_income=current_extra["pretax_income"],
        current_assets=fdata.current_assets,
        current_liabilities=fdata.current_liabilities,
        cash_and_equivalents=current_extra["cash_and_equivalents"],
        short_term_debt=current_extra["short_term_debt"],
        net_ppe=current_extra["net_ppe"],
        net_other_operating_assets=0.0,
        prior=RawOperatingFinancials(
            symbol=symbol,
            period_end=fdata.prior.period_end,
            revenue=fdata.prior.total_revenue,
            operating_income=prior_extra["operating_income"],
            amortization_of_intangibles=0.0,
            tax_expense=prior_extra["tax_expense"],
            pretax_income=prior_extra["pretax_income"],
            current_assets=fdata.prior.current_assets,
            current_liabilities=fdata.prior.current_liabilities,
            cash_and_equivalents=prior_extra["cash_and_equivalents"],
            short_term_debt=prior_extra["short_term_debt"],
            net_ppe=prior_extra["net_ppe"],
            net_other_operating_assets=0.0,
        ),
    )

    mscore_inputs = MScoreInputs(
        symbol=symbol,
        period_end=fdata.period_end,
        revenue=fdata.total_revenue,
        gross_profit=fdata.gross_profit,
        accounts_receivable=fdata.accounts_receivable,
        current_assets=fdata.current_assets,
        net_ppe=current_extra["net_ppe"],
        total_assets=fdata.total_assets,
        depreciation_expense=current_extra["depreciation_expense"],
        sga_expense=current_extra["sga_expense"],
        total_debt=fdata.total_debt,
        income_continuing_ops=current_extra["income_continuing_ops"],
        operating_cash_flow=fdata.operating_cash_flow,
        prior=MScoreInputs(
            symbol=symbol,
            period_end=fdata.prior.period_end,
            revenue=fdata.prior.total_revenue,
            gross_profit=fdata.prior.gross_profit,
            accounts_receivable=fdata.prior.accounts_receivable,
            current_assets=fdata.prior.current_assets,
            net_ppe=prior_extra["net_ppe"],
            total_assets=fdata.prior.total_assets,
            depreciation_expense=prior_extra["depreciation_expense"],
            sga_expense=prior_extra["sga_expense"],
            total_debt=fdata.prior.total_debt,
            income_continuing_ops=prior_extra["income_continuing_ops"],
            operating_cash_flow=fdata.prior.operating_cash_flow,
        ),
    )

    distress_inputs = DistressInputs(
        symbol=symbol,
        working_capital=fdata.current_assets - fdata.current_liabilities,
        total_assets=fdata.total_assets,
        retained_earnings=current_extra["retained_earnings"],
        ebit=current_extra["operating_income"],
        market_cap=market_cap,
        total_liabilities=current_extra["total_liabilities"],
        revenue=fdata.total_revenue,
        interest_expense=current_extra["interest_expense"],
        net_debt=net_debt,
        ebitda=current_extra["ebitda"],
        current_assets=fdata.current_assets,
        current_liabilities=fdata.current_liabilities,
        inventory=fdata.inventory,
        cash_and_equivalents=current_extra["cash_and_equivalents"],
    )

    multiples_inputs = MultiplesInputs(
        symbol=symbol,
        net_income=fdata.net_income,
        book_value=fdata.total_equity,
        ebitda=current_extra["ebitda"],
        ebit=current_extra["operating_income"],
        revenue=fdata.total_revenue,
        dividends_paid=current_extra["dividends_paid"],
        net_buybacks=current_extra["net_buybacks"],
        shares_outstanding=fdata.shares_outstanding,
    )

    return LiveFinancials(
        symbol=symbol,
        close_price=close_price,
        shares_outstanding=fdata.shares_outstanding,
        raw_operating=raw_operating,
        mscore_inputs=mscore_inputs,
        distress_inputs=distress_inputs,
        multiples_inputs=multiples_inputs,
        interest_expense=current_extra["interest_expense"],
        total_debt=fdata.total_debt,
        prior_total_debt=fdata.prior.total_debt,
        net_debt=net_debt,
    )


def _match_period(symbol: str, target: date, income: pd.DataFrame) -> pd.Timestamp:
    """Find the income-statement column matching the period fetch_fundamentals
    already selected -- balance/cashflow are looked up with the same
    Timestamp, since fetch_fundamentals only accepts periods common to
    all three statements."""
    for period in income.columns:  # yfinance columns are Timestamps; stubs type them as str.
        if period.date() == target:  # type: ignore[attr-defined]
            return period  # type: ignore[return-value]
    raise LiveDataError(symbol, f"no income statement period matching {target} found")


def _extract_extra_period(
    symbol: str,
    period: pd.Timestamp,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> dict[str, float]:
    values: dict[str, float] = {}
    missing: list[str] = []
    for field, labels in _EXTRA_INCOME_FIELDS.items():
        value = _lookup(income, period, labels)
        if value is None:
            missing.append(field)
        else:
            values[field] = value
    for field, labels in _EXTRA_BALANCE_FIELDS.items():
        value = _lookup(balance, period, labels)
        if value is None:
            missing.append(field)
        else:
            values[field] = value
    if missing:
        raise LiveDataError(
            symbol, f"missing required field(s) for {period.date()}: {', '.join(missing)}"
        )

    # A company that did no buybacks/paid no dividend that period has no
    # such row at all -- absence is read as zero, not as missing data,
    # same convention as data.fetch_fundamentals uses for inventory.
    values["dividends_paid"] = abs(_lookup_or_zero(cashflow, period, ["Cash Dividends Paid"]))
    repurchases = abs(_lookup_or_zero(cashflow, period, ["Repurchase Of Capital Stock"]))
    issuance = _lookup_or_zero(cashflow, period, ["Issuance Of Capital Stock"])
    values["net_buybacks"] = repurchases - issuance

    return values


def _lookup(frame: pd.DataFrame, period: pd.Timestamp, labels: list[str]) -> Optional[float]:
    if period not in frame.columns:
        return None
    for label in labels:
        if label in frame.index:
            value = frame.loc[label, period]  # type: ignore[index]
            if pd.notna(value):
                return float(value)
    return None


def _lookup_or_zero(frame: pd.DataFrame, period: pd.Timestamp, labels: list[str]) -> float:
    value = _lookup(frame, period, labels)
    return value if value is not None else 0.0
