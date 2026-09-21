"""Live yfinance adapter for the fields fundamentals/workbench need beyond
``data.FundamentalData``.

Verified against real yfinance output for RELIANCE.NS and TCS.NS before
writing any field mapping (the same evidence-first approach as every
other data decision in this project) -- see the field lists below.

Three approximations are made because yfinance does not separately
report the underlying line item for any NSE company observed:

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

This module does not do the fiscal-year-gap or unreported-placeholder
detection that ``data.fetch_fundamentals`` does -- it takes the two most
recent common annual columns as-is. That is a known, smaller-scope gap
against the live-data pipeline; the computed modules downstream still
fail fast on genuinely missing fields.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, ConfigDict

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


_INCOME_FIELDS: dict[str, list[str]] = {
    "revenue": ["Total Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["EBIT", "Operating Income"],
    "ebitda": ["EBITDA"],
    "pretax_income": ["Pretax Income"],
    "tax_expense": ["Tax Provision"],
    "interest_expense": ["Interest Expense"],
    "sga_expense": ["Selling General And Administration"],
    "net_income": ["Net Income"],
    "income_continuing_ops": [
        "Net Income From Continuing Operation Net Minority Interest",
        "Net Income",
    ],
    "depreciation_expense": ["Reconciled Depreciation"],
}

_BALANCE_FIELDS: dict[str, list[str]] = {
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "total_assets": ["Total Assets"],
    "total_debt": ["Total Debt"],
    "total_equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "retained_earnings": ["Retained Earnings"],
    "net_ppe": ["Net PPE"],
    "cash_and_equivalents": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    ],
    "short_term_debt": ["Current Debt", "Current Debt And Capital Lease Obligation"],
    "accounts_receivable": ["Accounts Receivable"],
    "inventory": ["Inventory"],
    "shares_outstanding": ["Ordinary Shares Number"],
}

_CASHFLOW_FIELDS: dict[str, list[str]] = {
    "operating_cash_flow": ["Operating Cash Flow"],
}

#: Lines yfinance omits entirely when the value is genuinely zero for that
#: period (a company that did no buybacks has no "Repurchase Of Capital
#: Stock" row at all) -- absence is read as zero, not as missing data.
_ZERO_WHEN_ABSENT = ["Repurchase Of Capital Stock", "Issuance Of Capital Stock"]


def fetch_live_financials(symbol: str) -> LiveFinancials:
    """Fetch and assemble every input model the fundamentals/workbench
    modules need for ``symbol``, in one pass.

    Raises ``LiveDataError`` naming the missing field if any required
    line item is absent from the two most recent common annual periods.
    """
    ticker = yf.Ticker(f"{symbol}.NS")
    income, balance, cashflow = ticker.income_stmt, ticker.balance_sheet, ticker.cashflow
    if income is None or income.empty:
        raise LiveDataError(symbol, "income_stmt unavailable")
    if balance is None or balance.empty:
        raise LiveDataError(symbol, "balance_sheet unavailable")
    if cashflow is None or cashflow.empty:
        raise LiveDataError(symbol, "cashflow unavailable")

    common_periods = sorted(
        set(income.columns) & set(balance.columns) & set(cashflow.columns), reverse=True
    )
    if len(common_periods) < 2:
        raise LiveDataError(symbol, "fewer than 2 common annual periods across statements")

    current_raw = _extract_period(symbol, common_periods[0], income, balance, cashflow)
    prior_raw = _extract_period(symbol, common_periods[1], income, balance, cashflow)

    history = ticker.history(period="5d")
    if history is None or history.empty:
        raise LiveDataError(symbol, "no recent price history for close/market cap")
    close_price = float(history["Close"].iloc[-1])
    shares = current_raw["shares_outstanding"]
    market_cap = close_price * shares
    net_debt = current_raw["total_debt"] - current_raw["cash_and_equivalents"]

    raw_operating = RawOperatingFinancials(
        symbol=symbol,
        period_end=common_periods[0].date(),
        revenue=current_raw["revenue"],
        operating_income=current_raw["operating_income"],
        amortization_of_intangibles=0.0,
        tax_expense=current_raw["tax_expense"],
        pretax_income=current_raw["pretax_income"],
        current_assets=current_raw["current_assets"],
        current_liabilities=current_raw["current_liabilities"],
        cash_and_equivalents=current_raw["cash_and_equivalents"],
        short_term_debt=current_raw["short_term_debt"],
        net_ppe=current_raw["net_ppe"],
        net_other_operating_assets=0.0,
        prior=RawOperatingFinancials(
            symbol=symbol,
            period_end=common_periods[1].date(),
            revenue=prior_raw["revenue"],
            operating_income=prior_raw["operating_income"],
            amortization_of_intangibles=0.0,
            tax_expense=prior_raw["tax_expense"],
            pretax_income=prior_raw["pretax_income"],
            current_assets=prior_raw["current_assets"],
            current_liabilities=prior_raw["current_liabilities"],
            cash_and_equivalents=prior_raw["cash_and_equivalents"],
            short_term_debt=prior_raw["short_term_debt"],
            net_ppe=prior_raw["net_ppe"],
            net_other_operating_assets=0.0,
        ),
    )

    mscore_inputs = MScoreInputs(
        symbol=symbol,
        period_end=common_periods[0].date(),
        revenue=current_raw["revenue"],
        gross_profit=current_raw["gross_profit"],
        accounts_receivable=current_raw["accounts_receivable"],
        current_assets=current_raw["current_assets"],
        net_ppe=current_raw["net_ppe"],
        total_assets=current_raw["total_assets"],
        depreciation_expense=current_raw["depreciation_expense"],
        sga_expense=current_raw["sga_expense"],
        total_debt=current_raw["total_debt"],
        income_continuing_ops=current_raw["income_continuing_ops"],
        operating_cash_flow=current_raw["operating_cash_flow"],
        prior=MScoreInputs(
            symbol=symbol,
            period_end=common_periods[1].date(),
            revenue=prior_raw["revenue"],
            gross_profit=prior_raw["gross_profit"],
            accounts_receivable=prior_raw["accounts_receivable"],
            current_assets=prior_raw["current_assets"],
            net_ppe=prior_raw["net_ppe"],
            total_assets=prior_raw["total_assets"],
            depreciation_expense=prior_raw["depreciation_expense"],
            sga_expense=prior_raw["sga_expense"],
            total_debt=prior_raw["total_debt"],
            income_continuing_ops=prior_raw["income_continuing_ops"],
            operating_cash_flow=prior_raw["operating_cash_flow"],
        ),
    )

    working_capital = current_raw["current_assets"] - current_raw["current_liabilities"]
    distress_inputs = DistressInputs(
        symbol=symbol,
        working_capital=working_capital,
        total_assets=current_raw["total_assets"],
        retained_earnings=current_raw["retained_earnings"],
        ebit=current_raw["operating_income"],
        market_cap=market_cap,
        total_liabilities=current_raw["total_liabilities"],
        revenue=current_raw["revenue"],
        interest_expense=current_raw["interest_expense"],
        net_debt=net_debt,
        ebitda=current_raw["ebitda"],
        current_assets=current_raw["current_assets"],
        current_liabilities=current_raw["current_liabilities"],
        inventory=current_raw["inventory"],
        cash_and_equivalents=current_raw["cash_and_equivalents"],
    )

    dividends_paid = abs(_lookup_or_zero(cashflow, common_periods[0], ["Cash Dividends Paid"]))
    repurchases = abs(
        _lookup_or_zero(cashflow, common_periods[0], ["Repurchase Of Capital Stock"])
    )
    issuance = _lookup_or_zero(cashflow, common_periods[0], ["Issuance Of Capital Stock"])
    net_buybacks = repurchases - issuance

    multiples_inputs = MultiplesInputs(
        symbol=symbol,
        net_income=current_raw["net_income"],
        book_value=current_raw["total_equity"],
        ebitda=current_raw["ebitda"],
        ebit=current_raw["operating_income"],
        revenue=current_raw["revenue"],
        dividends_paid=dividends_paid,
        net_buybacks=net_buybacks,
        shares_outstanding=shares,
    )

    return LiveFinancials(
        symbol=symbol,
        close_price=close_price,
        shares_outstanding=shares,
        raw_operating=raw_operating,
        mscore_inputs=mscore_inputs,
        distress_inputs=distress_inputs,
        multiples_inputs=multiples_inputs,
        interest_expense=current_raw["interest_expense"],
        total_debt=current_raw["total_debt"],
        prior_total_debt=prior_raw["total_debt"],
        net_debt=net_debt,
    )


def _extract_period(
    symbol: str,
    period: pd.Timestamp,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> dict[str, float]:
    frames_by_field: dict[str, tuple[pd.DataFrame, list[str]]] = {
        **{f: (income, labels) for f, labels in _INCOME_FIELDS.items()},
        **{f: (balance, labels) for f, labels in _BALANCE_FIELDS.items()},
        **{f: (cashflow, labels) for f, labels in _CASHFLOW_FIELDS.items()},
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
        raise LiveDataError(
            symbol, f"missing required field(s) for {period.date()}: {', '.join(missing)}"
        )
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
