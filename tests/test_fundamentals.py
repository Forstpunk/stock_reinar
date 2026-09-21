from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_screener.data import (
    InsufficientFundamentalsError,
    UnsupportedStatementFormatError,
    fundamentals_from_statements,
)

# Statement shapes below mirror what yfinance returns for real NSE names
# (observed 2026-09-21): rows are line items, columns are fiscal period ends.

_P26, _P25, _P24, _P23 = (
    pd.Timestamp("2026-03-31"),
    pd.Timestamp("2025-03-31"),
    pd.Timestamp("2024-03-31"),
    pd.Timestamp("2023-03-31"),
)


def _stmt(rows: dict[str, list[float]], periods: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=periods).T


def _classified_statements(periods: list[pd.Timestamp], *, with_inventory: bool = True):
    n = len(periods)
    income = _stmt(
        {
            "Total Revenue": [1000.0 + 10 * i for i in range(n)],
            "Gross Profit": [400.0] * n,
            "Net Income": [100.0] * n,
        },
        periods,
    )
    balance_rows = {
        "Total Assets": [2000.0] * n,
        "Stockholders Equity": [1000.0] * n,
        "Total Debt": [300.0] * n,
        "Current Assets": [500.0] * n,
        "Current Liabilities": [250.0] * n,
        "Ordinary Shares Number": [100.0] * n,
        "Receivables": [100.0] * n,
    }
    if with_inventory:
        balance_rows["Inventory"] = [200.0] * n
    balance = _stmt(balance_rows, periods)
    cashflow = _stmt({"Operating Cash Flow": [150.0] * n}, periods)
    return income, balance, cashflow


def test_classified_statements_with_two_consecutive_years_parse():
    income, balance, cashflow = _classified_statements([_P25, _P24])
    result = fundamentals_from_statements("X", income, balance, cashflow)
    assert result.period_end == _P25.date()
    assert result.prior is not None
    assert result.prior.period_end == _P24.date()
    assert result.inventory == 200.0


def test_unclassified_bank_balance_sheet_is_unsupported_not_missing_data():
    # HDFCBANK shape: Total Assets present, no Current Assets / Current
    # Liabilities / Inventory lines, no Gross Profit.
    income, balance, cashflow = _classified_statements([_P25, _P24])
    income = income.drop(index="Gross Profit")
    balance = balance.drop(index=["Current Assets", "Current Liabilities", "Inventory"])

    with pytest.raises(UnsupportedStatementFormatError, match="unclassified balance sheet"):
        fundamentals_from_statements("HDFCBANK", income, balance, cashflow)


def test_absent_inventory_line_on_classified_balance_sheet_reads_as_zero():
    # INFY shape: fully classified balance sheet, no Inventory row at all.
    income, balance, cashflow = _classified_statements([_P25, _P24], with_inventory=False)
    result = fundamentals_from_statements("INFY", income, balance, cashflow)
    assert result.inventory == 0.0
    assert result.prior is not None
    assert result.prior.inventory == 0.0


def test_inventory_line_present_but_nan_is_still_missing():
    income, balance, cashflow = _classified_statements([_P25, _P24])
    balance.loc["Inventory", _P25] = np.nan
    with pytest.raises(InsufficientFundamentalsError, match="inventory"):
        fundamentals_from_statements("X", income, balance, cashflow)


def test_unreported_placeholder_period_is_skipped():
    # NESTLEIND shape: newest column exists in all three statements but has
    # neither revenue nor net income -- the annual report has not landed.
    income, balance, cashflow = _classified_statements([_P26, _P25, _P24])
    income.loc[["Total Revenue", "Net Income"], _P26] = np.nan
    balance.loc[:, _P26] = np.nan
    cashflow.loc[:, _P26] = np.nan

    result = fundamentals_from_statements("NESTLEIND", income, balance, cashflow)
    assert result.period_end == _P25.date()
    assert result.prior is not None
    assert result.prior.period_end == _P24.date()


def test_non_consecutive_periods_are_rejected_with_reason():
    # NESTLEIND after its Dec -> Mar fiscal-year change: 2025-03-31 then 2022-12-31.
    periods = [_P25, pd.Timestamp("2022-12-31")]
    income, balance, cashflow = _classified_statements(periods)
    with pytest.raises(InsufficientFundamentalsError, match="not consecutive annual periods"):
        fundamentals_from_statements("NESTLEIND", income, balance, cashflow)


def test_partially_reported_period_is_not_skipped_but_fails_loudly():
    # A period with revenue but no balance sheet is a real gap, not a placeholder.
    income, balance, cashflow = _classified_statements([_P25, _P24, _P23])
    balance.loc[:, _P25] = np.nan
    with pytest.raises(InsufficientFundamentalsError, match="total_assets"):
        fundamentals_from_statements("X", income, balance, cashflow)


def test_fewer_than_two_reported_periods_is_rejected():
    income, balance, cashflow = _classified_statements([_P26, _P25])
    income.loc[["Total Revenue", "Net Income"], _P26] = np.nan
    with pytest.raises(InsufficientFundamentalsError, match="fewer than 2 reported"):
        fundamentals_from_statements("X", income, balance, cashflow)
