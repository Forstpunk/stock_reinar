from __future__ import annotations

from datetime import date

import pytest

from nse_screener.data import FundamentalData
from nse_screener.factors.value import book_to_price, earnings_yield


def _fundamentals(
    net_income: float, total_equity: float, shares_outstanding: float
) -> FundamentalData:
    return FundamentalData(
        symbol="AAA",
        period_end=date(2024, 3, 31),
        total_revenue=1000.0,
        gross_profit=400.0,
        net_income=net_income,
        operating_cash_flow=120.0,
        total_assets=1000.0,
        total_equity=total_equity,
        total_debt=200.0,
        current_assets=300.0,
        current_liabilities=150.0,
        shares_outstanding=shares_outstanding,
        accounts_receivable=100.0,
        inventory=80.0,
    )


def test_earnings_yield_matches_inverse_pe():
    f = _fundamentals(net_income=100.0, total_equity=600.0, shares_outstanding=100.0)
    close_price = 50.0  # market cap = 5000, P/E = 5000/100 = 50 -> E/P = 0.02
    assert earnings_yield(f, close_price) == pytest.approx(0.02)


def test_book_to_price_matches_inverse_pb():
    f = _fundamentals(net_income=100.0, total_equity=600.0, shares_outstanding=100.0)
    close_price = 50.0  # market cap = 5000, P/B = 5000/600 -> B/P = 600/5000 = 0.12
    assert book_to_price(f, close_price) == pytest.approx(0.12)


def test_earnings_yield_is_negative_for_loss_making_company():
    # A negative earnings yield is a legible, well-ordered signal (unlike a
    # negative P/E, which inverts the "lower is cheaper" reading).
    f = _fundamentals(net_income=-50.0, total_equity=600.0, shares_outstanding=100.0)
    assert earnings_yield(f, 50.0) == pytest.approx(-0.01)


def test_raises_on_non_positive_close_price():
    f = _fundamentals(net_income=100.0, total_equity=600.0, shares_outstanding=100.0)
    with pytest.raises(ValueError):
        earnings_yield(f, 0.0)
    with pytest.raises(ValueError):
        book_to_price(f, -10.0)


def test_raises_on_non_positive_market_cap_from_shares():
    f = _fundamentals(net_income=100.0, total_equity=600.0, shares_outstanding=0.0)
    with pytest.raises(ValueError):
        earnings_yield(f, 50.0)
