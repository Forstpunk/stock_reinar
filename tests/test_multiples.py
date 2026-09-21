from __future__ import annotations

from datetime import date

import pytest

from nse_screener.fundamentals.multiples import (
    MultiplesInputs,
    compute_multiples,
    premium_decomposition,
)
from nse_screener.fundamentals.peers import PeerStats
from nse_screener.fundamentals.reorganize import ReorganizedStatements


def _reorganized(**overrides: float) -> ReorganizedStatements:
    base = dict(
        symbol="AAA",
        period_end=date(2024, 3, 31),
        revenue=1200.0,
        ebita=210.0,
        effective_tax_rate=0.20,
        tax_rate_clamped=False,
        operating_tax=42.0,
        nopat=168.0,
        operating_working_capital=140.0,
        excess_cash=30.0,
        invested_capital=1000.0,
        average_invested_capital=950.0,
        increase_in_invested_capital=34.0,
        free_cash_flow=120.0,
    )
    base.update(overrides)
    return ReorganizedStatements(**base)


def _inputs(**overrides: float) -> MultiplesInputs:
    base = dict(
        symbol="AAA",
        net_income=150.0,
        book_value=800.0,
        ebitda=250.0,
        ebit=200.0,
        revenue=1200.0,
        dividends_paid=50.0,
        net_buybacks=30.0,
        shares_outstanding=100.0,
    )
    base.update(overrides)
    return MultiplesInputs(**base)


def test_multiples_match_hand_computed_values():
    result = compute_multiples(_inputs(), _reorganized(), market_cap=3000.0, net_debt=500.0)

    assert result.enterprise_value == pytest.approx(3500.0)
    assert result.pe == pytest.approx(20.0)
    assert result.pb == pytest.approx(3.75)
    assert result.ev_ebitda == pytest.approx(14.0)
    assert result.ev_ebit == pytest.approx(17.5)
    assert result.ev_sales == pytest.approx(3500.0 / 1200.0)
    assert result.ev_invested_capital == pytest.approx(3.5)
    assert result.fcf_yield == pytest.approx(120.0 / 3000.0)
    assert result.dividend_yield == pytest.approx(50.0 / 3000.0)
    assert result.payout_ratio == pytest.approx(50.0 / 150.0)
    assert result.total_payout_yield == pytest.approx(80.0 / 3000.0)


def test_pe_is_none_with_reason_for_negative_net_income():
    result = compute_multiples(
        _inputs(net_income=-10.0), _reorganized(), market_cap=3000.0, net_debt=500.0
    )
    assert result.pe is None
    assert result.pe_reason is not None
    # negative earnings must never surface as a negative P/E
    assert result.pe != pytest.approx(-300.0)


def test_ev_ebitda_and_ev_ebit_none_when_non_positive():
    result = compute_multiples(
        _inputs(ebitda=0.0, ebit=-5.0), _reorganized(), market_cap=3000.0, net_debt=500.0
    )
    assert result.ev_ebitda is None
    assert result.ev_ebit is None


def test_ev_invested_capital_none_when_invested_capital_non_positive():
    result = compute_multiples(
        _inputs(), _reorganized(invested_capital=0.0), market_cap=3000.0, net_debt=500.0
    )
    assert result.ev_invested_capital is None


def test_fcf_yield_can_be_negative_and_is_still_computed():
    result = compute_multiples(
        _inputs(), _reorganized(free_cash_flow=-50.0), market_cap=3000.0, net_debt=500.0
    )
    assert result.fcf_yield == pytest.approx(-50.0 / 3000.0)


def test_raises_on_non_positive_market_cap():
    with pytest.raises(ValueError, match="market_cap"):
        compute_multiples(_inputs(), _reorganized(), market_cap=0.0, net_debt=500.0)


def test_premium_decomposition_joins_peer_stats():
    multiple_stats = PeerStats(
        symbol="AAA",
        sector="Energy",
        metric_value=14.0,
        sector_median=10.0,
        sector_q1=8.0,
        sector_q3=12.0,
        peer_count=6,
        percentile_within_sector=0.9,
        sufficient_peers=True,
    )
    roic_stats = PeerStats(
        symbol="AAA",
        sector="Energy",
        metric_value=0.18,
        sector_median=0.12,
        sector_q1=0.09,
        sector_q3=0.15,
        peer_count=6,
        percentile_within_sector=0.85,
        sufficient_peers=True,
    )
    margin_stats = PeerStats(
        symbol="AAA",
        sector="Energy",
        metric_value=0.20,
        sector_median=0.15,
        sector_q1=0.10,
        sector_q3=0.18,
        peer_count=6,
        percentile_within_sector=0.80,
        sufficient_peers=True,
    )

    result = premium_decomposition("EV/EBITDA", multiple_stats, roic_stats, margin_stats)

    assert result.premium_to_sector_median_pct == pytest.approx(40.0)  # 14/10 - 1 = 40%
    assert result.roic_percentile_within_sector == pytest.approx(0.85)
    assert result.operating_margin_percentile_within_sector == pytest.approx(0.80)


def test_premium_decomposition_raises_on_mismatched_symbols():
    stats_a = PeerStats(
        symbol="AAA", sector="Energy", metric_value=1.0, sector_median=1.0,
        sector_q1=1.0, sector_q3=1.0, peer_count=5, percentile_within_sector=0.5,
        sufficient_peers=True,
    )
    stats_b = stats_a.model_copy(update={"symbol": "BBB"})
    with pytest.raises(ValueError, match="mismatched symbols"):
        premium_decomposition("P/E", stats_a, stats_b, stats_a)
