from __future__ import annotations

import pytest

from nse_screener.gate import GateVerdict, Verdict
from nse_screener.workbench.sizing import (
    MAX_RISK_PCT,
    ExpectancyResult,
    InsufficientSampleError,
    RiskParameterError,
    TradeRecord,
    calculate_position,
    expectancy,
)


def _gate(verdict: Verdict) -> GateVerdict:
    return GateVerdict(
        verdict=verdict,
        index_close=100.0,
        index_ma50=95.0,
        index_ma200=90.0,
        index_vs_50dma_pct=0.05,
        index_vs_200dma_pct=0.10,
        dma200_slope_21d_pct=0.01,
        breadth_trend_aligned_pct=0.5,
        breadth_above_ma200_pct=0.6,
        breadth_near_52w_high_pct=0.4,
        exposure_guidance="test",
    )


_HEALTHY = _gate(Verdict.HEALTHY)
_NEUTRAL = _gate(Verdict.NEUTRAL)
_HOSTILE = _gate(Verdict.HOSTILE)


def test_share_count_arithmetic_no_cap():
    plan = calculate_position(
        capital=500_000.0,
        risk_pct=0.01,
        entry_price=1000.0,
        atr_14=50.0,
        gate_verdict=_HEALTHY,
        graham_label="INVESTMENT",
        atr_multiple=2.0,
    )
    # stop_distance = 100, risk_amount = 5000, shares = floor(5000/100) = 50
    assert plan.stop_distance == pytest.approx(100.0)
    assert plan.stop_price == pytest.approx(900.0)
    assert plan.risk_amount == pytest.approx(5000.0)
    assert plan.shares == 50
    assert plan.position_value == pytest.approx(50_000.0)
    assert plan.position_value_capped is False
    assert plan.r_multiple_targets == {
        1: pytest.approx(1100.0),
        2: pytest.approx(1200.0),
        3: pytest.approx(1300.0),
    }


def test_gate_scaling_neutral_halves_shares():
    plan = calculate_position(
        capital=500_000.0, risk_pct=0.01, entry_price=1000.0, atr_14=50.0,
        gate_verdict=_NEUTRAL, graham_label="INVESTMENT",
    )
    assert plan.gate_multiplier == pytest.approx(0.5)
    assert plan.shares == 25
    assert plan.permitted is True


def test_gate_scaling_hostile_zeroes_shares_and_forbids():
    plan = calculate_position(
        capital=500_000.0, risk_pct=0.01, entry_price=1000.0, atr_14=50.0,
        gate_verdict=_HOSTILE, graham_label="INVESTMENT",
    )
    assert plan.gate_multiplier == pytest.approx(0.0)
    assert plan.shares == 0
    assert plan.permitted is False
    assert plan.reason == "market gate hostile"


def test_speculation_halves_effective_risk():
    plan = calculate_position(
        capital=500_000.0, risk_pct=0.01, entry_price=1000.0, atr_14=50.0,
        gate_verdict=_HEALTHY, graham_label="SPECULATION",
    )
    assert plan.speculation_multiplier == pytest.approx(0.5)
    assert plan.shares == 25


def test_position_value_capped_at_20_percent_of_capital():
    plan = calculate_position(
        capital=500_000.0, risk_pct=0.02, entry_price=1000.0, atr_14=5.0,
        gate_verdict=_HEALTHY, graham_label="INVESTMENT", atr_multiple=2.0,
    )
    # stop_distance=10, uncapped shares would be floor(10000/10)=1000 -> value 1,000,000
    assert plan.position_value_capped is True
    assert plan.position_value == pytest.approx(500_000.0 * 0.20)
    assert plan.shares == 100


def test_raises_above_max_risk_pct():
    with pytest.raises(RiskParameterError):
        calculate_position(
            capital=500_000.0, risk_pct=MAX_RISK_PCT + 0.01, entry_price=1000.0,
            atr_14=50.0, gate_verdict=_HEALTHY, graham_label="INVESTMENT",
        )


def test_at_exactly_max_risk_pct_does_not_raise():
    plan = calculate_position(
        capital=500_000.0, risk_pct=MAX_RISK_PCT, entry_price=1000.0,
        atr_14=50.0, gate_verdict=_HEALTHY, graham_label="INVESTMENT",
    )
    assert plan.risk_pct_requested == pytest.approx(MAX_RISK_PCT)


# -- Expectancy -----------------------------------------------------------------


def test_expectancy_raises_below_min_sample():
    trades = [TradeRecord(r_multiple=1.0) for _ in range(29)]
    with pytest.raises(InsufficientSampleError):
        expectancy(trades)


def test_expectancy_arithmetic_and_ci_brackets_point_estimate():
    # 20 wins at +2R, 10 losses at -1R -> win_rate=2/3, avg_win=2, loss_rate=1/3, avg_loss=1
    trades = [TradeRecord(r_multiple=2.0) for _ in range(20)] + [
        TradeRecord(r_multiple=-1.0) for _ in range(10)
    ]
    result = expectancy(trades, min_sample=30, bootstrap_iterations=2000, seed=42)

    assert isinstance(result, ExpectancyResult)
    assert result.sample_size == 30
    assert result.win_rate == pytest.approx(20 / 30)
    assert result.loss_rate == pytest.approx(10 / 30)
    assert result.avg_win_r == pytest.approx(2.0)
    assert result.avg_loss_r == pytest.approx(1.0)
    expected_expectancy = (20 / 30) * 2.0 - (10 / 30) * 1.0
    assert result.expectancy_r == pytest.approx(expected_expectancy)
    assert result.ci_low <= result.expectancy_r <= result.ci_high
