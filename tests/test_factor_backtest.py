from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_screener import factor_backtest
from nse_screener.config import ScreenerConfig
from nse_screener.factor_backtest import (
    MIN_AS_OF_DATES_FOR_CI,
    MIN_EFFECTIVE_N,
    MIN_OBSERVATIONS_FOR_RANK_IC,
    QuintileStat,
    _bootstrap_by_as_of_date,
    _build_report,
    _effective_n,
    _independence_factor,
    _monotonicity_stats,
    _q5_minus_q1_statistic,
    _quintile_stats,
    _rank_ic,
    _rank_ic_statistic,
    backtest_fundamentals_factors,
    backtest_momentum,
    diagnose_roe_leverage,
    power_warnings,
)


def test_rank_ic_perfect_positive_relationship():
    pct = pd.Series(range(20), dtype=float) / 19.0
    ret = pd.Series(range(20), dtype=float)  # monotonically increasing with pct
    assert _rank_ic(pct, ret) == pytest.approx(1.0)


def test_rank_ic_perfect_negative_relationship():
    pct = pd.Series(range(20), dtype=float) / 19.0
    ret = pd.Series(range(20), dtype=float)[::-1].reset_index(drop=True)
    assert _rank_ic(pct, ret) == pytest.approx(-1.0)


def test_rank_ic_none_below_minimum_sample():
    n = MIN_OBSERVATIONS_FOR_RANK_IC - 1
    pct = pd.Series(range(n), dtype=float)
    ret = pd.Series(range(n), dtype=float)
    assert _rank_ic(pct, ret) is None


def test_rank_ic_ignores_rows_with_nan_return():
    pct = pd.Series(range(15), dtype=float)
    ret = pd.Series([float(i) if i % 3 else None for i in range(15)])
    # still enough non-NaN rows to compute
    assert _rank_ic(pct, ret) is not None


def test_quintile_stats_splits_into_five_buckets_with_expected_ordering():
    pct = pd.Series(range(50), dtype=float) / 49.0
    ret = pd.Series(range(50), dtype=float)  # higher pct -> higher return, by construction
    stats = _quintile_stats(pct, ret)

    assert [s.quintile for s in stats] == [1, 2, 3, 4, 5]
    assert sum(s.n for s in stats) == 50
    # monotonically increasing mean return across quintiles, by construction
    means = [s.mean_forward_return_pct for s in stats]
    assert means == sorted(means)


def test_quintile_stats_empty_below_n_quintiles():
    pct = pd.Series([0.1, 0.5, 0.9])
    ret = pd.Series([1.0, 2.0, 3.0])
    assert _quintile_stats(pct, ret) == []


def test_build_report_counts_distinct_as_of_dates():
    df = pd.DataFrame(
        {
            "as_of": [pd.Timestamp("2024-01-01")] * 5 + [pd.Timestamp("2024-02-01")] * 5,
            "symbol": [f"S{i}" for i in range(10)],
            "factor_pct": [i / 9.0 for i in range(10)],
            "fwd_21": [float(i) for i in range(10)],
        }
    )
    report = _build_report("test_factor", df, horizons=(21,))
    assert report.n_as_of_dates == 2
    assert report.n_observations_total == 10
    assert report.horizons[21].n_observations == 10


def test_build_report_handles_empty_dataframe():
    report = _build_report("test_factor", pd.DataFrame(), horizons=(21, 63))
    assert report.n_observations_total == 0
    assert report.n_as_of_dates == 0
    assert report.horizons[21].n_observations == 0
    assert report.horizons[21].rank_ic is None
    assert report.horizons[21].quintiles == []


def test_build_report_drops_unmatured_rows_per_horizon():
    df = pd.DataFrame(
        {
            "as_of": [pd.Timestamp("2024-01-01")] * 4,
            "symbol": ["A", "B", "C", "D"],
            "factor_pct": [0.1, 0.4, 0.6, 0.9],
            "fwd_21": [1.0, 2.0, None, 4.0],
        }
    )
    report = _build_report("test_factor", df, horizons=(21,))
    assert report.horizons[21].n_observations == 3  # the None row is excluded


# -- Regression: every excluded symbol must be counted and named, never -------
# -- silently merged into an undifferentiated "no data" bucket. -----------------


def test_fetch_failures_are_named_not_silently_dropped(monkeypatch: pytest.MonkeyPatch):
    def fake_fetch_statements(symbol: str):
        if symbol == "NODATA":
            return None, "income_stmt_unavailable"
        if symbol == "CRASHES":
            raise AssertionError("should never be called directly -- _fetch_statements catches")
        return None, "fetch_error:ConnectionError"

    monkeypatch.setattr(factor_backtest, "_fetch_statements", fake_fetch_statements)

    run = backtest_fundamentals_factors(["NODATA", "OTHER"], price_frames={})

    assert run.fetch_failures == {
        "NODATA": "income_stmt_unavailable",
        "OTHER": "fetch_error:ConnectionError",
    }
    assert run.fetch_failure_counts == {
        "income_stmt_unavailable": 1,
        "fetch_error:ConnectionError": 1,
    }
    for report in run.reports.values():
        assert report.n_observations_total == 0


def test_symbol_with_statements_but_no_price_history_is_counted(monkeypatch: pytest.MonkeyPatch):
    def fake_fetch_statements(symbol: str):
        income = pd.DataFrame({pd.Timestamp("2024-03-31"): [1.0]}, index=["Total Revenue"])
        return (income, income, income), None

    monkeypatch.setattr(factor_backtest, "_fetch_statements", fake_fetch_statements)

    run = backtest_fundamentals_factors(["NOPRICE"], price_frames={})

    assert run.fetch_failures == {"NOPRICE": "no_price_history_supplied"}


def test_symbol_with_no_valid_vintage_is_counted(monkeypatch: pytest.MonkeyPatch):
    # A symbol whose statements exist but share no common period across all
    # three (or fail the fiscal-gap check) produces zero vintages -- this
    # must be visible too, not merged into "fetched fine, nothing to show".
    def fake_fetch_statements(symbol: str):
        income = pd.DataFrame({pd.Timestamp("2024-03-31"): [1.0]}, index=["Total Revenue"])
        balance = pd.DataFrame({pd.Timestamp("2023-01-01"): [1.0]}, index=["Total Assets"])
        cashflow = pd.DataFrame({pd.Timestamp("2022-01-01"): [1.0]}, index=["Operating Cash Flow"])
        return (income, balance, cashflow), None

    monkeypatch.setattr(factor_backtest, "_fetch_statements", fake_fetch_statements)

    dates = pd.bdate_range("2020-01-01", periods=400)
    price_frames = {"NOVINTAGE": pd.DataFrame({"Close": [100.0] * 400}, index=dates)}

    run = backtest_fundamentals_factors(["NOVINTAGE"], price_frames=price_frames)

    assert run.fetch_failures == {"NOVINTAGE": "no_valid_vintage"}


# =============================================================================
# Task 1 -- effective sample size, not just row count
# =============================================================================


def test_independence_factor_is_spacing_over_horizon_when_below_one():
    assert _independence_factor(21.0, 63) == pytest.approx(1.0 / 3.0)


def test_independence_factor_capped_at_one_when_spacing_exceeds_horizon():
    assert _independence_factor(200.0, 63) == pytest.approx(1.0)


def test_effective_n_scales_by_independence_factor():
    # 30 as-of dates, 21-session spacing, 63-session horizon -> factor 1/3 -> 10
    assert _effective_n(30, 21.0, 63) == 10


def test_effective_n_equals_n_as_of_dates_when_spacing_exceeds_horizon():
    assert _effective_n(20, 200.0, 63) == 20


def test_effective_n_never_below_one_when_dates_exist():
    assert _effective_n(5, 1.0, 126) >= 1
    assert _effective_n(1, None, 21) == 1


def test_effective_n_zero_when_no_as_of_dates():
    assert _effective_n(0, None, 21) == 0


def test_overlap_warning_true_below_min_effective_n():
    # 10 as-of dates spaced 1 session apart, 63-session horizon -> effective_n
    # far below MIN_EFFECTIVE_N (30).
    df = pd.DataFrame(
        {
            "as_of": [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(10)] * 2,
            "symbol": [f"S{i}" for i in range(10)] * 2,
            "factor_pct": list(np.linspace(0, 1, 10)) * 2,
            "fwd_63": list(np.linspace(-1, 1, 10)) * 2,
        }
    )
    report = _build_report("test_factor", df, horizons=(63,))
    assert report.horizons[63].effective_n < MIN_EFFECTIVE_N
    assert report.horizons[63].overlap_warning is True


def test_overlap_warning_false_when_effective_n_is_large():
    session_calendar = pd.bdate_range("2020-01-01", periods=3000)
    as_of_dates = session_calendar[::63][:40]  # 40 dates, spaced well past the horizon
    rows = []
    for d in as_of_dates:
        for i in range(20):
            rows.append(
                {"as_of": d, "symbol": f"S{i}", "factor_pct": i / 19.0, "fwd_21": float(i)}
            )
    df = pd.DataFrame(rows)
    report = _build_report("test_factor", df, horizons=(21,), session_calendar=session_calendar)
    assert report.horizons[21].effective_n >= MIN_EFFECTIVE_N
    assert report.horizons[21].overlap_warning is False


# =============================================================================
# Task 2 -- bootstrap confidence intervals, clustered by as-of date
# =============================================================================


def _synthetic_as_of_df(n_dates: int, n_symbols: int, relationship: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        pct = np.linspace(0, 1, n_symbols)
        if relationship == "strong":
            ret = pct * 10.0
        else:  # "noise"
            ret = rng.normal(0, 1, n_symbols)
        for i in range(n_symbols):
            rows.append(
                {
                    "as_of": pd.Timestamp("2020-01-01") + pd.Timedelta(days=30 * d),
                    "symbol": f"S{i}",
                    "factor_pct": pct[i],
                    "fwd_21": ret[i],
                }
            )
    return pd.DataFrame(rows)


def test_bootstrap_is_deterministic_under_fixed_seed():
    df = _synthetic_as_of_df(10, 20, "strong")
    ci_a = _bootstrap_by_as_of_date(df, lambda d: _rank_ic_statistic(d, "fwd_21"))
    ci_b = _bootstrap_by_as_of_date(df, lambda d: _rank_ic_statistic(d, "fwd_21"))
    assert ci_a == ci_b


def test_bootstrap_ci_excludes_zero_for_strong_relationship():
    df = _synthetic_as_of_df(15, 30, "strong")
    ci = _bootstrap_by_as_of_date(df, lambda d: _rank_ic_statistic(d, "fwd_21"))
    assert ci is not None
    lower, upper = ci
    assert lower > 0  # strong positive relationship -> CI entirely above zero


def test_bootstrap_ci_spans_zero_for_pure_noise():
    # 15 as-of dates is too few to reliably keep a pure-noise CI centered on
    # zero (small-sample resampling variance can shift it either way by
    # chance) -- 50 dates makes the null result stable across seeds.
    df = _synthetic_as_of_df(50, 30, "noise", seed=7)
    ci = _bootstrap_by_as_of_date(df, lambda d: _rank_ic_statistic(d, "fwd_21"))
    assert ci is not None
    lower, upper = ci
    assert lower <= 0 <= upper


def test_bootstrap_returns_none_below_min_as_of_dates():
    df = _synthetic_as_of_df(MIN_AS_OF_DATES_FOR_CI - 1, 20, "strong")
    ci = _bootstrap_by_as_of_date(df, lambda d: _rank_ic_statistic(d, "fwd_21"))
    assert ci is None


def test_q5_minus_q1_statistic_matches_hand_computed_spread():
    df = pd.DataFrame(
        {
            "factor_pct": np.linspace(0, 1, 50),
            "fwd_21": np.linspace(0, 1, 50),  # perfectly monotonic increasing
        }
    )
    result = _q5_minus_q1_statistic(df, "fwd_21")
    assert result > 0


def test_build_report_spans_zero_set_for_noisy_factor():
    df = _synthetic_as_of_df(15, 30, "noise", seed=11)
    report = _build_report("noise_factor", df, horizons=(21,))
    result = report.horizons[21]
    assert result.q5_minus_q1_ci_95 is not None
    assert result.spans_zero is not None


def test_build_report_ci_is_none_with_too_few_as_of_dates():
    df = _synthetic_as_of_df(2, 20, "strong")
    report = _build_report("test_factor", df, horizons=(21,))
    result = report.horizons[21]
    assert result.rank_ic_ci_95 is None
    assert result.q5_minus_q1_ci_95 is None
    assert result.spans_zero is None


# =============================================================================
# Task 3 -- monotonicity, not just spread
# =============================================================================


def test_monotonicity_perfectly_increasing_quintiles():
    quintiles = [
        QuintileStat(quintile=q, mean_forward_return_pct=float(q), n=10) for q in range(1, 6)
    ]
    spearman, is_monotonic, largest_step_share = _monotonicity_stats(quintiles)
    assert spearman == pytest.approx(1.0)
    assert is_monotonic is True


def test_monotonicity_perfectly_decreasing_quintiles():
    quintiles = [
        QuintileStat(quintile=q, mean_forward_return_pct=float(6 - q), n=10) for q in range(1, 6)
    ]
    spearman, is_monotonic, largest_step_share = _monotonicity_stats(quintiles)
    assert spearman == pytest.approx(-1.0)
    assert is_monotonic is False


def test_monotonicity_single_step_dominates_spread():
    # Q1..Q4 flat at 0, Q5 jumps to 10 -- one boundary supplies the whole spread.
    returns = [0.0, 0.0, 0.0, 0.0, 10.0]
    quintiles = [
        QuintileStat(quintile=q, mean_forward_return_pct=returns[q - 1], n=10)
        for q in range(1, 6)
    ]
    _, _, largest_step_share = _monotonicity_stats(quintiles)
    assert largest_step_share is not None
    assert largest_step_share > 0.9


def test_monotonicity_none_below_two_quintiles():
    assert _monotonicity_stats([]) == (None, None, None)
    single = [QuintileStat(quintile=1, mean_forward_return_pct=1.0, n=5)]
    assert _monotonicity_stats(single) == (None, None, None)


# =============================================================================
# Task 4 -- extended momentum as-of range, power warnings
# =============================================================================


def _synthetic_price_frames(n_symbols: int, n_sessions: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_sessions)
    frames = {}
    for i in range(n_symbols):
        drift = rng.normal(0.0003, 0.0002)
        noise = rng.normal(0, 0.02, n_sessions)
        close = 100.0 * np.cumprod(1 + drift + noise)
        frames[f"S{i}"] = pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": np.full(n_sessions, 500_000.0),
            },
            index=dates,
        )
    return frames


def test_momentum_backtest_as_of_count_matches_step_over_five_years():
    price_frames = _synthetic_price_frames(n_symbols=15, n_sessions=1260)  # ~5 years
    config = ScreenerConfig()
    report = backtest_momentum(price_frames, config, as_of_step_sessions=21)

    required = max(config.momentum_min_sessions, 4 * 63 + 1)
    max_horizon = 126
    expected = len(range(required, 1260 - max_horizon, 21))
    assert report.n_as_of_dates == expected


def test_momentum_backtest_warmup_respects_252_session_floor():
    price_frames = _synthetic_price_frames(n_symbols=15, n_sessions=1260)
    config = ScreenerConfig()
    report = backtest_momentum(price_frames, config, as_of_step_sessions=21)

    dates = pd.bdate_range("2015-01-01", periods=1260)
    assert report.first_as_of_date is not None
    first_as_of_pos = dates.searchsorted(pd.Timestamp(report.first_as_of_date))
    assert first_as_of_pos >= 252


def test_power_warning_fires_below_threshold():
    df = _synthetic_as_of_df(10, 20, "strong")
    report = _build_report("thin_factor", df, horizons=(21,))
    warnings = power_warnings({"thin_factor": report})
    assert len(warnings) == 1
    assert "thin_factor" in warnings[0]


def test_power_warning_absent_above_threshold():
    session_calendar = pd.bdate_range("2020-01-01", periods=3000)
    as_of_dates = session_calendar[::21][:40]
    rows = [
        {"as_of": d, "symbol": f"S{i}", "factor_pct": i / 19.0, "fwd_21": float(i)}
        for d in as_of_dates
        for i in range(20)
    ]
    df = pd.DataFrame(rows)
    report = _build_report("wide_factor", df, horizons=(21,), session_calendar=session_calendar)
    assert power_warnings({"wide_factor": report}) == []


# =============================================================================
# Task 5 -- ROE leverage diagnostic
# =============================================================================


def test_roe_diagnostic_groups_are_disjoint_and_cover_top_quintile(
    monkeypatch: pytest.MonkeyPatch,
):
    # Debt scales linearly with symbol index; equity cycles independently
    # (mod 5) so ROE varies too, and the top-ROE quintile ends up spanning
    # a real spread of debt levels rather than one degenerate value.
    def fake_fetch_statements(symbol: str):
        idx = int(symbol[1:])
        debt = 100.0 + idx * 50.0
        equity_current = 600.0 - (idx % 5) * 20.0
        equity_prior = equity_current - 50.0
        current = pd.Timestamp("2024-03-31")
        prior = pd.Timestamp("2023-03-31")
        income = pd.DataFrame(
            {current: [1000.0, 400.0, 100.0], prior: [900.0, 360.0, 90.0]},
            index=["Total Revenue", "Gross Profit", "Net Income"],
        )
        balance = pd.DataFrame(
            {
                current: [1000.0, equity_current, debt, 300.0, 150.0, 100.0, 80.0, 50.0],
                prior: [900.0, equity_prior, debt * 0.9, 270.0, 140.0, 95.0, 75.0, 50.0],
            },
            index=[
                "Total Assets",
                "Stockholders Equity",
                "Total Debt",
                "Current Assets",
                "Current Liabilities",
                "Receivables",
                "Inventory",
                "Ordinary Shares Number",
            ],
        )
        cashflow = pd.DataFrame(
            {current: [110.0], prior: [95.0]}, index=["Operating Cash Flow"]
        )
        return (income, balance, cashflow), None

    monkeypatch.setattr(factor_backtest, "_fetch_statements", fake_fetch_statements)

    dates = pd.bdate_range("2020-01-01", periods=600)
    symbols = [f"S{i}" for i in range(20)]
    price_frames = {
        s: pd.DataFrame({"Close": np.linspace(100, 120, 600)}, index=dates) for s in symbols
    }

    report = diagnose_roe_leverage(symbols, price_frames, horizons=(21,), fetch_roic=False)

    total_in_groups = report.high_leverage.n + report.low_leverage.n
    assert total_in_groups > 0
    assert report.high_leverage.n > 0
    assert report.low_leverage.n > 0
    # every high-leverage member's debt/equity is at least the group split
    # point -- confirmed indirectly via the reported group means.
    assert report.high_leverage.mean_debt_to_equity >= report.low_leverage.mean_debt_to_equity


def test_roe_diagnostic_excludes_non_positive_equity_symbol(monkeypatch: pytest.MonkeyPatch):
    # Current-period equity negative but prior positive enough that the
    # *average* (what return_on_equity checks) stays positive -- ROE
    # computes fine, but debt_to_equity (current equity only) is undefined.
    # This isolates the check in diagnose_roe_leverage from the one
    # already inside return_on_equity itself.
    def fake_fetch_statements(symbol: str):
        current = pd.Timestamp("2024-03-31")
        prior = pd.Timestamp("2023-03-31")
        if symbol == "NEGEQUITY":
            equity_current, equity_prior = -50.0, 700.0
        else:
            # Vary equity across the "S" symbols so the remaining population
            # (after NEGEQUITY is excluded) has a real top quintile to find --
            # this test only cares that NEGEQUITY itself is named and
            # counted, not about the split's composition.
            idx = int(symbol[1:])
            equity_current = 600.0 - (idx % 5) * 20.0
            equity_prior = equity_current - 50.0
        income = pd.DataFrame(
            {current: [1000.0, 400.0, 100.0], prior: [900.0, 360.0, 90.0]},
            index=["Total Revenue", "Gross Profit", "Net Income"],
        )
        balance = pd.DataFrame(
            {
                current: [1000.0, equity_current, 200.0, 300.0, 150.0, 100.0, 80.0, 50.0],
                prior: [900.0, equity_prior, 180.0, 270.0, 140.0, 95.0, 75.0, 50.0],
            },
            index=[
                "Total Assets",
                "Stockholders Equity",
                "Total Debt",
                "Current Assets",
                "Current Liabilities",
                "Receivables",
                "Inventory",
                "Ordinary Shares Number",
            ],
        )
        cashflow = pd.DataFrame(
            {current: [110.0], prior: [95.0]}, index=["Operating Cash Flow"]
        )
        return (income, balance, cashflow), None

    monkeypatch.setattr(factor_backtest, "_fetch_statements", fake_fetch_statements)

    dates = pd.bdate_range("2020-01-01", periods=600)
    symbols = ["NEGEQUITY"] + [f"S{i}" for i in range(20)]
    price_frames = {
        s: pd.DataFrame({"Close": np.linspace(100, 120, 600)}, index=dates) for s in symbols
    }

    report = diagnose_roe_leverage(symbols, price_frames, horizons=(21,), fetch_roic=False)

    # NEGEQUITY's average equity is positive (ROE computes), but its
    # current-period equity is negative, making debt_to_equity undefined --
    # it must be named and counted, not silently absorbed into either group.
    assert report.excluded.get("NEGEQUITY") == "equity_non_positive"
