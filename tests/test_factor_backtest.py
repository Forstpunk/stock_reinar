from __future__ import annotations

import pandas as pd
import pytest

from nse_screener import factor_backtest
from nse_screener.factor_backtest import (
    MIN_OBSERVATIONS_FOR_RANK_IC,
    _build_report,
    _quintile_stats,
    _rank_ic,
    backtest_fundamentals_factors,
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
