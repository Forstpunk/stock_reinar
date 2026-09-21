from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from nse_screener.config import ScreenerConfig
from nse_screener.data import FundamentalData
from nse_screener.ranking import build_momentum_only_shortlist, build_shortlist

_PRIOR_BASE = dict(
    period_end=date(2023, 3, 31),
    total_revenue=1000.0,
    gross_profit=400.0,
    net_income=100.0,
    operating_cash_flow=110.0,
    total_assets=1000.0,
    total_equity=600.0,
    total_debt=200.0,
    current_assets=300.0,
    current_liabilities=150.0,
    shares_outstanding=100.0,
    accounts_receivable=100.0,
    inventory=80.0,
)

_CURRENT_BASE = dict(
    period_end=date(2024, 3, 31),
    total_revenue=1100.0,
    gross_profit=440.0,
    net_income=110.0,
    operating_cash_flow=120.0,
    total_assets=1100.0,
    total_equity=650.0,
    total_debt=210.0,
    current_assets=330.0,
    current_liabilities=160.0,
    shares_outstanding=100.0,
    accounts_receivable=110.0,
    inventory=88.0,
)


def _fundamentals(symbol: str, current_overrides: dict | None = None) -> FundamentalData:
    prior = FundamentalData(symbol=symbol, **_PRIOR_BASE)
    current = FundamentalData(symbol=symbol, **{**_CURRENT_BASE, **(current_overrides or {})})
    return current.model_copy(update={"prior": prior})


def _price_frame(session_count: int, daily_return: float, volume: float) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=session_count)
    close = 1000.0 * (1 + daily_return) ** np.arange(session_count)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(session_count, volume),
        },
        index=dates,
    )


@pytest.fixture
def config() -> ScreenerConfig:
    return ScreenerConfig()


def test_build_shortlist_filters_and_ranks(config: ScreenerConfig):
    price_frames = {
        "GOOD1": _price_frame(320, daily_return=0.002, volume=300_000.0),
        "GOOD2": _price_frame(320, daily_return=0.001, volume=300_000.0),
        "THIN": _price_frame(100, daily_return=0.002, volume=300_000.0),
        "ILLIQUID": _price_frame(320, daily_return=0.002, volume=1_000.0),
        "NOFUND": _price_frame(320, daily_return=0.002, volume=300_000.0),
        "BAD": _price_frame(320, daily_return=0.002, volume=300_000.0),
    }
    fundamentals = {
        "GOOD1": _fundamentals("GOOD1"),
        "GOOD2": _fundamentals("GOOD2"),
        "THIN": _fundamentals("THIN"),
        "ILLIQUID": _fundamentals("ILLIQUID"),
        "BAD": _fundamentals("BAD", {"operating_cash_flow": 50.0, "total_debt": 400.0}),
        # NOFUND deliberately has no fundamentals entry.
    }

    shortlist = build_shortlist(price_frames, fundamentals, config, top_n=5)

    symbols_in_shortlist = [entry.symbol for entry in shortlist.entries]
    assert symbols_in_shortlist == ["GOOD1", "GOOD2"]
    assert len(shortlist.entries) < shortlist.requested_top_n
    assert shortlist.shortfall_note is not None

    assert shortlist.excluded_count["insufficient_price_history"] == 1
    assert shortlist.excluded_count["insufficient_liquidity"] == 1
    assert shortlist.excluded_count["fundamentals_unavailable"] == 1
    assert shortlist.excluded_count["forensically_disqualified"] == 1

    assert "BAD" in shortlist.disqualified
    assert "BAD" not in symbols_in_shortlist


def test_composite_score_arithmetic(config: ScreenerConfig):
    price_frames = {
        "GOOD1": _price_frame(320, daily_return=0.002, volume=300_000.0),
        "GOOD2": _price_frame(320, daily_return=0.001, volume=300_000.0),
    }
    fundamentals = {
        "GOOD1": _fundamentals("GOOD1"),
        "GOOD2": _fundamentals("GOOD2"),
    }

    shortlist = build_shortlist(price_frames, fundamentals, config, top_n=5)
    entries_by_symbol = {entry.symbol: entry for entry in shortlist.entries}

    for entry in shortlist.entries:
        expected = (
            config.weight_rs * entry.rs_percentile
            + config.weight_f_score * entry.f_score_percentile
            + config.weight_value * entry.value_percentile
            + config.weight_gross_profitability * entry.gross_profitability_percentile
            + config.weight_roe * entry.roe_percentile
        )
        assert entry.composite_score == pytest.approx(expected)

    # identical fundamentals for both symbols -> composite ordering driven
    # purely by the momentum (RS) differential
    assert entries_by_symbol["GOOD1"].composite_score > entries_by_symbol["GOOD2"].composite_score
    assert entries_by_symbol["GOOD1"].rs_percentile > entries_by_symbol["GOOD2"].rs_percentile


def test_shortlist_never_padded_when_nothing_qualifies(config: ScreenerConfig):
    price_frames = {
        "THIN": _price_frame(100, daily_return=0.002, volume=300_000.0),
    }
    shortlist = build_shortlist(price_frames, {}, config, top_n=5)

    assert shortlist.entries == []
    assert shortlist.qualifying_count == 0
    assert shortlist.shortfall_note is not None


# -- One momentum-uncomputable symbol must not abort the run ------------------


def _sparse_close_frame(n_valid: int, session_count: int = 400) -> pd.DataFrame:
    """Frame whose row count passes eligibility but whose *valid* Close count may not.

    Close is NaN for the first ``session_count - n_valid`` rows while Volume
    is not, so ``dropna(how="all")`` keeps every row and only the momentum
    computation discovers the shortfall.
    """
    dates = pd.bdate_range("2023-01-02", periods=session_count)
    close = pd.Series(np.linspace(100, 200, session_count), index=dates)
    close.iloc[: session_count - n_valid] = np.nan
    return pd.DataFrame(
        {"Open": 100.0, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e7},
        index=dates,
    )


def test_one_momentum_uncomputable_symbol_is_excluded_not_fatal(config: ScreenerConfig):
    price_frames = {"GOOD": _sparse_close_frame(400), "THIN": _sparse_close_frame(200)}
    fundamentals = {"GOOD": _fundamentals("GOOD"), "THIN": _fundamentals("THIN")}

    shortlist = build_shortlist(price_frames, fundamentals, config, top_n=5)

    assert [entry.symbol for entry in shortlist.entries] == ["GOOD"]
    assert shortlist.excluded_count["momentum_uncomputable"] == 1
    assert shortlist.qualifying_count == 1


def test_all_symbols_momentum_uncomputable_gives_empty_shortlist(config: ScreenerConfig):
    price_frames = {"THIN1": _sparse_close_frame(200), "THIN2": _sparse_close_frame(150)}
    fundamentals = {"THIN1": _fundamentals("THIN1"), "THIN2": _fundamentals("THIN2")}

    shortlist = build_shortlist(price_frames, fundamentals, config, top_n=5)

    assert shortlist.entries == []
    assert shortlist.qualifying_count == 0
    assert shortlist.excluded_count["momentum_uncomputable"] == 2
    assert shortlist.shortfall_note is not None


def test_momentum_only_path_excludes_uncomputable_symbol(config: ScreenerConfig):
    price_frames = {"GOOD": _sparse_close_frame(400), "THIN": _sparse_close_frame(200)}

    shortlist = build_momentum_only_shortlist(price_frames, config, top_n=5)

    assert [entry.symbol for entry in shortlist.entries] == ["GOOD"]
    assert shortlist.excluded_count["momentum_uncomputable"] == 1
    assert shortlist.fundamentals_included is False


def test_momentum_only_path_all_uncomputable_gives_empty_shortlist(config: ScreenerConfig):
    price_frames = {"THIN1": _sparse_close_frame(200), "THIN2": _sparse_close_frame(150)}

    shortlist = build_momentum_only_shortlist(price_frames, config, top_n=5)

    assert shortlist.entries == []
    assert shortlist.qualifying_count == 0
    assert shortlist.excluded_count["momentum_uncomputable"] == 2
    assert shortlist.shortfall_note is not None


# -- score_basis makes the meaning of composite_score explicit ---------------


def test_score_basis_distinguishes_the_two_ranking_paths(config: ScreenerConfig):
    price_frames = {
        "GOOD1": _price_frame(320, daily_return=0.002, volume=300_000.0),
        "GOOD2": _price_frame(320, daily_return=0.001, volume=300_000.0),
    }
    fundamentals = {"GOOD1": _fundamentals("GOOD1"), "GOOD2": _fundamentals("GOOD2")}

    full = build_shortlist(price_frames, fundamentals, config, top_n=5)
    momentum_only = build_momentum_only_shortlist(price_frames, config, top_n=5)

    assert {entry.score_basis for entry in full.entries} == {"composite"}
    assert {entry.score_basis for entry in momentum_only.entries} == {"rs_only"}
    for entry in momentum_only.entries:
        assert entry.composite_score == entry.rs_percentile
        assert entry.f_score is None
