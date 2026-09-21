"""Command-line entry point.

Exit codes: 0 success, 1 data integrity failure, 2 insufficient qualifying
names, 3 configuration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console

from nse_screener.config import ScreenerConfig
from nse_screener.data import (
    DataIntegrityError,
    FundamentalData,
    InsufficientFundamentalsError,
    UnsupportedStatementFormatError,
    fetch_fundamentals,
    fetch_price_history,
    validate_price_data,
)
from nse_screener.gate import evaluate_market_gate
from nse_screener.ranking import build_momentum_only_shortlist, build_shortlist
from nse_screener.report import RunMetadata, build_json_payload, print_report
from nse_screener.tracker import TrackedEntry, TrackedRun, log_run
from nse_screener.universe import fetch_full_nse_universe

#: A fundamentals period older than this (relative to the last price session)
#: means Yahoo has a newer fiscal year listed that is not yet fully reported;
#: the run still uses the last complete year, and says so.
STALE_FUNDAMENTALS_DAYS: int = 456  # ~15 months


def _read_universe(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"universe file not found: {path}")
    symbols = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not symbols:
        raise ValueError(f"universe file {path} contains no symbols")
    return symbols


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nse_screener", description="Evidence-based NSE equity screener"
    )
    parser.add_argument("--universe", type=Path, default=None, help="Path to universe.txt")
    parser.add_argument(
        "--full-market",
        action="store_true",
        help=(
            "Fetch the full live NSE-listed (EQ-series) equity universe instead of "
            "reading --universe. Slower and more exposed to source failures than a "
            "curated list -- see README."
        ),
    )
    parser.add_argument("--top", type=int, default=5, help="Shortlist size (default 5)")
    parser.add_argument(
        "--json", type=Path, default=None, help="Also write a JSON report to this path"
    )
    parser.add_argument(
        "--skip-fundamentals",
        action="store_true",
        help="Momentum + gate only. Skips quality/F-Score/forensic factors.",
    )
    parser.add_argument(
        "--track",
        type=Path,
        default=None,
        help=(
            "Append this run's shortlist (symbol + entry price) to a log file. "
            "Evaluate matured runs later with `python -m nse_screener.tracker "
            "--log PATH`."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console()

    try:
        config = ScreenerConfig(top_n=args.top)
        if bool(args.universe) == bool(args.full_market):
            raise ValueError("specify exactly one of --universe PATH or --full-market")
        if args.full_market:
            console.print("[bold]Fetching the live NSE equity list...[/bold]")
            symbols = fetch_full_nse_universe()
            console.print(
                f"[bold yellow]{len(symbols)} EQ-series symbols fetched. Screening all "
                "of them will take a while, especially with fundamentals enabled (one "
                "network call per symbol) -- consider --skip-fundamentals for a "
                "full-market run.[/bold yellow]"
            )
        else:
            symbols = _read_universe(args.universe)
    except Exception as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        return 3

    console.print(
        f"[bold]Minimum holding horizon: {config.holding_horizon_sessions} sessions "
        "(~3 months). This tool does not support a shorter-horizon mode.[/bold]"
    )
    if args.skip_fundamentals:
        console.print(
            "[bold yellow]WARNING: --skip-fundamentals uses only momentum + gate, "
            "the weakest-evidenced subset of this method. Quality, F-Score, and "
            "forensic factors are skipped entirely.[/bold yellow]"
        )

    try:
        price_frames, unavailable_symbols = fetch_price_history(
            symbols, period=config.price_history_period, index_symbol=config.index_symbol
        )
        if unavailable_symbols:
            console.print(
                f"[yellow]No Yahoo price data for {len(unavailable_symbols)} symbol(s) "
                f"(delisted/renamed/unresolvable), excluded: "
                f"{', '.join(unavailable_symbols[:20])}"
                f"{' ...' if len(unavailable_symbols) > 20 else ''}[/yellow]"
            )
        index_frame = price_frames.pop(config.index_symbol)
        validation = validate_price_data(
            price_frames,
            index_frame,
            min_sessions=config.min_sessions,
            calendar_alignment_lookback=config.calendar_alignment_lookback,
            max_missing_calendar_fraction=config.max_missing_calendar_fraction,
            max_recency_gap_days=config.max_recency_gap_days,
            impossible_move_abs_return=config.impossible_move_abs_return,
        )
    except DataIntegrityError as exc:
        console.print(f"[bold red]Data integrity failure:[/bold red] {exc}")
        return 1

    accepted_frames = {symbol: price_frames[symbol] for symbol in validation.accepted}

    gate = evaluate_market_gate(
        index_frame["Close"],
        pd.DataFrame({symbol: frame["Close"] for symbol, frame in accepted_frames.items()}),
        ma_short=config.gate_ma_short,
        ma_long=config.gate_ma_long,
        slope_lookback=config.gate_slope_lookback,
        breadth_high_proximity=config.gate_breadth_high_proximity,
    )

    if args.skip_fundamentals:
        shortlist = build_momentum_only_shortlist(accepted_frames, config, top_n=args.top)
    else:
        fundamentals: dict[str, FundamentalData] = {}
        unsupported_format: list[str] = []
        for symbol in validation.accepted:
            try:
                fundamentals[symbol] = fetch_fundamentals(symbol)
            except UnsupportedStatementFormatError:
                unsupported_format.append(symbol)
            except InsufficientFundamentalsError as exc:
                console.print(f"[yellow]Excluding {symbol}: {exc}[/yellow]")
        if unsupported_format:
            console.print(
                f"[yellow]Excluding {len(unsupported_format)} financial-sector name(s) "
                "(unclassified balance sheet -- F-Score, gross profitability and the "
                f"forensic screen are not defined for banks/NBFCs): "
                f"{', '.join(unsupported_format)}[/yellow]"
            )
        last_session = index_frame.sort_index().index[-1].date()
        for symbol, fdata in fundamentals.items():
            if (last_session - fdata.period_end).days > STALE_FUNDAMENTALS_DAYS:
                console.print(
                    f"[yellow]{symbol}: latest complete fundamentals period ends "
                    f"{fdata.period_end} -- a newer period exists on Yahoo but is not yet "
                    "fully reported[/yellow]"
                )
        shortlist = build_shortlist(accepted_frames, fundamentals, config, top_n=args.top)

    metadata = RunMetadata(
        run_date=date.today(),
        universe_size=len(symbols),
        data_window=config.price_history_period,
        last_session_date=index_frame.sort_index().index[-1].date(),
    )

    print_report(console, metadata, validation, gate, shortlist, args.skip_fundamentals)

    if args.json is not None:
        payload = build_json_payload(
            metadata, validation, gate, shortlist, config, args.skip_fundamentals
        )
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        console.print(f"JSON report written to {args.json}")

    if args.track is not None and shortlist.entries:
        tracked_run = TrackedRun(
            run_date=metadata.run_date,
            gate_verdict=gate.verdict.value,
            index_close_at_run=float(index_frame["Close"].sort_index().iloc[-1]),
            holding_horizon_sessions=config.holding_horizon_sessions,
            entries=[
                TrackedEntry(
                    symbol=entry.symbol,
                    rank=rank,
                    composite_score=entry.composite_score,
                    entry_close=float(accepted_frames[entry.symbol]["Close"].sort_index().iloc[-1]),
                )
                for rank, entry in enumerate(shortlist.entries, start=1)
            ],
        )
        log_run(tracked_run, args.track)
        console.print(
            f"Logged {len(tracked_run.entries)} pick(s) to {args.track} for later evaluation."
        )

    if not shortlist.entries:
        console.print("[bold red]No symbols qualified for the shortlist.[/bold red]")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
