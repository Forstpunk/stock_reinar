"""Outcome tracking: log every shortlist, evaluate it against real forward returns.

This is the empirical check on the whole tool: every backtest claim baked
into ``config.py``'s constants came from the original design brief, not
from independently verifying NSE data. ``log_run`` records what the
screener actually picked and at what price; ``evaluate_matured_runs``
later fetches real prices and reports what each pick actually did versus
the Nifty over the identical window, once the mandatory 63-session
holding horizon has elapsed. No backtest, no simulation -- only runs that
have genuinely aged past the horizon are evaluated.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from nse_screener.config import HOLDING_HORIZON_SESSIONS


class TrackedEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    rank: int
    composite_score: float
    entry_close: float


class TrackedRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_date: date
    gate_verdict: str
    index_close_at_run: float
    holding_horizon_sessions: int
    entries: list[TrackedEntry]


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_date: date
    symbol: str
    rank: int
    gate_verdict_at_entry: str
    entry_close: float
    exit_date: date
    exit_close: float
    stock_return_pct: float
    index_entry_close: float
    index_exit_close: float
    index_return_pct: float
    excess_return_pct: float


def log_run(run: TrackedRun, log_path: Path) -> None:
    """Append one run to the log. Never overwrites or rewrites prior entries."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(run.model_dump_json() + "\n")


def load_runs(log_path: Path) -> list[TrackedRun]:
    if not log_path.exists():
        return []
    runs: list[TrackedRun] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(TrackedRun.model_validate_json(line))
    return runs


def compute_evaluations(
    run: TrackedRun,
    prices: dict[str, pd.Series],
    index_symbol: str,
    min_sessions: int,
) -> list[EvaluationResult]:
    """Pure evaluation logic, given already-fetched forward price series.

    ``prices`` maps symbol (bare, plus ``index_symbol``) to a Close-price
    Series starting at or after ``run.run_date``, sorted ascending. A
    symbol is only evaluated if its series has more than ``min_sessions``
    points after entry -- otherwise the run hasn't matured yet for that
    name (or the symbol stopped trading before maturing), and it is
    silently excluded from the results rather than guessed at.
    """
    index_series = prices.get(index_symbol)
    if index_series is None or len(index_series) <= min_sessions:
        return []

    index_exit_close = float(index_series.iloc[min_sessions])
    index_return_pct = (index_exit_close / run.index_close_at_run - 1.0) * 100

    results: list[EvaluationResult] = []
    for entry in run.entries:
        series = prices.get(entry.symbol)
        if series is None or len(series) <= min_sessions:
            continue

        exit_close = float(series.iloc[min_sessions])
        exit_date = series.index[min_sessions]
        exit_date = exit_date.date() if hasattr(exit_date, "date") else exit_date
        stock_return_pct = (exit_close / entry.entry_close - 1.0) * 100

        results.append(
            EvaluationResult(
                run_date=run.run_date,
                symbol=entry.symbol,
                rank=entry.rank,
                gate_verdict_at_entry=run.gate_verdict,
                entry_close=entry.entry_close,
                exit_date=exit_date,
                exit_close=exit_close,
                stock_return_pct=stock_return_pct,
                index_entry_close=run.index_close_at_run,
                index_exit_close=index_exit_close,
                index_return_pct=index_return_pct,
                excess_return_pct=stock_return_pct - index_return_pct,
            )
        )
    return results


def _fetch_forward_prices(
    symbols: list[str], index_symbol: str, start: date
) -> dict[str, pd.Series]:
    ns_symbols = [f"{symbol}.NS" for symbol in symbols]
    all_symbols = ns_symbols + [index_symbol]

    raw = yf.download(
        all_symbols,
        start=start.isoformat(),
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if raw is None or raw.empty:
        return {}

    prices: dict[str, pd.Series] = {}
    n = len(all_symbols)
    for bare_symbol, ns_symbol in zip(symbols, ns_symbols):
        frame = raw[ns_symbol] if n > 1 else raw
        if frame is None:
            continue
        closes = frame["Close"].dropna().sort_index()
        if not closes.empty:
            prices[bare_symbol] = closes

    index_frame = raw[index_symbol] if n > 1 else raw
    if index_frame is not None:
        index_closes = index_frame["Close"].dropna().sort_index()
        if not index_closes.empty:
            prices[index_symbol] = index_closes

    return prices


def evaluate_matured_runs(
    log_path: Path, index_symbol: str = "^NSEI", min_sessions: int = 63
) -> tuple[list[EvaluationResult], list[TrackedRun]]:
    """Fetch real forward prices and evaluate every logged run.

    Returns ``(results, unmatured_runs)``: ``results`` covers every
    (run, symbol) pair old enough to have more than ``min_sessions``
    forward sessions of price data; ``unmatured_runs`` lists runs too
    recent to evaluate yet, so the caller can report "still pending"
    rather than silently omitting them.
    """
    runs = load_runs(log_path)
    results: list[EvaluationResult] = []
    unmatured: list[TrackedRun] = []

    for run in runs:
        symbols = [entry.symbol for entry in run.entries]
        prices = _fetch_forward_prices(symbols, index_symbol, run.run_date)
        run_results = compute_evaluations(run, prices, index_symbol, min_sessions)
        if run_results:
            results.extend(run_results)
        else:
            unmatured.append(run)

    return results, unmatured


def _print_evaluation(
    console: Console,
    results: list[EvaluationResult],
    unmatured: list[TrackedRun],
    min_sessions: int,
) -> None:
    console.rule("[bold]Shortlist outcome tracker[/bold]")

    if not results:
        console.print(
            f"[yellow]No runs have matured past {min_sessions} sessions yet -- "
            "nothing to evaluate.[/yellow]"
        )
    else:
        table = Table(title=f"Real forward returns, {min_sessions} sessions after entry")
        table.add_column("Run date")
        table.add_column("Symbol")
        table.add_column("Rank")
        table.add_column("Gate@entry")
        table.add_column("Entry close")
        table.add_column("Exit close")
        table.add_column("Stock return")
        table.add_column("Index return")
        table.add_column("Excess vs index")

        for r in sorted(results, key=lambda x: (x.run_date, x.rank)):
            style = "green" if r.excess_return_pct > 0 else "red"
            table.add_row(
                str(r.run_date),
                r.symbol,
                str(r.rank),
                r.gate_verdict_at_entry,
                f"{r.entry_close:.2f}",
                f"{r.exit_close:.2f}",
                f"{r.stock_return_pct:+.2f}%",
                f"{r.index_return_pct:+.2f}%",
                f"[{style}]{r.excess_return_pct:+.2f}%[/{style}]",
            )
        console.print(table)

        wins = sum(1 for r in results if r.excess_return_pct > 0)
        avg_excess = sum(r.excess_return_pct for r in results) / len(results)
        avg_stock = sum(r.stock_return_pct for r in results) / len(results)
        avg_index = sum(r.index_return_pct for r in results) / len(results)
        console.print(
            f"\n[bold]{len(results)} evaluated picks[/bold] -- "
            f"beat the index on {wins}/{len(results)} ({wins / len(results):.0%}), "
            f"average stock return {avg_stock:+.2f}%, "
            f"average index return {avg_index:+.2f}%, "
            f"average excess return {avg_excess:+.2f}%"
        )

    if unmatured:
        console.print(
            f"\n[dim]{len(unmatured)} run(s) not yet {min_sessions} sessions old, "
            f"pending: {', '.join(str(run.run_date) for run in unmatured)}[/dim]"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nse_screener.tracker",
        description="Evaluate past screener shortlists against real forward returns",
    )
    parser.add_argument(
        "--log", type=Path, default=Path("tracker.jsonl"), help="Path to the tracker log file"
    )
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=HOLDING_HORIZON_SESSIONS,
        help="Sessions after entry to measure the exit price at (default: the tool's "
        f"{HOLDING_HORIZON_SESSIONS}-session minimum holding horizon)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console()

    if not args.log.exists():
        console.print(
            f"[bold red]No tracker log found at {args.log}.[/bold red] Run the "
            f"screener with --track {args.log} at least once first."
        )
        return 1

    results, unmatured = evaluate_matured_runs(args.log, min_sessions=args.min_sessions)
    _print_evaluation(console, results, unmatured, args.min_sessions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
