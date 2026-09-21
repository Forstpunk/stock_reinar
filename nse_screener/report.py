"""Terminal and JSON reporting."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nse_screener.config import ScreenerConfig
from nse_screener.data import ValidationReport
from nse_screener.gate import GateVerdict, Verdict
from nse_screener.ranking import Shortlist

STANDING_CAVEATS: list[str] = [
    "- Minimum holding horizon: 63 sessions (~3 months). Shorter holds have",
    "  negative expected net edge after costs on this factor set.",
    "- Universe is currently-listed names only: results carry survivorship",
    "  bias and are biased upward.",
    "- Data source is an unofficial feed with verified missing sessions.",
    "- Factor weights are unvalidated and not optimised.",
    "- Momentum returns are negatively skewed: occasional severe drawdowns",
    "  are an expected property, not a malfunction.",
]


class RunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_date: date
    universe_size: int
    data_window: str
    last_session_date: date


def _output_label(verdict: Verdict) -> str:
    if verdict == Verdict.HOSTILE:
        return "WATCHLIST -- reduced or zero exposure indicated"
    return "SHORTLIST -- evidence-ranked candidates, not investment advice"


def print_report(
    console: Console,
    metadata: RunMetadata,
    validation: ValidationReport,
    gate: GateVerdict,
    shortlist: Shortlist,
    skip_fundamentals: bool,
) -> None:
    _print_header(console, metadata)
    _print_integrity(console, validation)
    _print_gate(console, gate)

    if skip_fundamentals:
        console.print(
            Panel(
                "[bold yellow]--skip-fundamentals was set: this run uses only the "
                "momentum + gate subset, the WEAKEST-EVIDENCED part of this method. "
                "Quality, F-Score, and forensic factors were not applied.[/bold yellow]",
                border_style="yellow",
            )
        )

    console.print(Panel(f"[bold]{_output_label(gate.verdict)}[/bold]", border_style="cyan"))
    _print_shortlist_table(console, shortlist)
    _print_per_name_detail(console, shortlist)
    _print_disqualified(console, shortlist)
    _print_excluded_summary(console, shortlist)
    _print_caveats(console)


def _print_header(console: Console, metadata: RunMetadata) -> None:
    console.rule("[bold]NSE Equity Screener[/bold]")
    console.print(f"Run date: {metadata.run_date}")
    console.print(f"Universe size: {metadata.universe_size}")
    console.print(f"Data window: {metadata.data_window}")
    console.print(f"Last session: {metadata.last_session_date}")


def _print_integrity(console: Console, validation: ValidationReport) -> None:
    console.rule("Data integrity")
    console.print(f"Accepted: {len(validation.accepted)} symbols")

    if validation.rejected:
        table = Table(title="Rejected symbols")
        table.add_column("Symbol")
        table.add_column("Reason")
        for symbol, reason in validation.rejected.items():
            table.add_row(symbol, reason)
        console.print(table)
    else:
        console.print("No symbols rejected.")

    if validation.flagged:
        table = Table(title="Flagged anomalies (not rejected -- for review)")
        table.add_column("Symbol")
        table.add_column("Date")
        table.add_column("Abs. move")
        for symbol, flag_date, move in validation.flagged:
            table.add_row(symbol, str(flag_date), f"{move:.1%}")
        console.print(table)


def _print_gate(console: Console, gate: GateVerdict) -> None:
    console.rule("Market gate")
    style = {"HEALTHY": "green", "NEUTRAL": "yellow", "HOSTILE": "red"}[gate.verdict.value]
    console.print(Panel(f"[bold {style}]{gate.verdict.value}[/bold {style}]", title="Verdict"))
    console.print(f"Index close: {gate.index_close:.2f}")
    console.print(f"Index vs 50DMA: {gate.index_vs_50dma_pct:+.2%}")
    console.print(f"Index vs 200DMA: {gate.index_vs_200dma_pct:+.2%}")
    console.print(f"200DMA slope (21d): {gate.dma200_slope_21d_pct:+.2%}")
    console.print(f"Breadth (trend-aligned): {gate.breadth_trend_aligned_pct:.1%}")
    console.print(f"Breadth (above 200DMA): {gate.breadth_above_ma200_pct:.1%}")
    console.print(f"Breadth (within 15% of 52w high): {gate.breadth_near_52w_high_pct:.1%}")
    console.print(f"Exposure guidance: [bold]{gate.exposure_guidance}[/bold]")


def _print_shortlist_table(console: Console, shortlist: Shortlist) -> None:
    console.rule("Shortlist")
    if shortlist.shortfall_note:
        console.print(f"[yellow]{shortlist.shortfall_note}[/yellow]")

    table = Table(title=f"Top {len(shortlist.entries)} of {shortlist.requested_top_n} requested")
    table.add_column("Rank")
    table.add_column("Symbol")
    table.add_column("Score")
    table.add_column("Basis")
    table.add_column("RS %ile")
    table.add_column("F-Score")
    table.add_column("Gross Profitability")
    table.add_column("ROE")
    table.add_column("Value %ile")
    table.add_column("Liquidity (Cr)")
    table.add_column("ATR %")

    for rank, entry in enumerate(shortlist.entries, start=1):
        f_score_cell = f"{entry.f_score.total_score}/9" if entry.f_score else "n/a"
        gp_cell = (
            f"{entry.gross_profitability:.2%}" if entry.gross_profitability is not None else "n/a"
        )
        roe_cell = f"{entry.roe:.2%}" if entry.roe is not None else "n/a"
        value_cell = (
            f"{entry.value_percentile:.0%}" if entry.value_percentile is not None else "n/a"
        )
        table.add_row(
            str(rank),
            entry.symbol,
            f"{entry.composite_score:.3f}",
            _basis_label(entry.score_basis),
            f"{entry.rs_percentile:.0%}",
            f_score_cell,
            gp_cell,
            roe_cell,
            value_cell,
            f"{entry.avg_traded_value_crore:.1f}",
            f"{entry.atr_pct:.1f}%",
        )
    console.print(table)
    if any(entry.score_basis == "rs_only" for entry in shortlist.entries):
        console.print(
            "[yellow]Basis 'RS only': score is the relative-strength percentile alone, "
            "not the full composite -- not comparable with composite scores.[/yellow]"
        )
    horizon = shortlist.entries[0].holding_horizon_sessions if shortlist.entries else 63
    console.print(f"Holding horizon for every entry: {horizon} sessions")


def _basis_label(score_basis: str) -> str:
    return {"composite": "composite", "rs_only": "RS only"}[score_basis]


def _print_per_name_detail(console: Console, shortlist: Shortlist) -> None:
    if not shortlist.entries:
        return
    console.rule("Per-name detail")
    for entry in shortlist.entries:
        f = entry.f_score
        console.print(
            f"[bold]{entry.symbol}[/bold] -- score {entry.composite_score:.3f} "
            f"({_basis_label(entry.score_basis)})"
        )
        if f is not None:
            console.print(
                "  F-Score breakdown: "
                f"profitability {f.profitability_score}/4 "
                f"(ROA>0={f.roa_positive}, CFO>0={f.cfo_positive}, "
                f"ROA improved={f.roa_improved}, accruals quality={f.accruals_quality}), "
                f"leverage/liquidity {f.leverage_liquidity_score}/3 "
                f"(leverage down={f.leverage_decreased}, "
                f"current ratio up={f.current_ratio_improved}, "
                f"no new shares={f.no_share_issuance}), "
                f"efficiency {f.efficiency_score}/2 "
                f"(gross margin up={f.gross_margin_improved}, "
                f"asset turnover up={f.asset_turnover_improved})"
            )
        else:
            console.print("  F-Score breakdown: n/a (--skip-fundamentals)")
        console.print(f"  Distance from 52w high: {entry.distance_from_52w_high_pct:.1f}%")
        if entry.forensic_flags:
            console.print(f"  [yellow]Forensic flag: {entry.forensic_flags[0]}[/yellow]")
        else:
            console.print("  Forensic flags: none")


def _print_disqualified(console: Console, shortlist: Shortlist) -> None:
    console.rule("Disqualified (forensic screen)")
    if not shortlist.disqualified:
        console.print("None disqualified.")
        return
    table = Table()
    table.add_column("Symbol")
    table.add_column("Flags")
    for symbol, flags in shortlist.disqualified.items():
        table.add_row(symbol, "; ".join(flags))
    console.print(table)


def _print_excluded_summary(console: Console, shortlist: Shortlist) -> None:
    console.rule("Excluded summary")
    if not shortlist.excluded_count:
        console.print("No exclusions.")
        return
    table = Table()
    table.add_column("Reason")
    table.add_column("Count")
    for reason, count in shortlist.excluded_count.items():
        table.add_row(reason, str(count))
    console.print(table)


def _print_caveats(console: Console) -> None:
    console.rule("Caveats")
    console.print("\n".join(STANDING_CAVEATS))


def build_json_payload(
    metadata: RunMetadata,
    validation: ValidationReport,
    gate: GateVerdict,
    shortlist: Shortlist,
    config: ScreenerConfig,
    skip_fundamentals: bool,
) -> dict[str, Any]:
    return {
        "metadata": metadata.model_dump(mode="json"),
        "skip_fundamentals": skip_fundamentals,
        "output_label": _output_label(gate.verdict),
        "data_integrity": validation.model_dump(mode="json"),
        "market_gate": gate.model_dump(mode="json"),
        "shortlist": shortlist.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
        "caveats": STANDING_CAVEATS,
    }
