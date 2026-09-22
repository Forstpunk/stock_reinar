"""Workbench CLI entry point: ``python -m nse_screener.workbench <command> SYMBOL ...``

Each command that requires a prior stage raises with a message naming
the missing stage (via ``session.StageOutOfOrderError``) -- it never
auto-runs a prior stage on the analyst's behalf.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table

from nse_screener.data import fetch_fundamentals, fetch_price_history, validate_price_data
from nse_screener.experimental.busted import BustedPattern, detect_busted_patterns
from nse_screener.experimental.validation_log import (
    DetectionLogEntry,
    load_detections,
    log_detection,
    score_open_detections,
)
from nse_screener.factors.forensic import forensic_screen
from nse_screener.fundamentals.distress import altman_z_score
from nse_screener.fundamentals.live_data import LiveDataError, LiveFinancials, fetch_live_financials
from nse_screener.fundamentals.multiples import compute_multiples
from nse_screener.fundamentals.peers import NotApplicableForSector, SectorInfo, load_sector_map
from nse_screener.fundamentals.reorganize import ReorganizationError, reorganize
from nse_screener.fundamentals.roic import DecompositionError, roic_analysis
from nse_screener.fundamentals.wacc import InsufficientReturnHistoryError, WACCInputs, compute_wacc
from nse_screener.gate import GateVerdict, evaluate_market_gate
from nse_screener.workbench.expectations import (
    NoSolutionError,
    SteadyStateViolationError,
    UnstableSolutionError,
    two_stage_implied_growth,
)
from nse_screener.workbench.mscore import (
    MScoreDataError,
    ReasoningRequiredError,
    beneish_m_score,
    combine_accounting_verdict,
)
from nse_screener.workbench.session import (
    AnalysisSession,
    AnalysisStage,
    BusinessContext,
    FinancialSummary,
    ProspectiveAnalysis,
    StageOutOfOrderError,
    complete_session,
    enter_accounting_verdict,
    enter_business_context,
    enter_financial_summary,
    enter_prospective,
    load_session,
    missing_requirements,
    save_session,
    start_session,
)
from nse_screener.workbench.sizing import RiskParameterError, calculate_position
from nse_screener.workbench.valuation import VALUE_RANGE_CAVEAT, apply_graham_label, value_range

#: Expected, meaningful domain errors every command can raise -- printed
#: cleanly, never a raw traceback. A result like UnstableSolutionError is
#: often the model correctly flagging something, not a bug.
_DOMAIN_ERRORS = (
    LiveDataError,
    ReorganizationError,
    NotApplicableForSector,
    DecompositionError,
    InsufficientReturnHistoryError,
    NoSolutionError,
    SteadyStateViolationError,
    UnstableSolutionError,
    MScoreDataError,
    ReasoningRequiredError,
    StageOutOfOrderError,
    RiskParameterError,
    ValueError,
)

SESSIONS_ROOT = Path("sessions")
DEFAULT_SECTOR_MAP = Path("data/sector_map.csv")


def _load_or_error(console: Console, symbol: str) -> AnalysisSession:
    matches = sorted(SESSIONS_ROOT.glob(f"{symbol}_*.json"))
    if not matches:
        console.print(
            f"[bold red]No session for {symbol}.[/bold red] Run "
            f"`python -m nse_screener.workbench start {symbol}` first."
        )
        raise SystemExit(1)
    return load_session(matches[-1])


def _sector_info(symbol: str) -> SectorInfo:
    sector_map = load_sector_map(DEFAULT_SECTOR_MAP)
    if symbol not in sector_map:
        raise SystemExit(
            f"{symbol} is not in {DEFAULT_SECTOR_MAP} -- add it before running the workbench."
        )
    return sector_map[symbol]


def _live_gate(symbol: str) -> GateVerdict:
    price_frames, _ = fetch_price_history([symbol])
    index_frame = price_frames.pop("^NSEI")
    validation = validate_price_data(price_frames, index_frame)
    if symbol not in validation.accepted:
        raise SystemExit(
            f"{symbol}: failed price data validation: {validation.rejected.get(symbol)}"
        )
    closes = pd.DataFrame({symbol: price_frames[symbol]["Close"]})
    return evaluate_market_gate(index_frame["Close"], closes)


def _raw_atr(history: pd.DataFrame, window: int = 14) -> float:
    high, low, close = history["High"], history["Low"], history["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(window).mean().iloc[-1]
    if pd.isna(atr):
        raise SystemExit(f"insufficient price history to compute ATR({window})")
    return float(atr)


def _years_listed(symbol: str) -> int:
    history = yf.Ticker(f"{symbol}.NS").history(period="max")
    if history is None or history.empty:
        return 0
    span_days = int((history.index[-1] - history.index[0]).days)
    return max(span_days // 365, 0)


def _parse_scenarios(raw: str) -> dict[str, float]:
    scenarios: dict[str, float] = {}
    for part in raw.split(","):
        name, _, rate = part.partition("=")
        if not name or not rate:
            raise SystemExit(f"bad --scenarios entry {part!r}, expected name=rate")
        scenarios[name.strip()] = float(rate)
    return scenarios


_CONTEXT_SECTIONS = ("industry economics", "competitive position", "revenue drivers")


def _parse_context_file(path: Path) -> tuple[str, str, list[str]]:
    """Parses a simple three-section markdown-ish file: ``INDUSTRY
    ECONOMICS:``, ``COMPETITIVE POSITION:``, ``REVENUE DRIVERS:``
    (case-insensitive), drivers as ``- `` bullet lines."""
    text = path.read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {name: [] for name in _CONTEXT_SECTIONS}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower().rstrip(":")
        if lowered in sections:
            current = lowered
            continue
        if current is not None and stripped:
            sections[current].append(stripped)

    industry_economics = " ".join(sections["industry economics"])
    competitive_position = " ".join(sections["competitive position"])
    revenue_drivers = [d.lstrip("-").strip() for d in sections["revenue drivers"] if d.strip()]
    return industry_economics, competitive_position, revenue_drivers


def _wacc_inputs(symbol: str, lf: LiveFinancials, effective_tax_rate: float) -> WACCInputs:
    return WACCInputs(
        symbol=symbol,
        market_cap=lf.close_price * lf.shares_outstanding,
        total_debt=lf.total_debt,
        prior_total_debt=lf.prior_total_debt,
        interest_expense=lf.interest_expense,
        effective_tax_rate=effective_tax_rate,
    )


def _price_history(symbol: str) -> tuple[pd.Series, pd.Series]:
    hist = yf.Ticker(f"{symbol}.NS").history(period="2y")["Close"]
    index_hist = yf.Ticker("^NSEI").history(period="2y")["Close"]
    return hist, index_hist


def cmd_start(args: argparse.Namespace, console: Console) -> None:
    existing = sorted(SESSIONS_ROOT.glob(f"{args.symbol}_*.json"))
    if existing:
        console.print(f"[yellow]Resuming existing session {existing[-1].name}[/yellow]")
        session = load_session(existing[-1])
    else:
        session = start_session(args.symbol)
        path = save_session(session, SESSIONS_ROOT)
        console.print(f"Started session for {args.symbol} at {path}")
    console.print(f"Stage: {session.stage.value}")


def cmd_context(args: argparse.Namespace, console: Console) -> None:
    session = _load_or_error(console, args.symbol)
    industry_economics, competitive_position, revenue_drivers = _parse_context_file(
        Path(args.file)
    )
    context = BusinessContext(
        industry_economics=industry_economics,
        competitive_position=competitive_position,
        revenue_drivers=revenue_drivers,
        analyst=args.analyst,
        entered_at=datetime.now(),
    )
    session = enter_business_context(session, context)
    save_session(session, SESSIONS_ROOT)
    console.print(
        f"[green]Business context recorded for {args.symbol}.[/green] "
        f"Stage: {session.stage.value}"
    )


def cmd_accounting(args: argparse.Namespace, console: Console) -> None:
    session = _load_or_error(console, args.symbol)
    sector = _sector_info(args.symbol)

    if sector.requires_specialist_analysis:
        # Checked before any live fetch: a financial-sector company's
        # statements don't have the fields fetch_live_financials needs
        # (gross_profit, operating_income, current_assets, ...), so
        # fetching first would only fail with a confusing "missing
        # fields" error instead of this clear, immediate explanation.
        console.print(
            f"[bold red]{args.symbol} is a financial-sector company.[/bold red] The forensic "
            "screen, M-Score and Z-score are not defined for it -- accounting quality here "
            "needs a specialist toolkit this system does not provide."
        )
        raise SystemExit(1)

    lf = fetch_live_financials(args.symbol)
    fdata = fetch_fundamentals(args.symbol)
    if fdata.prior is None:
        raise SystemExit(f"{args.symbol}: fewer than 2 years of fundamentals available")
    forensic = forensic_screen(fdata, fdata.prior)
    m_score = beneish_m_score(lf.mscore_inputs)
    distress = altman_z_score(lf.distress_inputs, sector)

    verdict = combine_accounting_verdict(
        forensic.flags,
        m_score,
        analyst=args.analyst,
        entered_at=datetime.now(),
        distress_verdict=distress.verdict,
        reasoning=args.reasoning or "",
    )
    session = enter_accounting_verdict(session, verdict)
    save_session(session, SESSIONS_ROOT)

    console.print(
        f"M-Score: {m_score.m_score:.2f} ({m_score.band}); "
        f"Z-Score: {distress.z_score:.2f} ({distress.verdict})"
    )
    console.print(f"[bold]Verdict: {verdict.verdict}[/bold]. Stage: {session.stage.value}")
    if session.terminal_state == "DISQUALIFIED":
        console.print(
            "[bold red]Session DISQUALIFIED -- no price is ever attached to this "
            "company.[/bold red]"
        )


def cmd_financial(args: argparse.Namespace, console: Console) -> None:
    session = _load_or_error(console, args.symbol)
    lf = fetch_live_financials(args.symbol)
    r = reorganize(lf.raw_operating)

    hist, index_hist = _price_history(args.symbol)
    wacc_inputs = _wacc_inputs(args.symbol, lf, r.effective_tax_rate)
    wacc = compute_wacc(
        wacc_inputs, hist, index_hist, args.risk_free_rate, args.equity_risk_premium
    )
    roic = roic_analysis(r, wacc)
    multiples = compute_multiples(
        lf.multiples_inputs,
        r,
        market_cap=lf.close_price * lf.shares_outstanding,
        net_debt=lf.net_debt,
    )

    summary = FinancialSummary(
        roic_analysis=roic, multiples=multiples, analyst=args.analyst, entered_at=datetime.now()
    )
    session = enter_financial_summary(session, summary)
    save_session(session, SESSIONS_ROOT)

    console.print(
        f"ROIC {roic.roic:.2%} vs WACC {wacc.wacc_point:.2%} -> "
        f"spread {roic.economic_spread:+.2%}"
    )
    console.print(
        f"P/E {multiples.pe}, EV/EBITDA {multiples.ev_ebitda}, "
        f"Total payout yield {multiples.total_payout_yield:.2%}"
    )
    console.print(f"Stage: {session.stage.value}")


def cmd_expectations(args: argparse.Namespace, console: Console) -> None:
    session = _load_or_error(console, args.symbol)
    lf = fetch_live_financials(args.symbol)
    r = reorganize(lf.raw_operating)

    hist, index_hist = _price_history(args.symbol)
    wacc_inputs = _wacc_inputs(args.symbol, lf, r.effective_tax_rate)
    wacc = compute_wacc(
        wacc_inputs, hist, index_hist, args.risk_free_rate, args.equity_risk_premium
    )
    roic = roic_analysis(r, wacc)

    enterprise_value = lf.close_price * lf.shares_outstanding + lf.net_debt
    result = two_stage_implied_growth(
        symbol=args.symbol,
        enterprise_value=enterprise_value,
        nopat=r.nopat,
        roic=roic.roic,
        wacc_low=wacc.wacc_low,
        wacc_point=wacc.wacc_point,
        wacc_high=wacc.wacc_high,
        years_listed=_years_listed(args.symbol),
        explicit_years=args.explicit_years,
    )

    prior_prospective = session.prospective
    prospective = ProspectiveAnalysis(
        implied_expectations=result,
        value_range=prior_prospective.value_range if prior_prospective else None,
        position_plan=prior_prospective.position_plan if prior_prospective else None,
        analyst=args.analyst,
        entered_at=datetime.now(),
    )
    session = enter_prospective(session, prospective)
    save_session(session, SESSIONS_ROOT)

    console.print(
        f"Implied growth (WACC low/point/high): {result.implied_growth_at_wacc_low:.2%} / "
        f"{result.implied_growth_at_wacc_point:.2%} / {result.implied_growth_at_wacc_high:.2%}"
    )
    dominant = " -- TERMINAL VALUE DOMINANT" if result.terminal_value_dominant else ""
    console.print(
        f"Continuing value share of total: {result.continuing_value_share_of_total:.1%}{dominant}"
    )
    console.print(
        "[bold yellow]PLAUSIBILITY ASSESSMENT REQUIRED[/bold yellow] -- run "
        f"`assess {args.symbol} --verdict ... --note ...`"
    )


def cmd_assess(args: argparse.Namespace, console: Console) -> None:
    session = _load_or_error(console, args.symbol)
    if session.prospective is None:
        console.print(f"[bold red]Run `expectations {args.symbol}` first.[/bold red]")
        raise SystemExit(1)
    updated_expectations = session.prospective.implied_expectations.model_copy(
        update={"plausibility_verdict": args.verdict, "plausibility_assessment": args.note}
    )
    prospective = session.prospective.model_copy(
        update={"implied_expectations": updated_expectations}
    )
    session = enter_prospective(session, prospective)
    save_session(session, SESSIONS_ROOT)
    console.print(f"Plausibility verdict recorded: {args.verdict}")


def cmd_value(args: argparse.Namespace, console: Console) -> None:
    session = _load_or_error(console, args.symbol)
    lf = fetch_live_financials(args.symbol)
    r = reorganize(lf.raw_operating)

    hist, index_hist = _price_history(args.symbol)
    wacc_inputs = _wacc_inputs(args.symbol, lf, r.effective_tax_rate)
    wacc = compute_wacc(
        wacc_inputs, hist, index_hist, args.risk_free_rate, args.equity_risk_premium
    )
    roic = roic_analysis(r, wacc)
    scenarios = _parse_scenarios(args.scenarios)

    result = value_range(
        symbol=args.symbol,
        nopat=r.nopat,
        roic=roic.roic,
        wacc=wacc.wacc_point,
        scenarios=scenarios,
        net_debt=lf.net_debt,
        shares=lf.shares_outstanding,
        current_price=lf.close_price,
    )

    if session.prospective is None:
        console.print(f"[bold red]Run `expectations {args.symbol}` first.[/bold red]")
        raise SystemExit(1)
    prospective = session.prospective.model_copy(update={"value_range": result})
    session = enter_prospective(session, prospective)
    save_session(session, SESSIONS_ROOT)

    table = Table(title=f"{args.symbol} value range")
    table.add_column("Scenario")
    table.add_column("Per-share value")
    for name, v in result.per_share_values.items():
        flag = " [UNSTABLE]" if name in result.unstable_scenarios else ""
        table.add_row(name + flag, f"{v:.2f}")
    console.print(table)
    console.print(
        f"Range: {result.low:.2f} - {result.high:.2f}. "
        f"Current price: {result.current_price:.2f} ({result.price_position})"
    )
    if result.unstable_scenarios:
        console.print(
            f"[bold red]UNSTABLE: {', '.join(result.unstable_scenarios)}[/bold red] -- growth "
            "too close to WACC, the formula's denominator is near zero. These values are not "
            "meaningful \"optimistic\" outcomes, they're numerical artifacts. Widen the gap "
            "from WACC and re-run."
        )
    console.print(f"[dim]{VALUE_RANGE_CAVEAT}[/dim]")
    console.print(
        "[bold yellow]GRAHAM LABEL REQUIRED[/bold yellow] -- run "
        f"`label {args.symbol} --label ...`"
    )


def cmd_label(args: argparse.Namespace, console: Console) -> None:
    session = _load_or_error(console, args.symbol)
    if session.prospective is None or session.prospective.value_range is None:
        console.print(f"[bold red]Run `value {args.symbol}` first.[/bold red]")
        raise SystemExit(1)
    labeled = apply_graham_label(session.prospective.value_range, args.label, args.justification)
    prospective = session.prospective.model_copy(update={"value_range": labeled})
    session = enter_prospective(session, prospective)
    save_session(session, SESSIONS_ROOT)
    console.print(f"Graham label recorded: {args.label}")


def cmd_size(args: argparse.Namespace, console: Console) -> None:
    session = _load_or_error(console, args.symbol)
    prospective = session.prospective
    value_range_result = prospective.value_range if prospective is not None else None
    graham_label = value_range_result.graham_label if value_range_result is not None else None
    if prospective is None or graham_label is None:
        console.print(f"[bold red]Run `label {args.symbol}` first.[/bold red]")
        raise SystemExit(1)

    gate = _live_gate(args.symbol)
    history = yf.Ticker(f"{args.symbol}.NS").history(period="3mo")
    entry_price = float(history["Close"].iloc[-1])
    atr = _raw_atr(history)

    plan = calculate_position(
        capital=args.capital,
        risk_pct=args.risk,
        entry_price=entry_price,
        atr_14=atr,
        gate_verdict=gate,
        graham_label=graham_label,
        atr_multiple=args.atr_multiple,
    )
    prospective = prospective.model_copy(update={"position_plan": plan})
    session = enter_prospective(session, prospective)
    save_session(session, SESSIONS_ROOT)

    console.print(
        f"Shares: {plan.shares}, position value: {plan.position_value:.0f}, "
        f"stop: {plan.stop_price:.2f}, permitted: {plan.permitted}"
    )

    missing = missing_requirements(session)
    if not missing:
        session = complete_session(session)
        save_session(session, SESSIONS_ROOT)
        console.print("[bold green]ANALYSIS_COMPLETE[/bold green]")
    else:
        console.print(f"[yellow]Still missing before complete: {', '.join(missing)}[/yellow]")


def _try_step(
    console: Console,
    fn: Callable[[argparse.Namespace, Console], None],
    ns: argparse.Namespace,
) -> bool:
    """Run one internal cmd_* step. Returns False (after printing the error
    cleanly) on any expected domain error, rather than aborting the whole
    `run` and losing everything computed so far -- the point of `run` is to
    show what's there AND what's wrong, in one pass."""
    try:
        fn(ns, console)
        return True
    except _DOMAIN_ERRORS as exc:
        console.print(f"[bold red]{type(exc).__name__}:[/bold red] {exc}")
        return False
    except SystemExit:
        return False


def cmd_run(args: argparse.Namespace, console: Console) -> None:
    """Run every *computable* stage in one shot, stopping cleanly the moment
    a human judgment call is needed (or a step fails), then always printing
    the full report of whatever was actually computed.

    Resumable: stages 1-3 (business context, accounting, financial) are
    one-shot and skipped if already done; stage 4's expectations/value are
    freely re-runnable and always refreshed against today's live data. This
    never fills in a plausibility verdict or a Graham label -- those stay
    separate commands, by design.
    """
    console.rule(f"[bold]{args.symbol} -- computing everything computable[/bold]")

    existing = sorted(SESSIONS_ROOT.glob(f"{args.symbol}_*.json"))
    session = load_session(existing[-1]) if existing else start_session(args.symbol)
    if not existing:
        save_session(session, SESSIONS_ROOT)

    if session.stage == AnalysisStage.BUSINESS_STRATEGY:
        if not args.context_file:
            console.print(
                "[bold yellow]No business context yet, and no --context-file given.[/bold yellow] "
                f"Run `context {args.symbol} --file ...` yourself -- this step is never "
                "auto-generated -- then re-run `run`."
            )
            _print_report(console, session)
            return
        ok = _try_step(
            console,
            cmd_context,
            argparse.Namespace(symbol=args.symbol, file=args.context_file, analyst=args.analyst),
        )
        session = _load_or_error(console, args.symbol)
        if not ok:
            _print_report(console, session)
            return

    if session.stage == AnalysisStage.ACCOUNTING_QUALITY:
        ok = _try_step(
            console,
            cmd_accounting,
            argparse.Namespace(symbol=args.symbol, reasoning=args.reasoning, analyst=args.analyst),
        )
        session = _load_or_error(console, args.symbol)
        if not ok or session.terminal_state == "DISQUALIFIED":
            _print_report(console, session)
            return

    if session.stage == AnalysisStage.FINANCIAL_ANALYSIS:
        ok = _try_step(
            console,
            cmd_financial,
            argparse.Namespace(
                symbol=args.symbol,
                risk_free_rate=args.risk_free_rate,
                equity_risk_premium=args.equity_risk_premium,
                analyst=args.analyst,
            ),
        )
        session = _load_or_error(console, args.symbol)
        if not ok:
            _print_report(console, session)
            return

    if session.stage == AnalysisStage.PROSPECTIVE:
        ok = _try_step(
            console,
            cmd_expectations,
            argparse.Namespace(
                symbol=args.symbol,
                risk_free_rate=args.risk_free_rate,
                equity_risk_premium=args.equity_risk_premium,
                explicit_years=args.explicit_years,
                analyst=args.analyst,
            ),
        )
        session = _load_or_error(console, args.symbol)

        if ok and args.scenarios:
            _try_step(
                console,
                cmd_value,
                argparse.Namespace(
                    symbol=args.symbol,
                    scenarios=args.scenarios,
                    risk_free_rate=args.risk_free_rate,
                    equity_risk_premium=args.equity_risk_premium,
                ),
            )
            session = _load_or_error(console, args.symbol)

    _print_report(console, session)


def _print_report(console: Console, session: AnalysisSession) -> None:
    console.print()
    cmd_report(argparse.Namespace(symbol=session.symbol), console)


def cmd_report(args: argparse.Namespace, console: Console) -> None:
    session = _load_or_error(console, args.symbol)
    console.rule(f"{args.symbol} -- {session.stage.value}")
    if session.business_context:
        bc = session.business_context
        console.print(f"[bold]Business context[/bold] ({bc.analyst}, {bc.entered_at}):")
        console.print(f"  Industry economics: {bc.industry_economics}")
        console.print(f"  Competitive position: {bc.competitive_position}")
        console.print(f"  Revenue drivers: {', '.join(bc.revenue_drivers)}")
    if session.accounting_verdict:
        av = session.accounting_verdict
        console.print(
            f"[bold]Accounting verdict[/bold]: {av.verdict} "
            f"(M-Score {av.m_score:.2f}/{av.m_score_band}, distress {av.distress_verdict}, "
            f"{len(av.forensic_flags)} forensic flags)"
        )
        if av.reasoning:
            console.print(f"  Reasoning: {av.reasoning}")
    if session.financial_summary and session.financial_summary.roic_analysis:
        roic = session.financial_summary.roic_analysis
        console.print(
            f"[bold]ROIC[/bold]: {roic.roic:.2%}, spread {roic.economic_spread:+.2%}, "
            f"value creating: {roic.value_creating}"
        )
    if session.prospective:
        ie = session.prospective.implied_expectations
        console.print(
            f"[bold]Implied growth[/bold]: {ie.implied_growth_at_wacc_point:.2%} "
            f"(plausibility: {ie.plausibility_verdict})"
        )
        if session.prospective.value_range:
            vr = session.prospective.value_range
            console.print(
                f"[bold]Value range[/bold]: {vr.low:.2f}-{vr.high:.2f}, label: {vr.graham_label}"
            )
        if session.prospective.position_plan:
            shares = session.prospective.position_plan.shares
            console.print(f"[bold]Position plan[/bold]: {shares} shares")

    console.print(f"[bold]Terminal state[/bold]: {session.terminal_state}")
    missing = missing_requirements(session)
    if session.terminal_state != "ANALYSIS_COMPLETE" and missing:
        console.print(f"[bold red]ANALYSIS INCOMPLETE[/bold red] -- missing: {', '.join(missing)}")


def _build_detection_entry(
    symbol: str, frame_close: "pd.Series", pattern: BustedPattern, gate_verdict: str
) -> DetectionLogEntry:
    """Build the forward-validation log entry for one confirmed busted pattern.

    ``detection_date`` must be ``bust_confirmation_date``, not
    ``breakout_date``: a pattern is only ever returned by
    ``detect_busted_patterns`` once its recovery has already happened
    (that recovery is what ``bust_confirmed`` means), so logging
    ``breakout_date`` as the entry point back-dates the entry to before
    the outcome was knowable -- forward-return scoring from there
    re-measures a move that already occurred, which is lookahead bias,
    not a real forward test.

    ``gate_verdict`` reflects the market gate at scan time, not as of
    this historical date -- like ``price_at_detection``, this is
    metadata only: ``score_open_detections`` never reads either field,
    it re-derives both from price history by date.
    """
    confirmation_date = pattern.bust_confirmation_date
    assert confirmation_date is not None  # always set when bust_confirmed
    return DetectionLogEntry(
        symbol=symbol,
        detection_date=confirmation_date,
        price_at_detection=float(frame_close.loc[pd.Timestamp(confirmation_date)]),
        gate_verdict_at_detection=gate_verdict,
        pattern_metadata={
            "bust_type": pattern.bust_type,
            "sessions_elapsed": pattern.sessions_elapsed,
        },
    )


def cmd_experimental_scan(args: argparse.Namespace, console: Console) -> None:
    console.print("[bold yellow]EXPERIMENTAL -- UNVALIDATED ON NSE[/bold yellow]")
    symbols = [s.strip() for s in Path(args.universe).read_text().splitlines() if s.strip()]
    price_frames, unavailable = fetch_price_history(symbols)
    index_frame = price_frames.pop("^NSEI")
    if unavailable:
        console.print(f"[yellow]Skipped (no data): {', '.join(unavailable)}[/yellow]")

    universe_closes = pd.DataFrame({s: f["Close"] for s, f in price_frames.items()})
    gate = evaluate_market_gate(index_frame["Close"], universe_closes)

    log_path = Path(args.log)
    total = 0
    for symbol, frame in price_frames.items():
        patterns = detect_busted_patterns(symbol, frame["Close"])
        for p in patterns:
            entry = _build_detection_entry(symbol, frame["Close"], p, gate.verdict.value)
            log_detection(entry, log_path)
            total += 1
    console.print(f"{total} detection(s) logged to {log_path}")


def cmd_validate_experimental(args: argparse.Namespace, console: Console) -> None:
    console.print("[bold yellow]EXPERIMENTAL -- UNVALIDATED ON NSE[/bold yellow]")
    log_path = Path(args.log)
    detections = load_detections(log_path)
    symbols = sorted({d.symbol for d in detections})
    if not symbols:
        console.print("No detections logged yet.")
        return
    price_frames, _ = fetch_price_history(symbols)
    price_frames.pop("^NSEI", None)
    closes = {s: f["Close"] for s, f in price_frames.items()}

    report = score_open_detections(log_path, closes)
    console.print(report.summary_line())
    for horizon, result in report.horizons.items():
        console.print(
            f"  {horizon}d: mean {result.detection_return_mean_pct:+.2f}%, "
            f"win rate {result.detection_win_rate:.0%}, "
            f"edge vs control {result.edge_vs_control_pp:+.2f}pp (n={result.matured_count})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nse_screener.workbench")
    parser.add_argument("--analyst", default=getpass.getuser())
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("start")
    p.add_argument("symbol")

    p = sub.add_parser("context")
    p.add_argument("symbol")
    p.add_argument("--file", required=True)

    p = sub.add_parser("accounting")
    p.add_argument("symbol")
    p.add_argument("--reasoning", default=None)

    p = sub.add_parser("financial")
    p.add_argument("symbol")
    p.add_argument("--risk-free-rate", type=float, required=True)
    p.add_argument("--equity-risk-premium", type=float, required=True)

    p = sub.add_parser("expectations")
    p.add_argument("symbol")
    p.add_argument("--risk-free-rate", type=float, required=True)
    p.add_argument("--equity-risk-premium", type=float, required=True)
    p.add_argument("--explicit-years", type=int, default=10)

    p = sub.add_parser("assess")
    p.add_argument("symbol")
    p.add_argument("--verdict", required=True, choices=["LOW", "FAIR", "HEROIC"])
    p.add_argument("--note", required=True)

    p = sub.add_parser("value")
    p.add_argument("symbol")
    p.add_argument("--scenarios", required=True)
    p.add_argument("--risk-free-rate", type=float, required=True)
    p.add_argument("--equity-risk-premium", type=float, required=True)

    p = sub.add_parser("label")
    p.add_argument("symbol")
    p.add_argument("--label", required=True, choices=["INVESTMENT", "SPECULATION"])
    p.add_argument("--justification", default=None)

    p = sub.add_parser("size")
    p.add_argument("symbol")
    p.add_argument("--capital", type=float, required=True)
    p.add_argument("--risk", type=float, required=True)
    p.add_argument("--atr-multiple", type=float, default=2.0)

    p = sub.add_parser("report")
    p.add_argument("symbol")

    p = sub.add_parser(
        "run",
        help="Run every computable stage in one shot; stops cleanly where a human "
        "judgment call is needed, then always prints the full report.",
    )
    p.add_argument("symbol")
    p.add_argument(
        "--context-file",
        default=None,
        help="Required only the first time, for stage 1 (business context).",
    )
    p.add_argument("--risk-free-rate", type=float, required=True)
    p.add_argument("--equity-risk-premium", type=float, required=True)
    p.add_argument("--explicit-years", type=int, default=10)
    p.add_argument(
        "--reasoning", default=None, help="Only needed if accounting reaches INVESTIGATE."
    )
    p.add_argument(
        "--scenarios",
        default=None,
        help="conservative=0.04,base=0.07,optimistic=0.10 -- if omitted, stops after expectations.",
    )

    p = sub.add_parser("experimental-scan")
    p.add_argument("--universe", required=True)
    p.add_argument("--log", default="data/experimental_detections.jsonl")

    p = sub.add_parser("validate-experimental")
    p.add_argument("--log", default="data/experimental_detections.jsonl")

    args = parser.parse_args(argv)
    console = Console()

    handlers = {
        "start": cmd_start,
        "context": cmd_context,
        "accounting": cmd_accounting,
        "financial": cmd_financial,
        "expectations": cmd_expectations,
        "assess": cmd_assess,
        "value": cmd_value,
        "label": cmd_label,
        "size": cmd_size,
        "report": cmd_report,
        "run": cmd_run,
        "experimental-scan": cmd_experimental_scan,
        "validate-experimental": cmd_validate_experimental,
    }

    try:
        handlers[args.command](args, console)
    except _DOMAIN_ERRORS as exc:
        console.print(f"[bold red]{type(exc).__name__}:[/bold red] {exc}")
        return 1
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
