# NSE Equity Screener

A research CLI that screens NSE-listed Indian equities and outputs a ranked
shortlist of candidates, using factor techniques with documented empirical
support. It is a **screening tool, not a trading signal generator**: it never
predicts price, direction, or timing, and it never suggests an entry point.

## Install

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"      # Windows
# source .venv/bin/activate && pip install -e ".[dev]"   # macOS/Linux
```

Dependencies and tool configuration live in `pyproject.toml`. The `dev`
extra adds `pytest`, `ruff` and `mypy`.

Requires Python 3.11+.

## Usage

```bash
# curated watchlist (one NSE symbol per line, no .NS suffix)
.venv\Scripts\python.exe -m nse_screener --universe universe.txt --top 5

# also write a machine-readable copy of the same report
.venv\Scripts\python.exe -m nse_screener --universe universe.txt --top 5 --json report.json

# momentum + market-gate only, skips fundamentals entirely (much faster,
# but the weakest-evidenced subset of this method -- see Constraints)
.venv\Scripts\python.exe -m nse_screener --universe universe.txt --top 5 --skip-fundamentals

# fetch the full live NSE-listed (EQ-series) universe instead of a
# curated list -- see Known limitations for why this is best-effort
.venv\Scripts\python.exe -m nse_screener --full-market --top 5

# log this run's shortlist for outcome tracking (see below)
.venv\Scripts\python.exe -m nse_screener --universe universe.txt --top 5 --track tracker.jsonl
```

Exit codes: `0` success, `1` data-integrity failure, `2` insufficient
qualifying names, `3` configuration error.

### Outcome tracking (`tracker.py`)

Every backtest claim in this README came from the original design brief,
not from independently verifying NSE data. `--track PATH` appends each
run's shortlist (symbol, rank, entry price) to a log; once a run turns
63 sessions old, evaluate it against real forward returns versus the
Nifty over the identical window:

```bash
.venv\Scripts\python.exe -m nse_screener.tracker --log tracker.jsonl
```

This is the empirical, falsifiable check on whether the method actually
works -- not a simulation, only runs that have genuinely aged past the
holding horizon are scored.

## What each factor measures, and its evidence basis

| Factor | What it measures | Evidence |
|---|---|---|
| **Relative strength (momentum)** | Quarterly-weighted price momentum (most recent quarter double-weighted) | Documented across the US, Europe, Japan and India since Jegadeesh & Titman (1993); own NSE measurement backing this tool found the top RS decile forward-returning +2.40% vs +1.44% for the mid-decile over 20 sessions, monotonic across deciles, n=457,160 |
| **Value (earnings yield / book-to-price)** | Net income ÷ market cap, and total equity ÷ market cap -- is the stock cheap relative to its own fundamentals | Basu (1977) P/E anomaly; Fama & French (1992, 1993) book-to-market -- one of the best-replicated factors in equity research, on par with momentum and profitability. No NSE-specific validation for this tool, and value is documented to be *negatively correlated* with momentum, which is why it is not weighted above rs or f_score here |
| **Gross profitability** | Gross profit ÷ total assets | Novy-Marx: predicts the cross-section about as strongly as book-to-market; most profitable quintile beat least profitable by ~4%/year (1963-2010), replicated across 19 developed markets |
| **Return on equity** | Net income ÷ average equity | ~5.1%/yr long-high-ROE/short-low-ROE spread reported in emerging markets; the weakest-weighted of the five here since it overlaps heavily with gross profitability rather than adding an independent signal |
| **Piotroski F-Score** | 9 binary signals across profitability, leverage/liquidity, and efficiency trends | Indian evidence: a one-point F-Score improvement associates with ~4.93% higher one-year market-adjusted return; a Nifty-100 study (2007-2024) found high-F-Score firms delivered stronger returns and cushioned drawdowns |
| **Forensic accrual screen** | CFO/NI ratio, accrual ratio, receivables/inventory growth vs. revenue/COGS growth, leverage jump -- a *disqualifier*, not a score | Accruals are less persistent than cash flow and the market tends to overreact to accrual-heavy earnings; the anomaly is reported to have weakened after ~2002, so this is a flag for investigation, not proof of anything |
| **Market gate** | Index vs. 50/200-day moving averages, 200DMA slope, universe breadth | Identical setups returned +0.32pp edge in healthy regimes vs. -0.86pp in hostile ones in the source backtests; Siegel's long-run 200-day SMA work supports trend gating as a drawdown-reduction tool |

Composite score: `0.35*RS + 0.20*F-Score + 0.20*Value + 0.15*GrossProfitability
+ 0.10*ROE`, all percentile-ranked within the eligible universe. The weights
reflect each factor's evidence strength (momentum strongest, hence highest
weight) — they are **not** an optimised blend, and should not be tuned
against historical returns without a proper out-of-sample split. The value
factor was added after the original four; adding it necessarily rebalanced
all five weights, done explicitly here (not silently) precisely because
this composite is unvalidated -- see the comment on the `weight_*` fields in
`config.py` for the full reasoning, including why value sits below rs and
f_score despite comparable underlying evidence strength.

### Point-in-time factor backtests (`nse_screener/factor_backtest.py`)

The table above cites external, mostly non-NSE literature. This module
answers the same question with this project's own real NSE history
instead, at two different rigor levels that must never be blurred
together:

- **Momentum** is pure price data with no restatement risk, so it gets a
  genuine walk-forward backtest: price history is truncated at each
  historical as-of date, the live screener's own
  `weighted_relative_strength` computes the score from that truncated
  view only, and the forward return is looked up from data that did not
  exist yet at computation time. Run with `period="max"` price history
  (back to 1996 for large caps) to maximise the as-of date count.
- **F-Score, gross profitability, ROE and value** cannot be backtested
  this way: yfinance keeps no point-in-time fundamentals archive, only
  ~4 stored annual filings per symbol. Instead, each symbol's ~3
  successive real annual filings are used as their own (current, prior)
  pairs, dated `REPORTING_LAG_DAYS` (75 days) after fiscal year-end as a
  conservative stand-in for when that filing actually became public.
  This is a structural ceiling on sample size that cannot be raised
  without a vintage fundamentals data source — unlike momentum's as-of
  range, which is just a data-fetch choice.

**Reading the output — three numbers matter more than the headline
statistic:**

- `effective_n`, not `n_observations`. Forward-return windows from
  as-of dates closer together than the horizon overlap almost
  completely and are not independent draws; `n_observations` counts
  every row, `effective_n` estimates how many of those rows are
  actually independent once overlap is accounted for.
  `overlap_warning: true` means `effective_n` is below 30 — treat the
  result as under-powered, not as a finding.
- The 95% confidence interval (`rank_ic_ci_95`, `q5_minus_q1_ci_95`),
  bootstrapped by resampling whole as-of dates (never individual rows —
  rows sharing an as-of date share a market regime and are not
  independent). **A wide interval, or one that spans zero
  (`spans_zero: true`), means "unknown", not "zero".** Do not read an
  interval that includes zero as proof a factor has no edge; it means
  this sample cannot yet tell the difference between an edge and noise.
- `spearman_quintile_monotonicity` and `largest_single_step_share`. A
  large Q5-Q1 spread can be driven by a single quintile-boundary
  outlier rather than genuine ordering across the whole range;
  `largest_single_step_share` near 1.0 is the signal to distrust a
  large spread as broad-based evidence.

Run it with `python -m nse_screener.factor_backtest [--universe PATH]
[--diagnose-roe]`. The dated snapshot it writes to `data/` is a
baseline: re-running periodically and diffing against it is how you
check whether a factor's edge is holding up or eroding, not a one-time
verdict.

## Constraints this tool enforces, and why

- **Minimum holding horizon: 63 sessions (~3 months), printed every run,
  with no shorter-horizon mode.** The relative-strength edge accrues at
  roughly 1pp/month gross, while round-trip costs run ~0.40% fixed; below
  ~10 sessions the net edge turns negative.
- **No breakout-day entry logic.** A tested trend-template + RS≥80 +
  tight-base + volume-confirmed breakout specification (1,253 NSE trades,
  1997-2026) produced a -0.01pp edge versus matched random entry — no
  better than chance.
- **No fixed-percentage stops.** An 8% fixed stop caused 31.8% stop-outs
  and cost ~0.5pp on identical signals in the source backtests.
- **No Fibonacci/harmonic patterns.** Worst-ranked technique in the source
  data (bearish AB=CD: 26.3% failure rate).
- **The market gate always gates the output label.** When the gate is
  HOSTILE, the tool labels its output a watchlist with reduced/zero
  exposure guidance, never a buy list.

## Known limitations

- **Survivorship bias.** The universe is currently-listed names only;
  results are biased upward because they omit companies that delisted or
  went to zero.
- **Financial-sector companies are excluded from the fundamentals path.**
  Banks and NBFCs (HDFCBANK, SBIN, BAJFINANCE, ...) publish an unclassified
  balance sheet with no current/non-current split, no gross profit and no
  inventory. The Piotroski F-Score, gross profitability and the forensic
  inventory/receivables checks are not defined for those statements --
  Piotroski's own study excludes financials -- so they are excluded with an
  explicit "unclassified balance sheet" reason rather than reported as
  missing data. No alternative data source can supply lines that do not
  exist; a bank-specific quality model would be a separate piece of work.
  They still rank normally under `--skip-fundamentals`.
- **How statement gaps are read.** An `Inventory` line that Yahoo omits
  entirely from an otherwise classified balance sheet is read as zero (a
  services company such as INFY has none); a line that is present but
  blank is still treated as missing and the symbol is excluded. Yahoo also
  lists the newest fiscal year before the annual report is filed, as a
  column with no revenue or net income; that placeholder is skipped and
  the last complete year is used, with a "latest complete fundamentals
  period ends ..." notice. The two periods used must be 330-400 days apart:
  a company that changed its fiscal year (NESTLEIND, Dec -> Mar) is
  excluded until it has two comparable annual periods, because
  year-on-year signals across a transition period are not meaningful.
- **Data source reliability.** `yfinance` is an unofficial scrape of
  Yahoo Finance, not an official exchange feed, and has been observed to
  silently drop trading sessions for some symbols on some days. All price
  data is validated (session count, calendar alignment, recency,
  non-positive prices, and anomalous single-session moves) before any
  factor touches it, and validation results are always printed — never
  silently dropped.
- **Fundamentals coverage is uneven, especially for banks/NBFCs.**
  Financial-sector statements don't carry line items like "gross profit"
  or "current assets/liabilities" the way industrial companies do, so
  those symbols are correctly excluded from fundamentals-based scoring
  (fail-fast rather than imputed) rather than silently mis-scored.
- **Unvalidated composite weights**, as above — evidence-strength-ordered,
  not backtested as a blend.
- **US/developed-market-derived quality research** (gross profitability,
  ROE spreads) is applied to India without local replication.
- **`--full-market` depends on NSE's own site being reachable**, which
  sits behind bot mitigation that blocks many data-center/cloud IP
  ranges outright. It works from an ordinary residential/office
  connection; it may not work from a hosted CI/sandbox environment.

## Stage 2: deep analysis workbench

Stage 1 (above) is a fully automated, systematic screen. Stage 2 is a
**per-company deep-dive workbench** that is deliberately *not* fully
automated: it enforces Palepu's four-step analysis sequence as a state
machine, and three steps carry irreducible human judgment that this
tool computes what it can for and then stops -- it never fabricates the
judgment itself. Stage 1 never depends on Stage 2; you can run the
workbench on any symbol, not just ones that cleared the Stage 1
shortlist.

```bash
# the fast path: run everything computable in one command, resumable
# across days, always ends with the full report
.venv\Scripts\python.exe -m nse_screener.workbench run RELIANCE \
    --context-file context_reliance.md \
    --risk-free-rate 0.07 --equity-risk-premium 0.05 \
    --scenarios conservative=0.04,base=0.07,optimistic=0.10

# or step through it by hand:
.venv\Scripts\python.exe -m nse_screener.workbench start RELIANCE
.venv\Scripts\python.exe -m nse_screener.workbench context RELIANCE --file context_reliance.md
.venv\Scripts\python.exe -m nse_screener.workbench accounting RELIANCE --reasoning "..."
.venv\Scripts\python.exe -m nse_screener.workbench financial RELIANCE --risk-free-rate 0.07 --equity-risk-premium 0.05
.venv\Scripts\python.exe -m nse_screener.workbench expectations RELIANCE --risk-free-rate 0.07 --equity-risk-premium 0.05
.venv\Scripts\python.exe -m nse_screener.workbench assess RELIANCE --verdict FAIR --note "..."       # HUMAN
.venv\Scripts\python.exe -m nse_screener.workbench value RELIANCE --scenarios conservative=0.04,base=0.07,optimistic=0.10 --risk-free-rate 0.07 --equity-risk-premium 0.05
.venv\Scripts\python.exe -m nse_screener.workbench label RELIANCE --label INVESTMENT --justification "..."   # HUMAN
.venv\Scripts\python.exe -m nse_screener.workbench size RELIANCE --capital 500000 --risk 0.01
.venv\Scripts\python.exe -m nse_screener.workbench report RELIANCE
```

`--context-file` points at a plain-text file with three sections the
analyst writes (never generated), each at least 100 characters:
```
INDUSTRY ECONOMICS:
...

COMPETITIVE POSITION:
...

REVENUE DRIVERS:
- ...
- ...
```

Only symbols listed in `data/sector_map.csv` (500 by default, the Nifty
500 constituents as published by NSE) can run through
`accounting`/`financial`/`size`, since peer benchmarking and the
financial-sector exclusion need a sector classification. Add a row
there for any other symbol.

The 25 symbols originally hand-curated for this file (matching
`universe.txt`) have a finer-grained `industry` value (e.g. `Private
Sector Bank`, `Passenger Vehicles`). The other ~475, bulk-loaded from
NSE's `ind_nifty500list.csv`, only carry one classification level, so
`sector` and `industry` are identical for those rows -- peer
benchmarking for them compares within the broader sector (e.g. all of
`Financial Services`), not a narrower peer group. `sector` alone drives
the financial-sector exclusion (`SPECIALIST_SECTORS` in `peers.py`), so
that check is unaffected either way.

### The Palepu sequence (`nse_screener/workbench/session.py`)

A strict state machine -- calling a stage's function before its
predecessor is complete raises `StageOutOfOrderError`, with no bypass:

1. **Business strategy.** Human-written industry/competitive-position
   text and revenue drivers. Cannot be computed. If never supplied, the
   session's valid terminal state is `INCOMPLETE`, not a guess.
2. **Accounting quality.** Combines three independent methods without
   double-counting: the v1 forensic screen, the Beneish M-Score, and the
   Altman Z-Score. Two of the three agreeing is what disqualifies a
   company outright (no price is ever attached to a disqualified
   company); any one firing alone requires a human reasoning note and
   yields `INVESTIGATE`, not an automatic pass or fail.
3. **Financial analysis.** ROIC (McKinsey-style: reorganized into
   NOPAT/invested capital first), compared against a real computed WACC
   band, plus valuation multiples.
4. **Prospective.** A two-stage reverse-DCF solves for the growth rate
   the current price already implies (never a forward forecast), a
   Graham-style margin-of-safety value *range*, and ATR-based position
   sizing. The reverse-DCF's plausibility verdict and the value range's
   Graham label are both human-only fields -- `ANALYSIS_COMPLETE`
   (renamed from "QUALIFIED" to avoid reading as a disguised buy signal)
   is withheld until both are filled in.

Stage 4 is freely re-runnable and always refreshes against today's
price, so `expectations`/`value` (and therefore `run`) decide whether a
judgment you already recorded still applies to the new numbers. A
verdict or label is **carried forward only while it is an answer to the
same question**, and the run says which happened either way:

| Judgment | Carried forward while | Dropped when |
|---|---|---|
| `plausibility_verdict` | same `--explicit-years`, and implied growth at the WACC point moved ≤ 1.0pp | the horizon changed, or growth moved further -- LOW/FAIR/HEROIC is then a verdict on a different number |
| `graham_label` | same scenario names, same `price_position`, and every per-share value and the price moved ≤ 2% | any of those changed -- the label asserts safety of principal *at a price*, and that margin has moved |

Both tolerances live in `workbench/session.py`
(`PLAUSIBILITY_CARRY_FORWARD_TOLERANCE_PP`,
`GRAHAM_LABEL_CARRY_FORWARD_TOLERANCE`) with the reasoning for each
value. A dropped judgment reappears in `missing_requirements`, so the
session drops back out of `ANALYSIS_COMPLETE` until you re-`assess` or
re-`label` it -- it is never silently kept and never silently discarded.

### The v3 fundamentals engine (`nse_screener/fundamentals/`)

Built to make step 3/4 rigorous rather than approximate:

| Module | What it does | Evidence basis |
|---|---|---|
| `peers.py` | Sector/peer benchmarking from a maintained `sector_map.csv`; flags financials as needing a specialist toolkit this system doesn't provide | Tracy: ratios are only meaningful compared within industry |
| `reorganize.py` | Separates operating from non-operating items into NOPAT and invested capital | McKinsey's key-value-driver framework |
| `wacc.py` | Real beta (2y weekly returns, Blume-adjusted) plus a sensitivity band, never a single point estimate | Beta is unstable and ERP is contested; propagate the band, not the point |
| `roic.py` | ROIC decomposed into margin × turnover, economic spread vs. the WACC band | Value comes from growth *at* a return above cost of capital, not growth alone |
| `distress.py` | Altman Z-Score + a liquidity/coverage family (interest coverage, net debt/EBITDA, current/acid-test/cash ratios) | A documented Beneish M-Score failure (Toshiba) was caught by Altman Z -- two independent methods, not one |
| `multiples.py` | P/E, P/B, EV/EBITDA, EV/EBIT, EV/Sales, EV/invested capital, FCF yield, and total shareholder payout yield (dividends + buybacks) | Total payout closes the gap in dividend-only reasoning now that buybacks rival dividends as a payout channel |
| `live_data.py` | Fetches everything above from yfinance in one pass, with documented approximations (below) |  |

**Approximations `live_data.py` makes, because yfinance does not report
the underlying line item separately for any NSE company observed:**
`amortization_of_intangibles` is always 0 (EBITA collapses to EBIT
exactly); `net_other_operating_assets` is always 0 (understates invested
capital by whatever non-PPE operating assets a company carries);
`income_continuing_ops` falls back to net income when not reported
separately. It also does not do the fiscal-year-gap or
unreported-placeholder detection that `data.fetch_fundamentals` does --
it takes the two most recent common annual columns as-is.

### Quarantined: busted-pattern detection (`nse_screener/experimental/`)

Busted-pattern outperformance is well-evidenced elsewhere (Bulkowski,
Grimes) but **has never been tested on NSE data**, and a closely related
specification (O'Neil breakout) tested at zero edge on 1,253 NSE trades.
This module is therefore quarantined: it never contributes to the v1
composite score or shortlist, its output only ever appears under an
"EXPERIMENTAL -- UNVALIDATED ON NSE" heading, and it raises
`NotValidatedError` at runtime -- not just by convention -- if anything
outside `experimental/`, the test suite, or the workbench's
`experimental-scan` command calls into it.

```bash
.venv\Scripts\python.exe -m nse_screener.workbench experimental-scan --universe universe.txt
.venv\Scripts\python.exe -m nse_screener.workbench validate-experimental
```

It would take 30+ matured detections showing a real edge versus a
matched random control to release this from quarantine. With only 25
symbols in the default universe, detections and their controls are
drawn from heavily overlapping time windows -- not independent trials --
so an "edge" reported today should not be trusted regardless of its sign.

## Planned upgrade path

1. Replace `yfinance` price fetching with an NSE Bhavcopy adapter for an
   authoritative, official price source.
2. Replace `yfinance` fundamentals with a filings-based adapter (and a
   proper mapping for financial-sector statement formats).
3. Add delisted-company data to remove survivorship bias from backtests
   and universe construction.

## Project layout

```
nse_screener/
  config.py       ScreenerConfig -- every tunable constant, hard-locks the
                  63-session holding horizon
  data.py         Price/fundamentals acquisition + integrity validation
  universe.py     Live full-market (NSE EQ-series) symbol discovery
  gate.py         Market regime gate (HEALTHY / NEUTRAL / HOSTILE)
  tracker.py      Outcome tracker: logs shortlist picks, scores real
                  forward returns vs. the index once they mature
  factors/
    momentum.py   Relative strength
    quality.py    Gross profitability, ROE
    value.py      Earnings yield, book-to-price
    financial.py  Piotroski F-Score
    forensic.py   Accrual/cash-flow disqualifier screen
  ranking.py      Eligibility filtering, composite scoring, shortlist
  report.py       Terminal (rich) + JSON reporting
  cli.py          Stage 1 entry point (python -m nse_screener)

  fundamentals/   Stage 2's rigor engine -- see "The v3 fundamentals engine"
    peers.py roic.py reorganize.py wacc.py distress.py multiples.py live_data.py

  workbench/      Stage 2 entry point (python -m nse_screener.workbench)
    session.py       Palepu state machine
    mscore.py        Beneish M-Score + combined accounting verdict
    expectations.py  Two-stage reverse-DCF
    valuation.py     Graham margin-of-safety value range
    sizing.py        ATR-based position sizing, expectancy
    __main__.py       CLI commands, incl. the one-shot `run`

  experimental/   Quarantined -- see "Quarantined: busted-pattern detection"
    busted.py validation_log.py

tests/            Synthetic-fixture unit tests, no network calls
universe.txt      Example curated watchlist
data/sector_map.csv  Sector/industry map for peer benchmarking (500 symbols, Nifty 500)
sessions/         Persisted workbench sessions, one JSON per symbol/date
```

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy nse_screener
```

The same three commands run in CI (`.github/workflows/ci.yml`) on Python
3.11 and 3.12 for every push and pull request.

## Known issues fixed in this revision

- **Re-running stage 4 silently erased the analyst's own judgments.**
  `expectations` rebuilt `implied_expectations` from scratch and `value`
  replaced `value_range` wholesale, so a recorded `plausibility_verdict`
  or `graham_label` was discarded with no message -- and because `run`
  calls both, the ordinary `assess` → `run` → `label` → `run` workflow
  wiped the very judgments the workbench exists to collect. A session
  that had reached `ANALYSIS_COMPLETE` quietly fell back to incomplete.
  Judgments are now carried forward when the numbers they were made
  about have not materially moved, and explicitly dropped *with a
  printed reason* when they have -- see the table under "The Palepu
  sequence". **If you ran earlier revisions, check any session whose
  `terminal_state` is not `ANALYSIS_COMPLETE` for a verdict or label you
  thought you had already recorded.**
- **Forensic screen was disqualifying healthy asset-light companies.** A
  ratio that was *undefined* (zero denominator: no debt, no inventory, zero
  net income) was being counted as a forensic flag. A debt-free,
  inventory-free business -- most IT-services and other asset-light names --
  therefore collected two "flags" for having no debt and no inventory and
  was excluded from every shortlist, even with cash flow above earnings.
  Undefined checks are now recorded separately in
  `ForensicResult.undefined_checks` for transparency and never count toward
  disqualification; only genuine threshold breaches do. **If you ran earlier
  revisions, debt-free and inventory-free companies were missing from your
  results.**
- **One unusable symbol aborted the whole run.** A symbol whose momentum
  could not be computed (enough rows, but too few valid closes) raised
  through `build_shortlist` and killed the run before any other symbol was
  scored. Momentum is now computed per symbol; an uncomputable one is counted
  under `excluded_count["momentum_uncomputable"]` and the run continues.
- **Off-by-one in the momentum session requirement.** The relative-strength
  formula needs `4 * quarter_sessions + 1` sessions (253 by default), but the
  length check accepted 252 and the failure surfaced from a later NaN guard
  with a misleading message. The requirement is now derived from
  `momentum_quarter_sessions`, and `ScreenerConfig` rejects an
  `eligibility_min_sessions` lower than `min_sessions` (both default to 300).
- **`composite_score` meant different things on the two ranking paths.**
  Every `ShortlistEntry` now carries `score_basis` (`"composite"` or
  `"rs_only"`), shown in the report, so an `--skip-fundamentals` score can
  never be mistaken for a full composite.
