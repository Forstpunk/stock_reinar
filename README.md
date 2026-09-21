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
```

Exit codes: `0` success, `1` data-integrity failure, `2` insufficient
qualifying names, `3` configuration error.

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
  factors/
    momentum.py   Relative strength
    quality.py    Gross profitability, ROE
    value.py      Earnings yield, book-to-price
    financial.py  Piotroski F-Score
    forensic.py   Accrual/cash-flow disqualifier screen
  ranking.py      Eligibility filtering, composite scoring, shortlist
  report.py       Terminal (rich) + JSON reporting
  cli.py          Entry point
tests/            Synthetic-fixture unit tests, no network calls
universe.txt      Example curated watchlist
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
