"""Sector classification and peer benchmarking.

Tracy's central doctrine: ratios must be read as trends and compared
within industry -- profit margins, P/E levels, inventory turnover and
asset intensity differ structurally across sectors. Comparing a bank's
ROA to a software firm's is an error, not an insight. This module is the
prerequisite for every sector-relative metric elsewhere in
``fundamentals/`` and for the financial-sector exclusion those metrics
need.

Sector data comes from a maintained local file (``data/sector_map.csv``),
never from a live API -- vendor sector classifications are inconsistent
and unversioned, and a wrong classification silently benchmarks a company
against the wrong peers, which is worse than refusing to benchmark it at
all. An unmapped symbol is reported, never guessed at or defaulted to
"Unknown".
"""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel, ConfigDict

#: Sector labels (as used in the ``sector`` column of sector_map.csv)
#: whose economics -- deposits as liabilities, credit costs, provisioning,
#: capital adequacy -- make industrial/services financial ratios
#: undefined. Matches NSE's own "Financial Services" sectoral grouping.
SPECIALIST_SECTORS: frozenset[str] = frozenset({"Financial Services"})

#: Metrics not defined for symbols in SPECIALIST_SECTORS, per the v3 build
#: spec section 1. Exposed so callers can check membership without
#: duplicating this list; ``check_applicable`` is the enforcement point.
NOT_APPLICABLE_FOR_FINANCIALS: frozenset[str] = frozenset(
    {
        "gross_profitability",
        "inventory_turnover",
        "current_ratio",
        "asset_turnover",
        "altman_z_score",
        "ev_based_multiples",
    }
)


class SectorInfo(BaseModel):
    """One symbol's sector/industry classification."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    sector: str
    industry: str

    @property
    def requires_specialist_analysis(self) -> bool:
        """True for financials (banks/NBFCs/insurers) -- see SPECIALIST_SECTORS."""
        return self.sector in SPECIALIST_SECTORS


class UnmappedSymbolError(Exception):
    """Raised when one or more symbols are absent from the sector map."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = sorted(symbols)
        super().__init__(
            f"{len(self.symbols)} symbol(s) absent from the sector map, "
            f"cannot be benchmarked: {', '.join(self.symbols)}"
        )


class InsufficientPeersError(Exception):
    """Raised by callers that require a sufficiently-benchmarked stat.

    ``peer_statistics`` itself never raises this -- it always returns a
    ``PeerStats`` for every requested symbol, with ``sufficient_peers``
    set honestly. This is for a caller that wants to treat an
    under-populated sector as a hard stop rather than a soft flag.
    """

    def __init__(self, sector: str, peer_count: int, min_peers: int) -> None:
        self.sector = sector
        self.peer_count = peer_count
        self.min_peers = min_peers
        super().__init__(
            f"sector {sector!r} has only {peer_count} peer(s) in this universe, "
            f"need at least {min_peers} to benchmark"
        )


class NotApplicableForSector(Exception):
    """Raised for a metric that is not defined for the subject's sector."""

    def __init__(self, symbol: str, metric: str, sector: str) -> None:
        self.symbol = symbol
        self.metric = metric
        self.sector = sector
        super().__init__(f"{symbol}: {metric!r} is not defined for sector {sector!r}")


class PeerStats(BaseModel):
    """Where one symbol's metric sits relative to its sector peers."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    sector: str
    metric_value: float
    sector_median: float
    sector_q1: float
    sector_q3: float
    peer_count: int
    percentile_within_sector: float
    sufficient_peers: bool


def load_sector_map(path: Path) -> dict[str, SectorInfo]:
    """Load ``symbol,sector,industry`` rows into a lookup by symbol.

    Raises ``FileNotFoundError`` if ``path`` doesn't exist, or
    ``ValueError`` if the header is missing a required column, a row has
    an empty symbol/sector/industry, or a symbol is duplicated. Never
    defaults a missing sector/industry to "Unknown" -- an incomplete row
    is a data problem to fix in the file, not something to paper over.
    """
    if not path.exists():
        raise FileNotFoundError(f"sector map not found: {path}")

    entries: dict[str, SectorInfo] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: sector map has no header row")
        missing_columns = {"symbol", "sector", "industry"} - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"{path}: missing required column(s): {sorted(missing_columns)}")

        for row_number, row in enumerate(reader, start=2):
            symbol = (row["symbol"] or "").strip()
            sector = (row["sector"] or "").strip()
            industry = (row["industry"] or "").strip()
            if not symbol or not sector or not industry:
                raise ValueError(
                    f"{path}:{row_number}: symbol/sector/industry must not be empty"
                )
            if symbol in entries:
                raise ValueError(f"{path}: duplicate symbol {symbol!r}")
            entries[symbol] = SectorInfo(symbol=symbol, sector=sector, industry=industry)

    return entries


def require_mapped(symbols: list[str], sector_map: dict[str, SectorInfo]) -> None:
    """Raise ``UnmappedSymbolError`` naming every symbol absent from ``sector_map``."""
    unmapped = [symbol for symbol in symbols if symbol not in sector_map]
    if unmapped:
        raise UnmappedSymbolError(unmapped)


def peer_statistics(
    metric_by_symbol: dict[str, float],
    sector_map: dict[str, SectorInfo],
    min_peers: int = 5,
) -> dict[str, PeerStats]:
    """Benchmark each symbol's metric against its sector peers within ``metric_by_symbol``.

    Every symbol in ``metric_by_symbol`` must be present in ``sector_map``
    -- raises ``UnmappedSymbolError`` (via ``require_mapped``) otherwise.
    A sector with fewer than ``min_peers`` members *within this specific
    set of symbols* still gets a ``PeerStats`` built from whatever peers
    it has, but with ``sufficient_peers=False`` -- it is never silently
    compared against the whole market instead of its own sector, and the
    caller decides whether to act on a stat flagged unbenchmarkable.
    """
    require_mapped(list(metric_by_symbol.keys()), sector_map)

    by_sector: dict[str, list[str]] = {}
    for symbol in metric_by_symbol:
        by_sector.setdefault(sector_map[symbol].sector, []).append(symbol)

    results: dict[str, PeerStats] = {}
    for sector, symbols in by_sector.items():
        values = sorted(metric_by_symbol[symbol] for symbol in symbols)
        peer_count = len(values)
        median = _percentile(values, 50)
        q1 = _percentile(values, 25)
        q3 = _percentile(values, 75)
        sufficient = peer_count >= min_peers

        for symbol in symbols:
            value = metric_by_symbol[symbol]
            rank = sum(1 for v in values if v <= value)
            results[symbol] = PeerStats(
                symbol=symbol,
                sector=sector,
                metric_value=value,
                sector_median=median,
                sector_q1=q1,
                sector_q3=q3,
                peer_count=peer_count,
                percentile_within_sector=rank / peer_count,
                sufficient_peers=sufficient,
            )

    return results


def require_sufficient_peers(stats: PeerStats, min_peers: int) -> PeerStats:
    """Raise ``InsufficientPeersError`` if ``stats`` was flagged unbenchmarkable."""
    if not stats.sufficient_peers:
        raise InsufficientPeersError(stats.sector, stats.peer_count, min_peers)
    return stats


def check_applicable(symbol: str, metric: str, sector_info: SectorInfo) -> None:
    """Raise ``NotApplicableForSector`` if ``metric`` is not defined for ``sector_info``."""
    if sector_info.requires_specialist_analysis and metric in NOT_APPLICABLE_FOR_FINANCIALS:
        raise NotApplicableForSector(symbol, metric, sector_info.sector)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy's default 'linear' method)."""
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
