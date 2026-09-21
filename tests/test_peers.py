from __future__ import annotations

from pathlib import Path

import pytest

from nse_screener.fundamentals.peers import (
    InsufficientPeersError,
    NotApplicableForSector,
    SectorInfo,
    UnmappedSymbolError,
    check_applicable,
    load_sector_map,
    peer_statistics,
    require_mapped,
    require_sufficient_peers,
)

_REAL_SECTOR_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "sector_map.csv"
_UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "universe.txt"


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "sector_map.csv"
    path.write_text("symbol,sector,industry\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_load_sector_map_reads_valid_csv(tmp_path: Path):
    path = _write_csv(tmp_path, ["AAA,Information Technology,IT Services"])
    result = load_sector_map(path)
    assert result["AAA"] == SectorInfo(
        symbol="AAA", sector="Information Technology", industry="IT Services"
    )


def test_load_sector_map_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_sector_map(tmp_path / "does_not_exist.csv")


def test_load_sector_map_raises_on_missing_column(tmp_path: Path):
    path = tmp_path / "sector_map.csv"
    path.write_text("symbol,sector\nAAA,Energy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        load_sector_map(path)


def test_load_sector_map_raises_on_empty_field(tmp_path: Path):
    path = _write_csv(tmp_path, ["AAA,,IT Services"])
    with pytest.raises(ValueError, match="must not be empty"):
        load_sector_map(path)


def test_load_sector_map_raises_on_duplicate_symbol(tmp_path: Path):
    path = _write_csv(
        tmp_path,
        ["AAA,Energy,Oil Gas", "AAA,Financial Services,Private Sector Bank"],
    )
    with pytest.raises(ValueError, match="duplicate symbol"):
        load_sector_map(path)


def test_real_sector_map_covers_the_curated_universe():
    sector_map = load_sector_map(_REAL_SECTOR_MAP_PATH)
    universe = [
        line.strip()
        for line in _UNIVERSE_PATH.read_text().splitlines()
        if line.strip()
    ]
    require_mapped(universe, sector_map)  # must not raise


def test_require_mapped_raises_unmapped_symbol_error():
    sector_map = {"AAA": SectorInfo(symbol="AAA", sector="Energy", industry="Oil Gas")}
    with pytest.raises(UnmappedSymbolError, match="BBB"):
        require_mapped(["AAA", "BBB"], sector_map)


def test_sector_info_requires_specialist_analysis():
    bank = SectorInfo(symbol="AAA", sector="Financial Services", industry="Private Sector Bank")
    industrial = SectorInfo(symbol="BBB", sector="Energy", industry="Oil Gas")
    assert bank.requires_specialist_analysis is True
    assert industrial.requires_specialist_analysis is False


def test_peer_statistics_arithmetic():
    sector_map = {
        f"P{i}": SectorInfo(symbol=f"P{i}", sector="TestSector", industry="TestIndustry")
        for i in range(1, 6)
    }
    metric_by_symbol = {"P1": 10.0, "P2": 20.0, "P3": 30.0, "P4": 40.0, "P5": 50.0}

    stats = peer_statistics(metric_by_symbol, sector_map, min_peers=5)

    assert stats["P3"].sector_median == pytest.approx(30.0)
    assert stats["P3"].sector_q1 == pytest.approx(20.0)
    assert stats["P3"].sector_q3 == pytest.approx(40.0)
    assert stats["P3"].percentile_within_sector == pytest.approx(0.6)  # 3 of 5 <= 30
    assert stats["P1"].percentile_within_sector == pytest.approx(0.2)
    assert stats["P5"].percentile_within_sector == pytest.approx(1.0)
    assert all(s.peer_count == 5 for s in stats.values())
    assert all(s.sufficient_peers is True for s in stats.values())


def test_peer_statistics_sufficient_peers_boundary():
    sector_map = {
        f"P{i}": SectorInfo(symbol=f"P{i}", sector="TestSector", industry="TestIndustry")
        for i in range(1, 5)  # only 4 peers
    }
    metric_by_symbol = {"P1": 10.0, "P2": 20.0, "P3": 30.0, "P4": 40.0}

    stats = peer_statistics(metric_by_symbol, sector_map, min_peers=5)

    assert all(s.sufficient_peers is False for s in stats.values())
    assert all(s.peer_count == 4 for s in stats.values())


def test_peer_statistics_raises_unmapped_symbol_error():
    sector_map = {"P1": SectorInfo(symbol="P1", sector="TestSector", industry="TestIndustry")}
    with pytest.raises(UnmappedSymbolError):
        peer_statistics({"P1": 10.0, "P2": 20.0}, sector_map, min_peers=1)


def test_require_sufficient_peers_raises_when_insufficient():
    sector_map = {
        "P1": SectorInfo(symbol="P1", sector="TestSector", industry="TestIndustry"),
    }
    stats = peer_statistics({"P1": 10.0}, sector_map, min_peers=5)
    with pytest.raises(InsufficientPeersError):
        require_sufficient_peers(stats["P1"], min_peers=5)


def test_require_sufficient_peers_passes_through_when_sufficient():
    sector_map = {
        f"P{i}": SectorInfo(symbol=f"P{i}", sector="TestSector", industry="TestIndustry")
        for i in range(1, 6)
    }
    metric_by_symbol = {f"P{i}": float(i) for i in range(1, 6)}
    stats = peer_statistics(metric_by_symbol, sector_map, min_peers=5)
    assert require_sufficient_peers(stats["P1"], min_peers=5) is stats["P1"]


def test_check_applicable_raises_for_financial_sector_metric():
    bank = SectorInfo(symbol="AAA", sector="Financial Services", industry="Private Sector Bank")
    with pytest.raises(NotApplicableForSector):
        check_applicable("AAA", "gross_profitability", bank)


def test_check_applicable_does_not_raise_for_non_financial_sector():
    industrial = SectorInfo(symbol="BBB", sector="Energy", industry="Oil Gas")
    check_applicable("BBB", "gross_profitability", industrial)  # must not raise


def test_check_applicable_does_not_raise_for_non_listed_metric():
    bank = SectorInfo(symbol="AAA", sector="Financial Services", industry="Private Sector Bank")
    check_applicable("AAA", "relative_strength", bank)  # must not raise
