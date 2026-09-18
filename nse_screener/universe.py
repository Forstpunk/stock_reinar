"""Live discovery of the full NSE-listed equity universe.

An alternative to hand-maintaining ``universe.txt``: fetches NSE's own
published master list of listed equities and returns every symbol in the
"EQ" (mainboard, normal trading) series.

NSE publishes this list as a plain CSV under ``nsearchives.nseindia.com``;
a simple GET with a normal browser User-Agent is sufficient, no session
cookies or homepage visit required.
"""

from __future__ import annotations

import csv
import io

import requests

EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv",
}


class NSEUniverseFetchError(Exception):
    """Raised when the live NSE equity list cannot be fetched or parsed."""


def fetch_full_nse_universe(timeout_seconds: int = 20) -> list[str]:
    """Fetch every currently NSE-listed, EQ-series equity as bare symbols.

    Raises ``NSEUniverseFetchError`` on any network failure, non-200
    response, or a response that doesn't parse as the expected CSV.
    """
    try:
        response = requests.get(EQUITY_LIST_URL, headers=_HEADERS, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise NSEUniverseFetchError(
            f"could not reach NSE to fetch the equity list: {exc}"
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if response.status_code != 200 or "csv" not in content_type.lower():
        raise NSEUniverseFetchError(
            f"NSE equity list fetch failed (HTTP {response.status_code}, "
            f"content-type={content_type!r})"
        )

    reader = csv.DictReader(io.StringIO(response.text))
    if reader.fieldnames is None:
        raise NSEUniverseFetchError("NSE equity list response had no header row")
    reader.fieldnames = [name.strip() for name in reader.fieldnames]

    if "SYMBOL" not in reader.fieldnames or "SERIES" not in reader.fieldnames:
        raise NSEUniverseFetchError(
            f"NSE equity list response is missing expected columns: {reader.fieldnames}"
        )

    symbols: set[str] = set()
    for row in reader:
        if row.get("SERIES", "").strip() == "EQ":
            symbol = row.get("SYMBOL", "").strip()
            if symbol:
                symbols.add(symbol)

    if not symbols:
        raise NSEUniverseFetchError(
            "NSE equity list parsed successfully but contained zero EQ-series symbols"
        )

    return sorted(symbols)
