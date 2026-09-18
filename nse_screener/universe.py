"""Live discovery of the full NSE-listed equity universe.

An alternative to hand-maintaining ``universe.txt``: fetches NSE's own
published master list of listed equities and returns every symbol in the
"EQ" (mainboard, normal trading) series.

NSE's website sits behind Akamai bot mitigation and is known to block
requests from many data-center / cloud IP ranges outright, independent of
headers, session cookies, or request shape. This module does nothing to
evade that. If the fetch is blocked or fails for any reason, it raises
``NSEUniverseFetchError`` rather than silently falling back to a stale or
partial list -- if this keeps failing, run it from a normal residential
or office network instead of a cloud sandbox.
"""

from __future__ import annotations

import csv
import io

import requests

EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equity/EQUITY_L.csv"
_HOME_URL = "https://www.nseindia.com"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class NSEUniverseFetchError(Exception):
    """Raised when the live NSE equity list cannot be fetched or parsed."""


def fetch_full_nse_universe(timeout_seconds: int = 20) -> list[str]:
    """Fetch every currently NSE-listed, EQ-series equity as bare symbols.

    Visits the NSE homepage first to pick up the session cookies its CDN
    expects, then requests the published equity-list CSV in the same
    session -- the standard approach for reaching this endpoint. Raises
    ``NSEUniverseFetchError`` on any network failure, non-200 response, or
    a response that doesn't parse as the expected CSV.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)

    try:
        session.get(_HOME_URL, timeout=timeout_seconds)
        response = session.get(
            EQUITY_LIST_URL, timeout=timeout_seconds, headers={"Accept": "text/csv"}
        )
    except requests.RequestException as exc:
        raise NSEUniverseFetchError(
            f"could not reach NSE to fetch the equity list: {exc}"
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if response.status_code != 200 or "csv" not in content_type.lower():
        raise NSEUniverseFetchError(
            f"NSE equity list fetch failed (HTTP {response.status_code}, "
            f"content-type={content_type!r}). NSE commonly blocks requests "
            "from data-center/cloud IP ranges; if this keeps failing, run "
            "from a normal residential or office network connection."
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
