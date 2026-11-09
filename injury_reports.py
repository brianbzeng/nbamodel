"""Read the NBA's official injury report index and normalize its PDFs."""

import re
from datetime import datetime
from importlib import import_module
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import pandas as pd
import requests
from PyPDF2 import PdfReader
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cleaner import TEAM_MAP_BREF


REPORT_INDEX_URL = "https://official.nba.com/nba-injury-report-{start}-{end}-season"
REPORT_PDF_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
QUARTER_HOUR_REPORTS_START = datetime(2025, 12, 22, 9)
REPORT_LINK = re.compile(
    r"/referee/injury/Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{1,2})(?:_(\d{2}))?([AP]M)\.pdf",
    re.IGNORECASE,
)
REPORT_HEADER = re.compile(r"Injury\s+Report:\s+(\d{1,2}/\d{1,2}/\d{2})\s+(\d{1,2}:\d{2})\s+([AP]M)")
REPORT_FIELDS = {
    "Game Date": "game_date",
    "Game Time": "game_time",
    "Matchup": "matchup",
    "Team": "team",
    "Player Name": "player_name",
    "Current Status": "status",
    "Reason": "reason",
}
TABULA_LOCK = Lock()


class ReportUnavailable(LookupError):
    """The NBA's CDN does not serve a report for this publication time."""


def prepare_report_parser() -> None:
    """Initialize Tabula's JVM on the main thread before concurrent fetches."""
    import_module("nbainjuries")


def _get(url: str) -> requests.Response:
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)


def fetch_report_index(season: int) -> list[datetime]:
    """Return report times from the indexed PDF filenames, in Eastern time."""
    url = REPORT_INDEX_URL.format(start=season - 1, end=f"{season % 100:02d}")
    response = _get(url)
    response.raise_for_status()
    links = REPORT_LINK.findall(response.text)
    if not links:
        raise ValueError(f"No official injury report links found at {url}.")

    timestamps = set()
    for date, hour, minute, period in links:
        timestamp = datetime.strptime(f"{date} {hour}:{minute or '00'}{period}", "%Y-%m-%d %I:%M%p")
        ending_year = timestamp.year + int(timestamp.month >= 7)
        if ending_year == season:
            timestamps.add(timestamp)
    return sorted(timestamps)


def _parse_pdf(content: bytes, filename: str, timestamp: datetime) -> pd.DataFrame:
    parser_timestamp = timestamp
    if datetime(2025, 12, 19, 15, 30) < timestamp < datetime(2025, 12, 19, 16, 45):
        # The library's legacy URL formatter has a gap during this transition.
        parser_timestamp = timestamp.replace(minute=30 if timestamp.hour == 15 else 45)
    # Parsing a local copy lets us bound and retry the single network request.
    with TemporaryDirectory(prefix="nba-injury-") as directory:
        (Path(directory) / filename).write_bytes(content)
        # Tabula uses shared JVM state, so only its parsing calls run serially.
        with TABULA_LOCK:
            from nbainjuries import injury

            return injury.get_reportdata(parser_timestamp, local=True, localdir=directory, return_df=True)


def fetch_report(timestamp: datetime) -> pd.DataFrame:
    """Download one PDF and return its actual publication time in Eastern time."""
    suffix = "%I_%M%p" if timestamp >= QUARTER_HOUR_REPORTS_START else "%I%p"
    url = REPORT_PDF_URL.format(stamp=timestamp.strftime("%Y-%m-%d_" + suffix))
    response = _get(url)
    # The CDN also uses 403 for nonexistent PDF object keys.
    if response.status_code in (403, 404):
        raise ReportUnavailable(f"Official NBA report at {timestamp} ET returned HTTP {response.status_code}.")
    response.raise_for_status()
    if not response.content.startswith(b"%PDF-"):
        raise ValueError(f"The official NBA report at {timestamp} ET is not a PDF.")
    header = PdfReader(BytesIO(response.content)).pages[0].extract_text()
    match = REPORT_HEADER.search(header)
    if match is None:
        raise ValueError(f"No publication time found in the official NBA report at {timestamp} ET.")
    published_at = datetime.strptime(" ".join(match.groups()), "%m/%d/%y %I:%M %p")
    if published_at.date() != timestamp.date():
        raise ValueError(f"The NBA returned the report for {published_at} ET instead of {timestamp} ET.")

    try:
        frame = _parse_pdf(response.content, url.rsplit("/", 1)[-1], timestamp)
    except Exception as exc:
        raise RuntimeError(f"Could not parse the official NBA report at {published_at} ET.") from exc
    if frame.empty or not set(REPORT_FIELDS).issubset(frame.columns):
        raise ValueError(f"Could not read the official NBA report at {timestamp} ET.")
    report = frame[list(REPORT_FIELDS)].rename(columns=REPORT_FIELDS)
    report["game_date"] = pd.to_datetime(report["game_date"], format="%m/%d/%Y")
    for column in ("game_time", "matchup", "team", "player_name", "status", "reason"):
        report[column] = report[column].astype("string").fillna("").str.strip()
    team_codes = report["team"].map(TEAM_MAP_BREF)
    if team_codes.isna().any():
        unknown_teams = ", ".join(report.loc[team_codes.isna(), "team"].unique())
        raise ValueError(f"Unrecognized teams in the official NBA report at {timestamp} ET: {unknown_teams}.")
    report["team"] = team_codes
    report["report_datetime"] = pd.Timestamp(published_at)
    return report
