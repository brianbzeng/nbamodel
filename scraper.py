"""Blazingly fast Basketball-Reference NBA game scraper.

The old implementation fetched and parsed pages sequentially with a fixed
``sleep`` after each request. This version replaces the serial pipeline with:

* ``httpx.AsyncClient`` to fetch pages concurrently.
* ``asyncio.Semaphore`` to cap in-flight HTTP requests.
* A ``ThreadPoolExecutor`` to parse HTML concurrently without blocking the
  async event loop.
* ``lxml`` for faster BeautifulSoup parsing.

The public API is unchanged, so ``app.py`` and callers keep working without
modifications.
"""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

BASE_URL = "https://www.basketball-reference.com/leagues/"

# Cap concurrent HTTP requests. Basketball-Reference doesn't rate-limit here,
# but keeping this bounded avoids saturating local sockets.
MAX_CONCURRENT_REQUESTS = 50

# For monthly schedule pages like NBA_2025_games-october.html
MONTH_SLUGS = {
    10: "october",
    11: "november",
    12: "december",
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
}


def _monthly_schedule_url(season: int, month: int) -> str:
    slug = MONTH_SLUGS[month]
    return f"{BASE_URL}NBA_{season}_games-{slug}.html"


def _season_hub_url(season: int) -> str:
    return f"{BASE_URL}NBA_{season}_games.html"


def _parse_season_hub(html: str, season: int) -> list[int]:
    """Extract available month slugs from the season hub page HTML."""
    month_lookup = {slug: month for month, slug in MONTH_SLUGS.items()}
    discovered: set[int] = set()

    # A simple regex is faster than building a full BeautifulSoup tree for the
    # hub page, which only exists to give us the month links.
    for match in re.finditer(
        rf"NBA_{season}_games-([a-z]+)\.html", html, flags=re.IGNORECASE
    ):
        slug = match.group(1).casefold()
        month = month_lookup.get(slug)
        if month is not None:
            discovered.add(month)

    if discovered:
        return sorted(discovered, key=lambda month: (month < 10, month))

    # Fall back to the standard NBA season months.
    return [10, 11, 12, 1, 2, 3, 4]


def _parse_month_html(payload: tuple[int, int, str]) -> pd.DataFrame:
    """Parse a single monthly schedule page HTML into a DataFrame."""
    season, month, html = payload

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": "schedule"})
    if table is None:
        # Fallback to any sortable table if the ID differs.
        table = soup.find("table", class_="sortable")

    if table is None:
        return pd.DataFrame(
            columns=[
                "date",
                "season",
                "home_team",
                "away_team",
                "home_pts",
                "away_pts",
            ]
        )

    tbody = table.find("tbody")
    if tbody is None:
        return pd.DataFrame(
            columns=[
                "date",
                "season",
                "home_team",
                "away_team",
                "home_pts",
                "away_pts",
            ]
        )

    games = []
    for row in tbody.find_all("tr"):
        if row.get("class") and "thead" in row["class"]:
            continue

        th = row.find("th")
        cols = row.find_all("td")
        if th is None or len(cols) == 0:
            continue

        date_text = th.get_text(strip=True)
        if not date_text:
            continue

        date = pd.to_datetime(date_text)

        away_cell = row.find("td", {"data-stat": "visitor_team_name"})
        home_cell = row.find("td", {"data-stat": "home_team_name"})
        away_pts_cell = row.find("td", {"data-stat": "visitor_pts"})
        home_pts_cell = row.find("td", {"data-stat": "home_pts"})

        if not (away_cell and home_cell and away_pts_cell and home_pts_cell):
            continue

        away_team = away_cell.get_text(strip=True)
        home_team = home_cell.get_text(strip=True)
        away_pts = away_pts_cell.get_text(strip=True)
        home_pts = home_pts_cell.get_text(strip=True)

        # Skip unplayed/future games.
        if away_pts == "" or home_pts == "":
            continue

        games.append(
            {
                "date": date,
                "season": season,
                "home_team": home_team,
                "away_team": away_team,
                "home_pts": int(home_pts),
                "away_pts": int(away_pts),
            }
        )

    if not games:
        return pd.DataFrame(
            columns=[
                "date",
                "season",
                "home_team",
                "away_team",
                "home_pts",
                "away_pts",
            ]
        )

    return pd.DataFrame(games)


async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.text


async def _fetch_month(
    client: httpx.AsyncClient,
    season: int,
    month: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, int, str]:
    url = _monthly_schedule_url(season, month)
    async with semaphore:
        html = await _fetch_text(client, url)
    return season, month, html


async def _scrape_season_with_client(
    season: int,
    client: httpx.AsyncClient,
    executor: ThreadPoolExecutor,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame:
    """Scrape one season using a shared async client and thread executor."""
    print(f"Discovering month pages for season {season}")
    hub_html = await _fetch_text(client, _season_hub_url(season))
    months = _parse_season_hub(hub_html, season)

    month_tasks = [
        _fetch_month(client, season, month, semaphore) for month in months
    ]
    payloads = await asyncio.gather(*month_tasks, return_exceptions=True)

    valid_payloads: list[tuple[int, int, str]] = []
    for item in payloads:
        if isinstance(item, Exception):
            print(f"Skipping month fetch for season {season}: {item}")
            continue
        valid_payloads.append(item)

    if not valid_payloads:
        return pd.DataFrame(
            columns=[
                "date",
                "season",
                "home_team",
                "away_team",
                "home_pts",
                "away_pts",
            ]
        )

    # Parse HTML concurrently using thread workers.
    dfs = list(executor.map(_parse_month_html, valid_payloads))
    valid_dfs = [df for df in dfs if not df.empty]

    if not valid_dfs:
        return pd.DataFrame(
            columns=[
                "date",
                "season",
                "home_team",
                "away_team",
                "home_pts",
                "away_pts",
            ]
        )

    season_games = pd.concat(valid_dfs, ignore_index=True)
    return (
        season_games.drop_duplicates(
            subset=["date", "season", "home_team", "away_team", "home_pts", "away_pts"]
        )
        .sort_values(["date", "home_team", "away_team"])
        .reset_index(drop=True)
    )


async def _scrape_multiple_seasons_async(
    start_season: int, end_season: int
) -> pd.DataFrame:
    limits = httpx.Limits(max_connections=MAX_CONCURRENT_REQUESTS)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    with ThreadPoolExecutor(max_workers=16) as executor:
        async with httpx.AsyncClient(
            headers=REQUEST_HEADERS, timeout=30.0, limits=limits
        ) as client:
            tasks = [
                _scrape_season_with_client(season, client, executor, semaphore)
                for season in range(start_season, end_season + 1)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

    all_games: list[pd.DataFrame] = []
    for item in results:
        if isinstance(item, Exception):
            print(f"Skipping season due to error: {item}")
            continue
        if not item.empty:
            all_games.append(item)

    if not all_games:
        return pd.DataFrame(
            columns=[
                "date",
                "season",
                "home_team",
                "away_team",
                "home_pts",
                "away_pts",
            ]
        )

    return (
        pd.concat(all_games, ignore_index=True)
        .sort_values(["date", "season", "home_team", "away_team"])
        .reset_index(drop=True)
    )


async def _scrape_season_async(season: int) -> pd.DataFrame:
    limits = httpx.Limits(max_connections=MAX_CONCURRENT_REQUESTS)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    with ThreadPoolExecutor(max_workers=16) as executor:
        async with httpx.AsyncClient(
            headers=REQUEST_HEADERS, timeout=30.0, limits=limits
        ) as client:
            return await _scrape_season_with_client(
                season, client, executor, semaphore
            )


async def _scrape_month_async(season: int, month: int) -> pd.DataFrame:
    limits = httpx.Limits(max_connections=MAX_CONCURRENT_REQUESTS)
    async with httpx.AsyncClient(
        headers=REQUEST_HEADERS, timeout=30.0, limits=limits
    ) as client:
        season_, month_, html = await _fetch_month(
            client, season, month, asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        )
    return _parse_month_html((season_, month_, html))


# Public API -----------------------------------------------------------------


def scrape_bref_season_games(season: int, sleep: float = 0) -> pd.DataFrame:
    """Scrape a full season of NBA games month by month from Basketball-Reference.

    season: ending year (e.g., 2025 = 2024-25 season)
    sleep: no longer enforced; kept for API compatibility.
    """
    return asyncio.run(_scrape_season_async(season))


def scrape_multiple_seasons(
    start_season: int = 2016, end_season: int = 2025, sleep: float = 0
) -> pd.DataFrame:
    """Scrape multiple seasons of NBA games from Basketball-Reference.

    sleep: no longer enforced; kept for API compatibility.
    """
    return asyncio.run(_scrape_multiple_seasons_async(start_season, end_season))


def scrape_bref_month_games(
    season: int, month: int, sleep: float = 0
) -> pd.DataFrame:
    """Scrape all games for a given month of a given season.

    season: ending year (e.g., 2026 for the 2025-26 season)
    month: month as an integer (10=Oct, 11=Nov, 12=Dec, 1=Jan, 2=Feb, ...)
    sleep: no longer enforced; kept for API compatibility.
    """
    if month not in MONTH_SLUGS:
        raise ValueError(f"Month {month} not supported for NBA schedule scraping.")
    return asyncio.run(_scrape_month_async(season, month))


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    games = scrape_multiple_seasons(2016, 2025)
    raw_file = RAW_DIR / "bref_games_2016_2025.csv"
    games.to_csv(raw_file, index=False)
    print(f"Saved raw games to {raw_file}")
