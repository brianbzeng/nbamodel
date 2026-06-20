# scraper.py

from pathlib import Path
import re
import time
from bs4 import BeautifulSoup, Comment
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

BASE_URL = "https://www.basketball-reference.com/leagues/"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

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


def _fetch_soup(url: str) -> BeautifulSoup:
    res = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    res.raise_for_status()
    return BeautifulSoup(res.text, "html.parser")


def _collect_tables(soup: BeautifulSoup) -> list:
    tables = list(soup.find_all("table", class_="sortable"))

    # Basketball-Reference sometimes wraps tables in HTML comments.
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "table" not in comment:
            continue
        commented_soup = BeautifulSoup(comment, "html.parser")
        tables.extend(commented_soup.find_all("table", class_="sortable"))

    return tables


def _available_months_for_season(season: int) -> list[int]:
    url = f"{BASE_URL}NBA_{season}_games.html"
    print(f"Discovering month pages from {url}")

    soup = _fetch_soup(url)
    month_lookup = {slug: month for month, slug in MONTH_SLUGS.items()}
    discovered_months = []

    # Basketball-Reference exposes season schedule pages as month links on the season hub.
    for link in soup.find_all("a", href = True):
        href = link["href"]
        match = re.search(rf"NBA_{season}_games-([a-z]+)\.html", href)
        if not match:
            continue

        slug = match.group(1).casefold()
        month = month_lookup.get(slug)
        if month is not None:
            discovered_months.append(month)

    unique_months = sorted(set(discovered_months), key = lambda month: (month < 10, month))
    if unique_months:
        return unique_months

    # Fall back to the standard NBA season months if the hub page does not expose links.
    return [10, 11, 12, 1, 2, 3, 4]


def scrape_bref_season_games(season: int, sleep=3) -> pd.DataFrame:
    """
    Scrape a full season of NBA games month by month from Basketball-Reference.
    season: ending year (e.g., 2025 = 2024-25 season)
    """
    all_months = []
    months = _available_months_for_season(season)

    for month in months:
        try:
            month_games = scrape_bref_month_games(season, month, sleep = sleep)
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code == 404:
                print(f"Skipping unavailable month {month} for season {season}")
                continue
            raise

        if not month_games.empty:
            all_months.append(month_games)

    if not all_months:
        return pd.DataFrame(
            columns=["date", "season", "home_team", "away_team", "home_pts", "away_pts"]
        )

    season_games = pd.concat(all_months, ignore_index = True)

    # Drop any duplicated rows if Basketball-Reference exposes overlapping month links.
    season_games = season_games.drop_duplicates(
        subset = ["date", "season", "home_team", "away_team", "home_pts", "away_pts"]
    ).sort_values(["date", "home_team", "away_team"]).reset_index(drop = True)

    return season_games


def scrape_multiple_seasons(start_season=2016, end_season=2025, sleep=3) -> pd.DataFrame:
    all_games = []
    for season in range(start_season, end_season + 1):
        df = scrape_bref_season_games(season, sleep=sleep)
        if not df.empty:
            all_games.append(df)

    if not all_games:
        return pd.DataFrame(
            columns=["date", "season", "home_team", "away_team", "home_pts", "away_pts"]
        )

    return pd.concat(all_games, ignore_index = True).sort_values(
        ["date", "season", "home_team", "away_team"]
    ).reset_index(drop = True)


def scrape_bref_month_games(season: int, month: int, sleep=3) -> pd.DataFrame:
    """
    Scrape all games for a given month of a given season from Basketball-Reference.

    season: ending year (e.g., 2026 for the 2025-26 season)
    month: month as an integer (10=Oct, 11=Nov, 12=Dec, 1=Jan, 2=Feb, ...)

    Returns DataFrame with columns:
        ['date', 'season', 'home_team', 'away_team', 'home_pts', 'away_pts']
    """
    if month not in MONTH_SLUGS:
        raise ValueError(f"Month {month} not supported for NBA schedule scraping.")

    slug = MONTH_SLUGS[month]
    url = f"{BASE_URL}NBA_{season}_games-{slug}.html"
    print(f"Scraping monthly schedule: {url}")

    soup = _fetch_soup(url)
    games = []

    tables = _collect_tables(soup)
    for table in tables:
        tbody = table.find("tbody")
        if not tbody:
            continue

        for row in tbody.find_all("tr"):
            # Skip header separator rows
            if "class" in row.attrs and "thead" in row["class"]:
                continue

            th = row.find("th")
            cols = row.find_all("td")
            if th is None or len(cols) == 0:
                continue

            date_text = th.text.strip()
            if not date_text:
                continue

            date = pd.to_datetime(date_text)

            away_cell = row.find("td", {"data-stat": "visitor_team_name"})
            home_cell = row.find("td", {"data-stat": "home_team_name"})
            away_pts_cell = row.find("td", {"data-stat": "visitor_pts"})
            home_pts_cell = row.find("td", {"data-stat": "home_pts"})

            if not (away_cell and home_cell and away_pts_cell and home_pts_cell):
                continue

            away_team = away_cell.text.strip()
            home_team = home_cell.text.strip()
            away_pts = away_pts_cell.text.strip()
            home_pts = home_pts_cell.text.strip()

            # Skip unplayed/future games
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

    time.sleep(sleep)

    if not games:
        return pd.DataFrame(
            columns=["date", "season", "home_team", "away_team", "home_pts", "away_pts"]
        )

    return pd.DataFrame(games)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    games = scrape_multiple_seasons(2016, 2025)
    raw_file = RAW_DIR / "bref_games_2016_2025.csv"
    games.to_csv(raw_file, index=False)
    print(f"Saved raw games to {raw_file}")
