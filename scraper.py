# scraper.py

from pathlib import Path
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


def scrape_bref_season_games(season: int, sleep=3) -> pd.DataFrame:
    """
    Scrape regular-season NBA games for a given season from Basketball-Reference.
    season: ending year (e.g., 2019 = 2018-19 season)
    """
    url = f"{BASE_URL}NBA_{season}_games.html"
    print(f"Scraping {url}")

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

            cols = row.find_all("td")
            if len(cols) < 5:
                continue

            date = cols[0].text.strip()
            away_team = cols[1].text.strip()
            away_pts = cols[2].text.strip()
            home_team = cols[3].text.strip()
            home_pts = cols[4].text.strip()

            # Skip future/unplayed games
            if not away_pts or not home_pts:
                continue

            games.append(
                {
                    "date": pd.to_datetime(date),
                    "season": season,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_pts": int(home_pts),
                    "away_pts": int(away_pts),
                }
            )

    time.sleep(sleep)  # be polite between seasons

    if not games:
        return pd.DataFrame(
            columns=["date", "season", "home_team", "away_team", "home_pts", "away_pts"]
        )

    return pd.DataFrame(games)


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

    return pd.concat(all_games, ignore_index=True)


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
