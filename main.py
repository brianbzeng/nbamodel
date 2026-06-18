import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Paths

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

BASE_URL = "https://www.basketball-reference.com/leagues/"

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


# Helper Functions

def safe_request(url: str, sleep: float = 3) -> str:
    # Avoid rate limit/ip ban
    print(f"Scraping: {url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    time.sleep(sleep)
    return response.text


def parse_schedule_table(soup: BeautifulSoup, season: int) -> pd.DataFrame:

    # Parses schedule tables into a standardized df. Returns date, season, home_team, away_team, home_pts, away_pts
    
    games = []

    rows = soup.select("table tbody tr")

    for row in rows:
        if "thead" in row.get("class", []):
            continue

        date_cell = row.find("th", {"data-stat": "date_game"})
        away_cell = row.find("td", {"data-stat": "visitor_team_name"})
        away_pts_cell = row.find("td", {"data-stat": "visitor_pts"})
        home_cell = row.find("td", {"data-stat": "home_team_name"})
        home_pts_cell = row.find("td", {"data-stat": "home_pts"})

        if not all([date_cell, away_cell, away_pts_cell, home_cell, home_pts_cell]):
            continue

        date_text = date_cell.get_text(strip=True)
        away_team = away_cell.get_text(strip=True)
        home_team = home_cell.get_text(strip=True)
        away_pts = away_pts_cell.get_text(strip=True)
        home_pts = home_pts_cell.get_text(strip=True)

        # Skip future games
        if not away_pts or not home_pts:
            continue

        try:
            games.append(
                {
                    "date": pd.to_datetime(date_text),
                    "season": season,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_pts": int(away_pts) if False else int(home_pts),
                    "away_pts": int(away_pts),
                }
            )
        except ValueError:
            continue

    return pd.DataFrame(games)


def get_season_month_urls(soup: BeautifulSoup, season: int) -> list[str]:
    # Basketball Reference provides each season schedule as pages seperated by month.
    month_urls = []
    seen = set()
    prefix = f"/leagues/NBA_{season}_games-"
    suffix = ".html"

    for link in soup.select('a[href^="/leagues/NBA_"]'):
        href = link.get("href", "")
        if not href.startswith(prefix) or not href.endswith(suffix):
            continue

        full_url = urljoin(BASE_URL, href)
        if full_url in seen:
            continue

        seen.add(full_url)
        month_urls.append(full_url)

    return month_urls


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Engineered but basic features that will help in analysis
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_win"] = (df["home_pts"] > df["away_pts"]).astype(int)
    df["margin"] = (df["home_pts"] - df["away_pts"]).abs()
    df["total_pts"] = df["home_pts"] + df["away_pts"]
    df["point_diff_home"] = df["home_pts"] - df["away_pts"]

    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return df


def save_df(df: pd.DataFrame, filename: str) -> Path:
    # Saving CSV function, helps with file organization as well
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} games to: {output_path}")
    return output_path


def combine_with_existing(new_df: pd.DataFrame, existing_file: Path) -> pd.DataFrame:
    # Data appending function
    if not existing_file.exists():
        return new_df

    old_df = pd.read_csv(existing_file, parse_dates=["date"])
    combined = pd.concat([old_df, new_df], ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team"],
        keep="last",
    )
    after = len(combined)

    removed = before - after
    if removed > 0:
        print(f"Removed {removed} duplicate game(s).")

    combined = combined.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return combined


# Scrapers 

def scrape_bref_season_games(season: int, sleep: float = 3) -> pd.DataFrame:
# Scrape the entire season by visiting each monthly schedule page.
    url = f"{BASE_URL}NBA_{season}_games.html"
    html = safe_request(url, sleep=sleep)
    soup = BeautifulSoup(html, "html.parser")

    month_urls = get_season_month_urls(soup, season)
    frames = []

    # The landing page can overlap with the first month, so we only scrape the discovered month pages and dedupe after concatenation.
    for month_url in month_urls:
        month_html = safe_request(month_url, sleep=sleep)
        month_soup = BeautifulSoup(month_html, "html.parser")
        month_df = parse_schedule_table(month_soup, season)

        if not month_df.empty:
            frames.append(month_df)

    if not frames:
        df = parse_schedule_table(soup, season)
    else:
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")

    df = add_basic_features(df)
    return df


def scrape_multiple_seasons(
    start_season: int,
    end_season: int,
    sleep: float = 3,
) -> pd.DataFrame:
    # Scrape multiple full seasons.
    frames = []

    for season in range(start_season, end_season + 1):
        try:
            df = scrape_bref_season_games(season, sleep=sleep)
            print(f"Finished {season}: {len(df):,} games")
            frames.append(df)
        except requests.HTTPError as e:
            print(f"HTTP error for season {season}: {e}")
        except Exception as e:
            print(f"Error for season {season}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def scrape_bref_month_games(season: int, month: int, sleep: float = 3) -> pd.DataFrame:
    # Scrape completed games for one month of one season. 
    if month not in MONTH_SLUGS:
        raise ValueError(
            f"Month {month} is not supported. Use one of: {sorted(MONTH_SLUGS)}"
        )

    slug = MONTH_SLUGS[month]
    url = f"{BASE_URL}NBA_{season}_games-{slug}.html"

    html = safe_request(url, sleep=sleep)
    soup = BeautifulSoup(html, "html.parser")

    df = parse_schedule_table(soup, season)
    df = add_basic_features(df)

    return df


# Menu

def get_int(prompt: str, min_value: Optional[int] = None) -> int:
    while True:
        value = input(prompt).strip()

        try:
            value_int = int(value)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if min_value is not None and value_int < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue

        return value_int


def menu() -> str:
    print("\n=== Basketball-Reference NBA Game Scraper ===")
    print("1) Scrape one full season")
    print("2) Scrape multiple full seasons")
    print("3) Scrape one month")
    print("4) Append scrape to an existing CSV")
    print("0) Exit")

    return input("Choose an option: ").strip()


def run_one_season() -> None:
    season = get_int("Season ending year, e.g. 2025 for 2024-25: ", min_value=1947)

    df = scrape_bref_season_games(season)

    if df.empty:
        print("No completed games found.")
        return

    save_df(df, f"bref_games_{season}.csv")


def run_multiple_seasons() -> None:
    start = get_int("Start season ending year, e.g. 2016: ", min_value=1947)
    end = get_int("End season ending year, e.g. 2025: ", min_value=start)

    df = scrape_multiple_seasons(start, end)

    if df.empty:
        print("No completed games found.")
        return

    save_df(df, f"bref_games_{start}_{end}.csv")


def run_one_month() -> None:
    season = get_int("Season ending year, e.g. 2026 for 2025-26: ", min_value=1947)
    month = get_int("Month number, e.g. 10, 11, 12, 1, 2, 3, 4, 5, 6: ")

    try:
        df = scrape_bref_month_games(season, month)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if df.empty:
        print("No completed games found.")
        return

    save_df(df, f"bref_games_{season}_{month:02d}.csv")


def run_append_mode() -> None:
    print("\nAppend mode scrapes data and merges it into an existing CSV without duplicates.")
    print("1) Append one full season")
    print("2) Append one month")

    choice = input("Choose append type: ").strip()

    existing_name = input(
        "Existing CSV filename inside data/raw, e.g. bref_games_master.csv: "
    ).strip()

    existing_file = RAW_DIR / existing_name

    if choice == "1":
        season = get_int("Season ending year: ", min_value=1947)
        new_df = scrape_bref_season_games(season)
    elif choice == "2":
        season = get_int("Season ending year: ", min_value=1947)
        month = get_int("Month number: ")
        try:
            new_df = scrape_bref_month_games(season, month)
        except ValueError as e:
            print(f"Error: {e}")
            return
    else:
        print("Invalid append choice.")
        return

    if new_df.empty:
        print("No completed games found.")
        return

    combined = combine_with_existing(new_df, existing_file)
    save_df(combined, existing_name)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        choice = menu()

        if choice == "1":
            run_one_season()
        elif choice == "2":
            run_multiple_seasons()
        elif choice == "3":
            run_one_month()
        elif choice == "4":
            run_append_mode()
        elif choice == "0":
            print("Exiting scraper.")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
