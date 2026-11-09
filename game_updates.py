"""Maintain the completed NBA games added after the bundled 2021–25 history."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from filelock import FileLock
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cleaner import TEAM_MAP_BREF

LIVE_GAMES_FILE = Path(__file__).resolve().parent / "data/raw/nba_game_updates.csv"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
FIRST_UPDATE_DATE = date(2025, 10, 1)
NBA_TIMEZONE = ZoneInfo("America/New_York")
GAME_KEY = ["date", "season", "home_team", "away_team"]
GAME_COLUMNS = [*GAME_KEY, "home_pts", "away_pts"]
GAME_SEASON_TYPES = {2, 3, 5}  # Regular season (including the Cup), playoffs, play-in.
SCOREBOARD_LIMIT = 500


def read_game_updates(path: Path = LIVE_GAMES_FILE) -> pd.DataFrame:
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except FileNotFoundError:
        return pd.DataFrame(columns=GAME_COLUMNS)


def write_snapshot(frame: pd.DataFrame, path: Path, date_format: str | None = None) -> None:
    # Readers always see either the previous complete CSV or the new one.
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
        temporary_path = Path(output.name)
    try:
        frame.to_csv(temporary_path, index=False, date_format=date_format, lineterminator="\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def fetch_scoreboard(period: str) -> list[dict]:
    retries = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.get(
            SCOREBOARD_URL, params={"dates": period, "limit": SCOREBOARD_LIMIT}, timeout=20
        )
        response.raise_for_status()
        events = response.json()["events"]
    if not isinstance(events, list) or len(events) >= SCOREBOARD_LIMIT:
        raise ValueError(f"Incomplete NBA scoreboard for {period}.")
    return events


def nba_competition(event: dict) -> dict | None:
    if event["season"]["type"] not in GAME_SEASON_TYPES:
        return None
    competition = event["competitions"][0]
    return None if competition.get("type", {}).get("abbreviation") == "ALLSTAR" else competition


def completed_games(events: list[dict], start: date, today: date) -> pd.DataFrame:
    games = []
    for event in events:
        competition = nba_competition(event)
        if competition is None:
            continue
        if competition["status"]["type"]["name"] not in {"STATUS_FINAL", "STATUS_FINAL_OVERTIME"}:
            continue

        game_date = datetime.fromisoformat(competition["date"]).astimezone(NBA_TIMEZONE).date()
        if not start <= game_date <= today:
            continue
        teams = {team["homeAway"]: team for team in competition["competitors"]}
        home_team = teams["home"]["team"]["displayName"]
        away_team = teams["away"]["team"]["displayName"]
        home_pts = int(teams["home"]["score"])
        away_pts = int(teams["away"]["score"])
        if (
            home_team not in TEAM_MAP_BREF
            or away_team not in TEAM_MAP_BREF
            or TEAM_MAP_BREF[home_team] == TEAM_MAP_BREF[away_team]
            or min(home_pts, away_pts) <= 0
            or home_pts == away_pts
        ):
            raise ValueError(f"Invalid final NBA result for ESPN game {event['id']}.")

        games.append((game_date, int(event["season"]["year"]), home_team, away_team, home_pts, away_pts))

    frame = pd.DataFrame(games, columns=GAME_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def refresh_game_updates(path: Path = LIVE_GAMES_FILE) -> tuple[pd.DataFrame, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(".lock"), timeout=120):
        existing = read_game_updates(path)
        today = datetime.now(NBA_TIMEZONE).date()
        start = FIRST_UPDATE_DATE
        if not existing.empty:
            start = max(start, existing["date"].max().date() - timedelta(days=14))

        periods = list(pd.period_range(start=start, end=today, freq="M").strftime("%Y%m"))
        # Dated feeds can publish late-night finals before the monthly feed catches up.
        periods.extend(day.strftime("%Y%m%d") for day in (today - timedelta(days=1), today))
        with ThreadPoolExecutor(max_workers=3) as pool:
            scoreboards = list(pool.map(fetch_scoreboard, periods))
        fresh = completed_games([event for board in scoreboards for event in board], start, today)
        if fresh.empty:
            if existing.empty:
                raise ValueError("The NBA scoreboard returned no completed games to seed the snapshot.")
            return existing, False

        combined = fresh if existing.empty else pd.concat([existing, fresh], ignore_index=True)
        updated = combined.drop_duplicates(GAME_KEY, keep="last").sort_values(GAME_KEY).reset_index(drop=True)
        if updated.equals(existing):
            return existing, False

        write_snapshot(updated, path, date_format="%Y-%m-%d")
        return updated, True


if __name__ == "__main__":
    games, changed = refresh_game_updates()
    state = "Updated" if changed else "Verified"
    print(f"{state} {len(games):,} completed games through {games['date'].max():%Y-%m-%d}.")
