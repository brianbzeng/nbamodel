"""Maintain player performance from completed NBA box scores."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import gettempdir

import pandas as pd
import requests
from filelock import FileLock

from cleaner import TEAM_MAP_BREF
from game_updates import (
    GAME_COLUMNS,
    GAME_KEY,
    NBA_TIMEZONE,
    completed_games,
    fetch_scoreboard,
    read_game_updates,
    write_snapshot,
)


PLAYER_STATS_FILE = Path(__file__).resolve().parent / "data/raw/nba_player_game_stats.csv"
HISTORICAL_GAMES_FILE = PLAYER_STATS_FILE.with_name("bref_games_2022_2025.csv")
ARCHIVE_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_nba_player_boxscores/player_box_{season}.csv"
)
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
FINAL_STATUSES = {"STATUS_FINAL", "STATUS_FINAL_OVERTIME"}
TEAM_GAME_KEY = ["date", "season", "team"]
PLAYER_GAME_KEY = [*TEAM_GAME_KEY, "player_id"]
PLAYER_STATS_COLUMNS = [*PLAYER_GAME_KEY, "player", "minutes", "game_score"]


def read_player_game_stats(path: Path = PLAYER_STATS_FILE) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"], dtype={"player_id": str, "minutes": float})
    return pd.DataFrame(columns=PLAYER_STATS_COLUMNS).astype(
        {"date": "datetime64[ns]", "season": int, "minutes": float, "game_score": float}
    )


def _game_score(
    points, field_goals_made, field_goals_attempted, free_throws_made, free_throws_attempted,
    offensive_rebounds, defensive_rebounds, steals, assists, blocks, fouls, turnovers,
):
    # Hollinger's box-score productivity estimate, used only after the game is final.
    return (
        points + 0.4 * field_goals_made - 0.7 * field_goals_attempted
        - 0.4 * (free_throws_attempted - free_throws_made)
        + 0.7 * offensive_rebounds + 0.3 * defensive_rebounds + steals
        + 0.7 * assists + 0.7 * blocks - 0.4 * fouls - turnovers
    )


def _canonical_games(games: pd.DataFrame) -> pd.DataFrame:
    games = games[GAME_COLUMNS].copy()
    for side in ("home_team", "away_team"):
        games[side] = games[side].map(TEAM_MAP_BREF)
    if games[["home_team", "away_team"]].isna().any().any():
        raise ValueError("Unrecognized NBA team in completed games.")
    return games.drop_duplicates(GAME_KEY, keep="last")


def _archive_rows(season: int, games: pd.DataFrame) -> pd.DataFrame | None:
    response = requests.get(ARCHIVE_URL.format(season=season), timeout=60)
    if response.status_code == 404:
        # An archive for the new season may not have been published yet.
        return None
    response.raise_for_status()
    stat_columns = [
        "points", "field_goals_made", "field_goals_attempted", "free_throws_made",
        "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "steals",
        "assists", "blocks", "fouls", "turnovers",
    ]
    rows = pd.read_csv(
        BytesIO(response.content),
        usecols=[
            "game_date", "season", "athlete_id", "athlete_display_name", "minutes",
            "team_display_name", "opponent_team_display_name", "home_away",
            "team_score", "opponent_team_score", *stat_columns,
        ],
        parse_dates=["game_date"],
        dtype={"athlete_id": str, "minutes": float},
    ).rename(columns={"game_date": "date", "athlete_id": "player_id", "athlete_display_name": "player"})
    rows["team"] = rows["team_display_name"].map(TEAM_MAP_BREF)
    opponent = rows["opponent_team_display_name"].map(TEAM_MAP_BREF)
    is_home = rows["home_away"].eq("home")
    rows["home_team"] = rows["team"].where(is_home, opponent)
    rows["away_team"] = rows["team"].where(~is_home, opponent)
    rows["home_pts"] = rows["team_score"].where(is_home, rows["opponent_team_score"])
    rows["away_pts"] = rows["team_score"].where(~is_home, rows["opponent_team_score"])
    # Match the ledger on both teams, local game date and final score. This also excludes exhibitions.
    rows = rows.loc[rows["minutes"].gt(0)].merge(games, on=GAME_COLUMNS, validate="many_to_one")
    rows["game_score"] = _game_score(*(rows[column] for column in stat_columns))
    return rows.dropna(subset=["player_id", "player", "game_score"])[PLAYER_STATS_COLUMNS]


def _summary_rows(summary: dict, game: dict) -> pd.DataFrame:
    competition = summary["header"]["competitions"][0]
    if competition["status"]["type"]["name"] not in FINAL_STATUSES:
        raise ValueError("NBA box score is not final.")
    competitors = {team["homeAway"]: team for team in competition["competitors"]}
    game_date = pd.Timestamp(datetime.fromisoformat(competition["date"]).astimezone(NBA_TIMEZONE).date())
    season = int(summary["header"]["season"]["year"])
    if game_date != game["date"] or season != game["season"] or any(
        TEAM_MAP_BREF[competitors[side]["team"]["displayName"]] != game[f"{side}_team"]
        or int(competitors[side]["score"]) != game[f"{side}_pts"]
        for side in ("home", "away")
    ):
        raise ValueError("NBA box score does not match its final game result.")

    rows = []
    for team in summary["boxscore"]["players"]:
        abbreviation = TEAM_MAP_BREF[team["team"]["displayName"]]
        for group in team["statistics"]:
            for athlete in group["athletes"]:
                if athlete.get("didNotPlay") or not athlete.get("stats"):
                    continue
                stats = dict(zip(group["keys"], athlete["stats"], strict=True))
                minute, _, second = stats["minutes"].partition(":")
                minutes = int(minute) + int(second or 0) / 60
                if minutes <= 0:
                    continue
                fgm, fga = map(int, stats["fieldGoalsMade-fieldGoalsAttempted"].split("-"))
                ftm, fta = map(int, stats["freeThrowsMade-freeThrowsAttempted"].split("-"))
                score = _game_score(
                    int(stats["points"]), fgm, fga, ftm, fta, int(stats["offensiveRebounds"]),
                    int(stats["defensiveRebounds"]), int(stats["steals"]), int(stats["assists"]),
                    int(stats["blocks"]), int(stats["fouls"]), int(stats["turnovers"]),
                )
                player = athlete["athlete"]
                rows.append((game_date, season, abbreviation, player["id"], player["displayName"], minutes, score))

    frame = pd.DataFrame(rows, columns=PLAYER_STATS_COLUMNS)
    if set(frame["team"]) != {game["home_team"], game["away_team"]} or frame.groupby("team").size().lt(5).any():
        raise ValueError("Incomplete NBA player box score.")
    return frame


def _fetch_summary(event_id: str, game: dict) -> pd.DataFrame | None:
    try:
        response = requests.get(SUMMARY_URL, params={"event": event_id}, timeout=20)
        response.raise_for_status()
        return _summary_rows(response.json(), game)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
        print(f"Skipping ESPN box score {event_id}: {error}", file=sys.stderr)
        return None


def _missing_games(games: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    covered = rows.groupby(TEAM_GAME_KEY).size()
    return games.loc[
        [
            any(covered.get((game.date, game.season, team), 0) < 5 for team in (game.home_team, game.away_team))
            for game in games.itertuples(index=False)
        ]
    ]


def _missing_box_scores(games: pd.DataFrame) -> list[pd.DataFrame]:
    if games.empty:
        return []
    first_day = games["date"].min().date()
    last_day = games["date"].max().date()
    # A date-specific feed can contain a final that the monthly feed has not published yet.
    periods = sorted(set(games["date"].dt.strftime("%Y%m")))
    periods.extend(day.strftime("%Y%m%d") for day in sorted(set(games["date"].dt.date))[-2:])
    with ThreadPoolExecutor(max_workers=4) as pool:
        boards = list(pool.map(fetch_scoreboard, periods))
        wanted = games.set_index(GAME_COLUMNS).index
        requests_by_game = {}
        for board in boards:
            for event in board:
                final = completed_games([event], first_day, last_day)
                if final.empty:
                    continue
                final = _canonical_games(final)
                game = final.iloc[0].to_dict()
                key = tuple(game[column] for column in GAME_COLUMNS)
                if key in wanted:
                    requests_by_game[key] = (event["id"], game)
        summaries = pool.map(lambda args: _fetch_summary(*args), requests_by_game.values())
        return [summary for summary in summaries if summary is not None]


def refresh_player_stats(path: Path = PLAYER_STATS_FILE, *, backfill: bool = False) -> tuple[pd.DataFrame, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    live_games = _canonical_games(read_game_updates())
    if backfill:
        history = _canonical_games(pd.read_csv(HISTORICAL_GAMES_FILE, parse_dates=["date"]))
        games = pd.concat([history, live_games]).drop_duplicates(GAME_KEY, keep="last")
    else:
        games = live_games

    lock_path = Path(gettempdir()) / f"nbamodel-{path.name}.lock"
    with FileLock(lock_path, timeout=120):
        existing = read_player_game_stats(path)
        rows = existing
        if backfill:
            seasons = list(games.groupby("season", sort=True))
            with ThreadPoolExecutor(max_workers=3) as pool:
                archives = list(pool.map(lambda item: _archive_rows(*item), seasons))
            rows = pd.concat([existing, *(archive for archive in archives if archive is not None)], ignore_index=True)
        summaries = _missing_box_scores(_missing_games(games, rows))
        if summaries:
            rows = pd.concat(summaries if rows.empty else [rows, *summaries], ignore_index=True)
        rows = rows.drop_duplicates(PLAYER_GAME_KEY, keep="last").sort_values(PLAYER_GAME_KEY).reset_index(drop=True)
        rows["game_score"] = rows["game_score"].round(1)
        rows["minutes"] = rows["minutes"].round(2)
        if not rows.equals(existing):
            write_snapshot(rows, path, date_format="%Y-%m-%d")
        return rows, len(_missing_games(games, rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill", action="store_true", help="Seed from historical ESPN archive files.")
    args = parser.parse_args()
    rows, missing = refresh_player_stats(backfill=args.backfill)
    print(f"Verified {len(rows):,} player box-score lines. Missing {missing:,} final games.")
