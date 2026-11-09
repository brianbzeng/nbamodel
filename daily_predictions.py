"""Build and save pregame probabilities for the upcoming NBA slate."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
from filelock import FileLock

from cleaner import TEAM_MAP_BREF, clean_games, normalize_team_names
from game_updates import LIVE_GAMES_FILE, NBA_TIMEZONE, fetch_scoreboard, nba_competition, read_game_updates, write_snapshot

DAILY_PREDICTIONS_FILE = LIVE_GAMES_FILE.with_name("nba_daily_predictions.csv")
BASELINE_GAMES_FILE = LIVE_GAMES_FILE.with_name("bref_games_2022_2025.csv")
PREDICTION_COLUMNS = [
    "game_id", "game_date", "tipoff_utc", "season", "home_team", "away_team", "status",
    "home_win_probability", "predicted_at_utc",
]
STATUS_NAMES = {
    "STATUS_SCHEDULED": "scheduled",
    "STATUS_FINAL": "final",
    "STATUS_FINAL_OVERTIME": "final",
    "STATUS_POSTPONED": "postponed",
    "STATUS_SUSPENDED": "postponed",
    "STATUS_CANCELED": "canceled",
    "STATUS_CANCELLED": "canceled",
}


def _typed_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame["tipoff_utc"] = pd.to_datetime(frame["tipoff_utc"], format="mixed", utc=True)
    frame["predicted_at_utc"] = pd.to_datetime(frame["predicted_at_utc"], format="mixed", utc=True)
    frame["home_win_probability"] = frame["home_win_probability"].astype(float)
    return frame


def read_daily_predictions(path: Path = DAILY_PREDICTIONS_FILE) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype={"game_id": str})
    except FileNotFoundError:
        frame = pd.DataFrame(columns=PREDICTION_COLUMNS)
    return _typed_predictions(frame)


def scheduled_games(events: list[dict], first_date: date, last_date: date) -> list[dict]:
    games = {}
    for event in events:
        competition = nba_competition(event)
        if competition is None:
            continue
        status_type = competition["status"]["type"]
        status = STATUS_NAMES.get(status_type["name"])
        if status is None and status_type.get("state") == "in":
            status = "in_progress"
        if status is None or (status == "scheduled" and competition.get("timeValid") is False):
            continue

        tipoff = datetime.fromisoformat(competition["date"])
        if tipoff.tzinfo is None:
            raise ValueError(f"ESPN game {event['id']} has no time zone.")
        game_date = tipoff.astimezone(NBA_TIMEZONE).date()
        if not first_date <= game_date <= last_date:
            continue

        teams = {
            team["homeAway"]: TEAM_MAP_BREF.get(team["team"].get("displayName"))
            for team in competition["competitors"]
        }
        home_team, away_team = teams.get("home"), teams.get("away")
        if not home_team or not away_team or home_team == away_team:
            continue
        game_id = str(event["id"])
        games[game_id] = {
            "game_id": game_id,
            "game_date": pd.Timestamp(game_date),
            "tipoff_utc": tipoff.astimezone(UTC),
            "season": int(event["season"]["year"]),
            "home_team": home_team,
            "away_team": away_team,
            "status": status,
            "home_win_probability": None,
            "predicted_at_utc": None,
        }
    return list(games.values())


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _load_model_games() -> pd.DataFrame:
    history = pd.read_csv(BASELINE_GAMES_FILE)
    updates = read_game_updates()
    if not updates.empty:
        history = pd.concat([history, updates], ignore_index=True)
    return normalize_team_names(clean_games(history))


def _predict_games(matchups: list[dict], as_of: datetime) -> dict[str, float]:
    from predictor import load_optional_team_injuries, predict_slate

    games = _load_model_games()
    injuries = load_optional_team_injuries(games)
    if injuries is not None:
        eastern_as_of = as_of.astimezone(NBA_TIMEZONE).replace(tzinfo=None)
        injuries = injuries[injuries["latest_report_datetime"] <= eastern_as_of]

    slates = {}
    for matchup in matchups:
        slates.setdefault(matchup["game_date"], []).append(matchup)
    probabilities = {}
    for game_date, slate in sorted(slates.items()):
        probabilities.update(predict_slate(games, slate, injuries, game_date))
    return probabilities


def refresh_daily_predictions(path: Path = DAILY_PREDICTIONS_FILE) -> tuple[pd.DataFrame, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(".lock"), timeout=120):
        existing = read_daily_predictions(path)
        records = {row["game_id"]: row for row in existing.to_dict("records")}
        today = _utc_now().astimezone(NBA_TIMEZONE).date()
        yesterday, tomorrow = today - timedelta(days=1), today + timedelta(days=1)
        periods = [day.strftime("%Y%m%d") for day in (yesterday, today, tomorrow)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            boards = list(pool.map(fetch_scoreboard, periods))
        slate = scheduled_games([event for board in boards for event in board], yesterday, tomorrow)

        for game in slate:
            previous = records.get(game["game_id"])
            if previous is not None:
                # Keep final results and started-game picks if a cached schedule regresses.
                if previous["status"] == "final" and game["status"] != "final":
                    continue
                same_matchup = all(previous[key] == game[key] for key in ("home_team", "away_team", "game_date"))
                if same_matchup and previous["status"] == "in_progress" and game["status"] == "scheduled":
                    continue
                if same_matchup and pd.notna(previous["predicted_at_utc"]) and previous["predicted_at_utc"] < game["tipoff_utc"]:
                    game["home_win_probability"] = previous["home_win_probability"]
                    game["predicted_at_utc"] = previous["predicted_at_utc"]
            records[game["game_id"]] = game

        as_of = _utc_now()
        matchups = [
            records[game["game_id"]] for game in slate
            if records[game["game_id"]]["status"] == "scheduled"
            and as_of < records[game["game_id"]]["tipoff_utc"]
        ]
        if matchups:
            probabilities = _predict_games(matchups, as_of)
            generated_at = _utc_now()
            for game in matchups:
                # The slate can tip off while the batch is fitting.
                if generated_at >= game["tipoff_utc"]:
                    continue
                probability = float(probabilities[game["game_id"]])
                if not 0 <= probability <= 1:
                    raise ValueError(f"Invalid home win probability for ESPN game {game['game_id']}.")
                probability = round(probability, 6)
                if probability != game["home_win_probability"]:
                    game["home_win_probability"] = probability
                    game["predicted_at_utc"] = generated_at.replace(microsecond=0)

        updated = _typed_predictions(pd.DataFrame(records.values(), columns=PREDICTION_COLUMNS))
        updated = updated.sort_values(["tipoff_utc", "game_id"]).reset_index(drop=True)
        if path.exists() and updated.equals(existing):
            return existing, False
        write_snapshot(updated, path)
        return updated, True


if __name__ == "__main__":
    snapshot, changed = refresh_daily_predictions()
    print(f"{'Saved' if changed else 'Verified'} {len(snapshot):,} scheduled NBA games and "
          f"{snapshot['home_win_probability'].count():,} pregame predictions.")
