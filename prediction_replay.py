"""Reconstruct this season's pregame picks from archived scores and reports."""

from datetime import datetime
from pathlib import Path

import pandas as pd
from filelock import FileLock

from cleaner import clean_games, normalize_team_names
from daily_predictions import BASELINE_GAMES_FILE, read_daily_predictions
from game_updates import GAME_KEY, LIVE_GAMES_FILE, NBA_TIMEZONE, read_game_updates, write_snapshot
from predictor import LIVE_INJURIES_FILE, load_optional_team_injuries, predict_slate

REPLAY_FILE = LIVE_GAMES_FILE.with_name("nba_prediction_replay.csv")
MATCHUP_COLUMNS = ["season", "home_team", "away_team"]
REPLAY_KEY = ["game_date", *MATCHUP_COLUMNS]
REPLAY_COLUMNS = [*REPLAY_KEY, "home_win_probability"]


def read_prediction_replay(path: Path = REPLAY_FILE) -> pd.DataFrame:
    try:
        return pd.read_csv(path, parse_dates=["game_date"])
    except FileNotFoundError:
        return pd.DataFrame(columns=REPLAY_COLUMNS)


def _load_replay_injuries(games: pd.DataFrame, first_game: pd.Timestamp) -> pd.DataFrame | None:
    historical = load_optional_team_injuries(games)
    if historical is not None:
        historical = historical[historical["game_date"] < first_game]

    try:
        # This archive is filtered to reports published before each team's tipoff during ingestion.
        verified = pd.read_csv(LIVE_INJURIES_FILE, parse_dates=["game_date", "latest_report_datetime"])
    except FileNotFoundError:
        return historical
    verified = verified[verified["game_date"] >= first_game]
    verified = verified[verified["latest_report_datetime"].dt.normalize() <= verified["game_date"]]
    frames = [frame for frame in (historical, verified) if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else None


def build_prediction_replay(
    games: pd.DataFrame, injuries: pd.DataFrame | None, targets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for game_date, slate in targets.groupby("date", sort=True):
        matchups = slate[MATCHUP_COLUMNS].to_dict("records")
        for game_id, matchup in enumerate(matchups):
            matchup["game_id"] = str(game_id)

        # Historical results are dated, so only earlier dates have a verified ordering here.
        prior = games[games["date"] < game_date]
        eligible_reports = injuries[injuries["game_date"] <= game_date] if injuries is not None else None
        probabilities = predict_slate(prior, matchups, eligible_reports, game_date, model="random_forest")
        for matchup in matchups:
            probability = float(probabilities[matchup["game_id"]])
            if not 0 <= probability <= 1:
                raise ValueError(f"Invalid replay probability for {matchup['away_team']} at {matchup['home_team']} on {game_date:%Y-%m-%d}.")
            rows.append({"game_date": game_date, **matchup, "home_win_probability": round(probability, 6)})

    return pd.DataFrame(rows, columns=REPLAY_COLUMNS)


def refresh_prediction_replay(path: Path = REPLAY_FILE) -> tuple[pd.DataFrame, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(".lock"), timeout=120):
        existing = read_prediction_replay(path)
        today = datetime.now(NBA_TIMEZONE).date()
        history = pd.read_csv(BASELINE_GAMES_FILE, parse_dates=["date"])
        updates = read_game_updates()
        raw = pd.concat([history, updates], ignore_index=True) if not updates.empty else history
        games = normalize_team_names(clean_games(raw)).drop_duplicates(GAME_KEY, keep="last")
        games = games[games["date"] <= pd.Timestamp(today)]
        season = updates.loc[pd.to_datetime(updates["date"]) <= pd.Timestamp(today), "season"].max()
        current = games[games["season"] == season]

        live = read_daily_predictions()
        saved = live[
            live["home_win_probability"].notna() & (live["predicted_at_utc"] < live["tipoff_utc"])
        ]
        scored_keys = pd.MultiIndex.from_frame(existing[REPLAY_KEY]).union(pd.MultiIndex.from_frame(saved[REPLAY_KEY]))
        current_keys = pd.MultiIndex.from_frame(current[GAME_KEY])
        pending = current[~current_keys.isin(scored_keys)]
        if pending.empty and path.exists():
            return existing, False

        injuries = _load_replay_injuries(games, current["date"].min()) if not pending.empty else None
        fresh = build_prediction_replay(games, injuries, pending)
        replay = (pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh).sort_values(
            ["game_date", "home_team", "away_team"], ascending=[False, True, True], ignore_index=True,
        )

        write_snapshot(replay, path, date_format="%Y-%m-%d")
        return replay, True


if __name__ == "__main__":
    replay, changed = refresh_prediction_replay()
    print(f"{'Saved' if changed else 'Verified'} {len(replay):,} archived replay picks.")
