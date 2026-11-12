"""Build and save pregame probabilities for the upcoming NBA slate."""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import pandas as pd
from filelock import FileLock

from cleaner import TEAM_MAP_BREF, clean_games, normalize_team_names
from game_updates import (
    LIVE_GAMES_FILE,
    NBA_TIMEZONE,
    completed_games,
    fetch_scoreboard,
    nba_competition,
    read_game_updates,
    write_snapshot,
)

DAILY_PREDICTIONS_FILE = LIVE_GAMES_FILE.with_name("nba_daily_predictions.csv")
BASELINE_GAMES_FILE = LIVE_GAMES_FILE.with_name("bref_games_2022_2025.csv")
PREGAME_WINDOW = timedelta(minutes=60)
PREDICTION_DISPLAY_TZ = ZoneInfo("America/Los_Angeles")
NEXT_SLATE_OPENS = time(23, 30)
DEFAULT_PREDICTION_MODEL = "random_forest"
COMPARISON_MODEL = "logistic"
MATCHUP_KEY = ["game_date", "home_team", "away_team"]
PROVENANCE_TIMES = ["home_injury_report_utc", "away_injury_report_utc"]
REST_DAY_COLUMNS = ["home_rest_days", "away_rest_days"]
PROVENANCE_COUNTS = ["same_day_finals", "home_out_count", "away_out_count", *REST_DAY_COLUMNS]
PLAYER_IMPACT_COLUMN = "home_player_impact_pp"
BASELINE_PROBABILITY_COLUMN = "baseline_home_win_probability"
LOGISTIC_PROBABILITY_COLUMN = "logistic_home_win_probability"
PLAYER_IMPACT_PROBABILITY_COLUMN = "player_impact_home_win_probability"
EXPERIMENT_VERSION_COLUMN = "experiment_version"
PLAYER_IMPACT_EXPERIMENT = "rf_player_impact_v1"
ELO_PROBABILITY_COLUMN = "elo_home_win_probability"
ELO_EXPERIMENT_VERSION_COLUMN = "elo_experiment_version"
ELO_EXPERIMENT = "rf_elo_v1"
NO_REST_PROBABILITY_COLUMN = "no_rest_home_win_probability"
NO_REST_EXPERIMENT_VERSION_COLUMN = "no_rest_experiment_version"
NO_REST_EXPERIMENT = "rf_no_rest_v1"


class ShadowExperiment(NamedTuple):
    key: str
    probability_column: str
    version_column: str
    version: str
    label: str


def _shadow_experiment(key: str, column_prefix: str, version: str, label: str) -> ShadowExperiment:
    return ShadowExperiment(key, f"{column_prefix}_home_win_probability", f"{column_prefix}_experiment_version", version, label)


PLAYER_IMPACT_KEY = "player_impact"
BLEND_KEY = "rf_logistic_blend"
IMPACT_BLEND_KEY = "player_impact_logistic_blend"
IMPACT_BLEND_LOW_KEY = "player_impact_logistic_blend_30"
IMPACT_BLEND_HIGH_KEY = "player_impact_logistic_blend_70"
IMPACT_BLEND_WEIGHTS = {
    IMPACT_BLEND_LOW_KEY: 0.3,
    IMPACT_BLEND_KEY: 0.5,
    IMPACT_BLEND_HIGH_KEY: 0.7,
}
SHADOW_EXPERIMENTS = (
    ShadowExperiment(PLAYER_IMPACT_KEY, PLAYER_IMPACT_PROBABILITY_COLUMN, EXPERIMENT_VERSION_COLUMN,
                     PLAYER_IMPACT_EXPERIMENT, "Random forest + player impact"),
    ShadowExperiment("random_forest_elo", ELO_PROBABILITY_COLUMN, ELO_EXPERIMENT_VERSION_COLUMN,
                     ELO_EXPERIMENT, "Random forest + Elo"),
    ShadowExperiment("random_forest_no_rest", NO_REST_PROBABILITY_COLUMN, NO_REST_EXPERIMENT_VERSION_COLUMN,
                     NO_REST_EXPERIMENT, "Random forest without rest"),
    _shadow_experiment("random_forest_margin", "margin", "rf_margin_v1", "Point-margin forest"),
    _shadow_experiment(BLEND_KEY, "blend", "rf_logistic_blend_v1", "Random forest + logistic blend"),
    _shadow_experiment("random_forest_weighted_form", "weighted_form", "rf_weighted_form_v1", "Weighted recent form"),
    _shadow_experiment("random_forest_schedule_density", "schedule_density", "rf_schedule_density_v1", "Schedule density"),
    _shadow_experiment(IMPACT_BLEND_KEY, "impact_blend", "rf_player_impact_logistic_blend_v1",
                       "Player impact + logistic blend"),
    _shadow_experiment(IMPACT_BLEND_LOW_KEY, "impact_blend_30", "rf_player_impact_logistic_blend_30_v1",
                       "Player impact 30% + logistic 70%"),
    _shadow_experiment(IMPACT_BLEND_HIGH_KEY, "impact_blend_70", "rf_player_impact_logistic_blend_70_v1",
                       "Player impact 70% + logistic 30%"),
)
SHADOW_COLUMNS = [column for spec in SHADOW_EXPERIMENTS for column in (spec.probability_column, spec.version_column)]
PICK_VALUES = [
    "home_win_probability", "model", *PROVENANCE_COUNTS, *PROVENANCE_TIMES,
    PLAYER_IMPACT_COLUMN, BASELINE_PROBABILITY_COLUMN, LOGISTIC_PROBABILITY_COLUMN, *SHADOW_COLUMNS,
]
PICK_COLUMNS = [*PICK_VALUES, "predicted_at_utc"]
PREDICTION_COLUMNS = [
    "game_id", "game_date", "tipoff_utc", "season", "home_team", "away_team", "status",
    "home_win_probability", "predicted_at_utc", "same_day_finals",
    "home_injury_report_utc", "away_injury_report_utc", "home_out_count", "away_out_count", "model",
    PLAYER_IMPACT_COLUMN, *REST_DAY_COLUMNS, LOGISTIC_PROBABILITY_COLUMN, *SHADOW_COLUMNS,
    BASELINE_PROBABILITY_COLUMN,
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
    frame = frame.reindex(columns=PREDICTION_COLUMNS)
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    for column in ("tipoff_utc", "predicted_at_utc", *PROVENANCE_TIMES):
        frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True)
    for column in PROVENANCE_COUNTS:
        frame[column] = frame[column].astype("Int64")
    for column in ("home_win_probability", BASELINE_PROBABILITY_COLUMN, PLAYER_IMPACT_COLUMN, LOGISTIC_PROBABILITY_COLUMN):
        frame[column] = frame[column].astype(float)
    for spec in SHADOW_EXPERIMENTS:
        frame[spec.probability_column] = frame[spec.probability_column].astype(float)
        frame[spec.version_column] = frame[spec.version_column].astype("string")
    # Saved picks from before the model switch were all logistic regression.
    frame["model"] = frame["model"].astype("string").fillna("logistic").where(frame["home_win_probability"].notna())
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
            **dict.fromkeys(PICK_COLUMNS),
        }
    return list(games.values())


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _fetch_slate(as_of: datetime) -> tuple[list[dict], list[dict]]:
    today = as_of.astimezone(NBA_TIMEZONE).date()
    yesterday, tomorrow = today - timedelta(days=1), today + timedelta(days=1)
    periods = [day.strftime("%Y%m%d") for day in (yesterday, today, tomorrow)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        boards = list(pool.map(fetch_scoreboard, periods))
    events = [event for board in boards for event in board]
    return scheduled_games(events, yesterday, tomorrow), events


def _within_pregame_window(tipoff: datetime, as_of: datetime) -> bool:
    return timedelta(0) < tipoff - as_of <= PREGAME_WINDOW


def _slate_window_open(tipoff: datetime, as_of: datetime) -> bool:
    local_now = as_of.astimezone(PREDICTION_DISPLAY_TZ)
    game_day = tipoff.astimezone(PREDICTION_DISPLAY_TZ).date()
    last_day = local_now.date()
    if local_now.time() >= NEXT_SLATE_OPENS:
        last_day += timedelta(days=1)
    return as_of < tipoff and local_now.date() <= game_day <= last_day


def _can_predict(game: dict, as_of: datetime) -> bool:
    return game["status"] == "scheduled" and _slate_window_open(game["tipoff_utc"], as_of)


def pregame_window_open() -> bool:
    slate, _ = _fetch_slate(_utc_now())
    as_of = _utc_now()
    return any(
        _can_predict(game, as_of) and _within_pregame_window(game["tipoff_utc"], as_of)
        for game in slate
    )


def _final_games(events: list[dict], slate: list[dict], as_of: datetime) -> pd.DataFrame:
    today = as_of.astimezone(NBA_TIMEZONE).date()
    final_ids = {
        game["game_id"] for game in slate
        if game["status"] == "final" and game["tipoff_utc"] < as_of
    }
    fresh = completed_games(
        [event for event in events if str(event["id"]) in final_ids],
        today - timedelta(days=1),
        today,
    )
    saved = read_game_updates()
    saved = saved[saved["date"] <= pd.Timestamp(today)]
    frames = [frame for frame in (saved, fresh) if not frame.empty]
    scores = pd.concat(frames, ignore_index=True) if frames else fresh
    finals = normalize_team_names(clean_games(scores)).rename(columns={"date": "game_date"})
    future_games = {
        tuple(game[column] for column in MATCHUP_KEY)
        for game in slate if game["tipoff_utc"] >= as_of
    }
    if future_games:
        finals = finals[~pd.MultiIndex.from_frame(finals[MATCHUP_KEY]).isin(future_games)]
    return finals.drop_duplicates(MATCHUP_KEY, keep="last")


def _predict_games(matchups: list[dict], as_of: datetime, finals: pd.DataFrame) -> dict[str, dict]:
    from player_impact import PLAYER_IMPACT_MODEL, player_adjusted_picks
    from predictor import load_optional_team_injuries, predict_slate_models

    history = normalize_team_names(clean_games(pd.read_csv(BASELINE_GAMES_FILE)))
    updates = finals.rename(columns={"game_date": "date"})
    games = pd.concat([history, updates], ignore_index=True) if not updates.empty else history
    eastern_as_of = as_of.astimezone(NBA_TIMEZONE).replace(tzinfo=None)
    injuries = load_optional_team_injuries(games, as_of=eastern_as_of)

    slates = {}
    for matchup in matchups:
        slates.setdefault(matchup["game_date"], []).append(matchup)
    predictions = {}
    derived_models = (PLAYER_IMPACT_KEY, BLEND_KEY, *IMPACT_BLEND_WEIGHTS)
    shadow_models = tuple(spec.key for spec in SHADOW_EXPERIMENTS if spec.key not in derived_models)
    for game_date, slate in sorted(slates.items()):
        same_day_finals = updates[updates["date"] == game_date]
        model_probabilities, rest_days = predict_slate_models(
            games, slate, injuries, game_date, same_day_finals=same_day_finals,
            models=(DEFAULT_PREDICTION_MODEL, COMPARISON_MODEL, *shadow_models),
            optional_models=shadow_models,
        )
        probabilities = model_probabilities[DEFAULT_PREDICTION_MODEL]
        logistic_picks = model_probabilities[COMPARISON_MODEL]
        try:
            model_probabilities[PLAYER_IMPACT_KEY] = player_adjusted_picks(slate, probabilities, injuries)
        except (KeyError, OSError, ValueError):
            logging.getLogger(__name__).exception("Could not score player impact for %s", game_date.date())
        impact_picks = model_probabilities.get(PLAYER_IMPACT_KEY, {})
        blends = [(BLEND_KEY, probabilities, 0.5)]
        blends.extend((key, impact_picks, weight) for key, weight in IMPACT_BLEND_WEIGHTS.items())
        for blend, forest_picks, weight in blends:
            model_probabilities[blend] = {
                game_id: weight * probability + (1 - weight) * logistic_picks[game_id]
                for game_id, probability in forest_picks.items()
                if probability is not None and 0 <= probability <= 1 and game_id in logistic_picks
            }
        experiments = [(spec, model_probabilities[spec.key]) for spec in SHADOW_EXPERIMENTS if spec.key in model_probabilities]
        reports = {}
        if injuries is not None:
            day_reports = injuries.loc[
                (injuries["game_date"] == game_date) & injuries["latest_report_datetime"].notna()
            ]
            reports = day_reports.set_index("team").to_dict("index")
        for matchup in slate:
            game_id = matchup["game_id"]
            probability = probabilities[game_id]
            logistic_probability = logistic_picks[game_id]
            if not (0 <= probability <= 1 and 0 <= logistic_probability <= 1):
                raise ValueError(f"Invalid home win probability for ESPN game {game_id}.")
            values = {
                **dict.fromkeys(PICK_VALUES),
                "home_win_probability": round(probability, 6),
                BASELINE_PROBABILITY_COLUMN: round(probability, 6),
                LOGISTIC_PROBABILITY_COLUMN: round(logistic_probability, 6),
                "model": DEFAULT_PREDICTION_MODEL,
                "same_day_finals": len(same_day_finals),
                **dict(zip(REST_DAY_COLUMNS, rest_days[game_id])),
            }
            impact_probability = impact_picks.get(game_id)
            if impact_probability is not None and 0 <= impact_probability <= 1:
                values["home_win_probability"] = round(impact_probability, 6)
                values["model"] = PLAYER_IMPACT_MODEL
                values[PLAYER_IMPACT_COLUMN] = round(100 * (impact_probability - probability), 2)
            for spec, picks in experiments:
                experiment_probability = picks.get(game_id)
                if experiment_probability is not None and 0 <= experiment_probability <= 1:
                    values[spec.probability_column] = round(experiment_probability, 6)
                    values[spec.version_column] = spec.version
            for side in ("home", "away"):
                report = reports.get(matchup[f"{side}_team"])
                if report is not None:
                    stamp = report["latest_report_datetime"].tz_localize(
                        NBA_TIMEZONE, ambiguous=False, nonexistent="shift_forward"
                    )
                    values[f"{side}_injury_report_utc"] = stamp.tz_convert("UTC")
                    values[f"{side}_out_count"] = report["out_count"]
            predictions[game_id] = values
    return predictions


def _same_pick(previous: dict, values: dict) -> bool:
    return all(
        (pd.isna(previous[column]) and pd.isna(value))
        or (pd.notna(previous[column]) and pd.notna(value) and previous[column] == value)
        for column, value in values.items()
    )


def refresh_daily_predictions(path: Path = DAILY_PREDICTIONS_FILE) -> tuple[pd.DataFrame, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(".lock"), timeout=120):
        existing = read_daily_predictions(path)
        records = {row["game_id"]: row for row in existing.to_dict("records")}
        slate, events = _fetch_slate(_utc_now())
        as_of = _utc_now()
        finals = _final_games(events, slate, as_of)
        final_matchups = set(finals[MATCHUP_KEY].itertuples(index=False, name=None))

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
                    game.update({column: previous[column] for column in PICK_COLUMNS})
            records[game["game_id"]] = game

        for game in records.values():
            matchup = tuple(game[column] for column in MATCHUP_KEY)
            if matchup in final_matchups:
                game["status"] = "final"
            if (
                game["status"] == "scheduled"
                and pd.notna(game["predicted_at_utc"])
                and not _slate_window_open(game["tipoff_utc"], game["predicted_at_utc"])
            ):
                game.update(dict.fromkeys(PICK_COLUMNS))

        matchups = [
            records[game["game_id"]] for game in slate if _can_predict(records[game["game_id"]], as_of)
        ]
        if matchups:
            predictions = _predict_games(matchups, as_of, finals)
            generated_at = _utc_now()
            for game in matchups:
                # The slate can tip off while the batch is fitting.
                if generated_at >= game["tipoff_utc"]:
                    continue
                values = predictions[game["game_id"]]
                if not _same_pick(game, values):
                    game.update(values)
                    game["predicted_at_utc"] = generated_at.replace(microsecond=0)

        updated = _typed_predictions(pd.DataFrame(records.values(), columns=PREDICTION_COLUMNS))
        updated = updated.sort_values(["tipoff_utc", "game_id"]).reset_index(drop=True)
        if path.exists() and updated.equals(existing):
            return existing, False
        write_snapshot(updated, path)
        return updated, True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregame-window-open", action="store_true")
    if parser.parse_args().pregame_window_open:
        print("true" if pregame_window_open() else "false")
    else:
        snapshot, changed = refresh_daily_predictions()
        print(f"{'Saved' if changed else 'Verified'} {len(snapshot):,} scheduled NBA games and "
              f"{snapshot['home_win_probability'].count():,} pregame predictions.")
