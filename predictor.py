"""Home-win prediction helpers for the Flask app."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from math import exp
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cleaner import TEAM_MAP_BREF
from model import run_elo, win_prob


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
REFERENCE_DIR = BASE_DIR / "data" / "reference"
OFFICIAL_INJURY_FIRST_SEASON = 2022
INJURY_REPORT_HOURS_ET = tuple(range(23, 10, -1))
LONG_TERM_REASON_KEYWORDS = (
    "acl",
    "achilles",
    "fracture",
    "surgery",
    "tear",
    "torn",
)

BASE_NUMERIC_FEATURES = [
    "season",
    "home_games_played",
    "away_games_played",
    "home_win_pct",
    "away_win_pct",
    "win_pct_diff",
    "home_home_win_pct",
    "away_away_win_pct",
    "venue_win_pct_diff",
    "home_avg_pts_for",
    "away_avg_pts_for",
    "offense_diff",
    "home_avg_pts_against",
    "away_avg_pts_against",
    "defense_diff",
    "home_avg_margin",
    "away_avg_margin",
    "margin_diff",
    "home_team_strength",
    "away_team_strength",
    "team_strength_diff",
    "home_recent_win_pct",
    "away_recent_win_pct",
    "recent_win_pct_diff",
    "home_recent_margin",
    "away_recent_margin",
    "recent_margin_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
]

ELO_NUMERIC_FEATURES = [
    "home_elo_rating",
    "away_elo_rating",
    "elo_diff",
    "elo_home_win_prob",
    "home_blended_strength",
    "away_blended_strength",
    "blended_strength_diff",
]

INJURY_NUMERIC_FEATURES = [
    "home_out_count",
    "away_out_count",
    "home_doubtful_count",
    "away_doubtful_count",
    "home_questionable_count",
    "away_questionable_count",
    "home_probable_count",
    "away_probable_count",
    "home_weighted_injury_score",
    "away_weighted_injury_score",
    "home_reported_player_count",
    "away_reported_player_count",
    "home_estimated_absence_days_total",
    "away_estimated_absence_days_total",
    "home_estimated_absence_days_max",
    "away_estimated_absence_days_max",
    "home_injury_impact_score",
    "away_injury_impact_score",
    "home_long_term_absence_count",
    "away_long_term_absence_count",
    "home_injury_report_available",
    "away_injury_report_available",
    "out_count_diff",
    "doubtful_count_diff",
    "questionable_count_diff",
    "probable_count_diff",
    "weighted_injury_diff",
    "estimated_absence_days_diff",
    "max_absence_days_diff",
    "injury_impact_diff",
    "long_term_absence_diff",
]

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ELO_NUMERIC_FEATURES + INJURY_NUMERIC_FEATURES

CATEGORICAL_FEATURES = ["home_team", "away_team"]
INJURY_OVERRIDE_FIELDS = (
    "out_count",
    "doubtful_count",
    "questionable_count",
    "probable_count",
)


def get_prediction_seasons(games: pd.DataFrame) -> list[int]:
    return sorted(int(season) for season in games["season"].unique())


def get_prediction_teams(games: pd.DataFrame, season: Optional[int] = None) -> list[str]:
    frame = games if season is None else games[games["season"] == season]
    teams = set(frame["home_team"]).union(frame["away_team"])
    return sorted(str(team) for team in teams)


def get_latest_prediction_context(games: pd.DataFrame) -> dict[str, object]:
    if games.empty:
        raise ValueError("Processed games are empty. Scrape data before using the predictor.")

    latest_season = int(games["season"].max())
    latest_season_games = games[games["season"] == latest_season].copy()
    latest_completed_date = pd.to_datetime(latest_season_games["date"]).max().normalize()
    prediction_date = latest_completed_date + pd.Timedelta(days = 1)

    return {
        "season": latest_season,
        "latest_completed_date": latest_completed_date,
        "prediction_date": prediction_date,
        "teams": get_prediction_teams(latest_season_games),
    }


def _injury_csv_candidates() -> list[Path]:
    preferred_paths = [
        *sorted(RAW_DIR.glob("official_nba_injuries_by_team*.csv")),
        *sorted(REFERENCE_DIR.glob("official_nba_injuries_by_team*.csv")),
    ]
    deduped_by_name: dict[str, Path] = {}

    # Prefer freshly scraped raw files over bundled reference files with the same name.
    for path in reversed(preferred_paths):
        deduped_by_name[path.name] = path

    return sorted(deduped_by_name.values())


def get_cached_injury_dataset_path() -> Optional[Path]:
    candidates = _injury_csv_candidates()
    return candidates[-1] if candidates else None


def load_optional_team_injuries(games: pd.DataFrame) -> Optional[pd.DataFrame]:
    if games.empty:
        return None

    injury_paths = _injury_csv_candidates()
    if not injury_paths:
        return None

    frames = [pd.read_csv(path) for path in injury_paths]
    injury_df = pd.concat(frames, ignore_index = True)
    injury_df["game_date"] = pd.to_datetime(injury_df["game_date"]).dt.normalize()
    injury_df["latest_report_datetime"] = pd.to_datetime(injury_df["latest_report_datetime"])
    injury_df = injury_df.sort_values(
        ["game_date", "latest_report_datetime", "team"]
    ).drop_duplicates(
        subset = ["game_date", "team"], keep = "last"
    )
    return injury_df


def _status_count(frame: pd.DataFrame, status: str) -> int:
    return int((frame["Current Status"].astype(str).str.casefold() == status).sum())


def _estimate_absence_days(status: str, reason: str) -> int:
    status_value = str(status).casefold()
    reason_value = str(reason).casefold()
    if status_value == "out":
        return 30 if any(keyword in reason_value for keyword in LONG_TERM_REASON_KEYWORDS) else 10
    if status_value == "doubtful":
        return 7
    if status_value == "questionable":
        return 4
    if status_value == "probable":
        return 2
    return 0


def summarize_injury_report(report_df: pd.DataFrame, report_timestamp: datetime) -> pd.DataFrame:
    if report_df.empty:
        return pd.DataFrame()

    frame = report_df.copy()
    frame["Team"] = frame["Team"].map(TEAM_MAP_BREF).fillna(frame["Team"])
    frame["Current Status"] = frame["Current Status"].fillna("Available")
    frame["Reason"] = frame["Reason"].fillna("")
    frame["estimated_absence_days"] = frame.apply(
        lambda row: _estimate_absence_days(row["Current Status"], row["Reason"]),
        axis = 1,
    )

    rows = []
    for team, team_frame in frame.groupby("Team", sort = True):
        out_count = _status_count(team_frame, "out")
        doubtful_count = _status_count(team_frame, "doubtful")
        questionable_count = _status_count(team_frame, "questionable")
        probable_count = _status_count(team_frame, "probable")
        available_count = _status_count(team_frame, "available")
        weighted_injury_score = (
            out_count * 1.0
            + doubtful_count * 0.75
            + questionable_count * 0.5
            + probable_count * 0.25
        )
        long_term_absence_count = int(
            team_frame[
                (team_frame["Current Status"].astype(str).str.casefold() == "out")
                & team_frame["Reason"].astype(str).str.casefold().str.contains(
                    "|".join(LONG_TERM_REASON_KEYWORDS),
                    regex = True,
                )
            ].shape[0]
        )

        rows.append(
            {
                "game_date": pd.Timestamp(report_timestamp.date()),
                "team": team,
                "latest_report_datetime": pd.Timestamp(report_timestamp),
                "out_count": out_count,
                "doubtful_count": doubtful_count,
                "questionable_count": questionable_count,
                "probable_count": probable_count,
                "available_count": available_count,
                "weighted_injury_score": weighted_injury_score,
                "reported_player_count": int(len(team_frame)),
                "estimated_absence_days_total": float(team_frame["estimated_absence_days"].sum()),
                "estimated_absence_days_max": float(team_frame["estimated_absence_days"].max()),
                "injury_impact_score": weighted_injury_score * 7.0,
                "long_term_absence_count": long_term_absence_count,
            }
        )

    return pd.DataFrame(rows)


def refresh_official_injury_dataset(games: pd.DataFrame) -> dict[str, object]:
    try:
        from nbainjuries import injury
    except ImportError:
        return {
            "status": "missing_dependency",
            "message": "Install the nbainjuries package to refresh official injury data.",
            "rows": 0,
            "file": None,
        }

    if games.empty:
        return {
            "status": "no_games",
            "message": "No predictor games are available yet, so injury refresh was skipped.",
            "rows": 0,
            "file": None,
        }

    eligible_games = games[games["season"] >= OFFICIAL_INJURY_FIRST_SEASON].copy()
    if eligible_games.empty:
        return {
            "status": "no_overlap",
            "message": f"Official NBA injury reports begin with season {OFFICIAL_INJURY_FIRST_SEASON}.",
            "rows": 0,
            "file": None,
        }

    game_dates = sorted(pd.to_datetime(eligible_games["date"]).dt.normalize().unique())
    report_rows = []
    missing_dates = []

    for game_date in game_dates:
        report_timestamp = None
        report_df = None
        base_date = pd.Timestamp(game_date).to_pydatetime()

        # Walk backward through likely end-of-day report times to find the latest valid report.
        for hour in INJURY_REPORT_HOURS_ET:
            candidate = datetime.combine(base_date.date(), datetime.min.time()).replace(
                hour = hour,
                minute = 30,
            )
            try:
                if injury.check_reportvalid(candidate):
                    report_timestamp = candidate
                    report_df = injury.get_reportdata(candidate, return_df = True)
                    break
            except Exception:
                continue

        if report_timestamp is None or report_df is None or report_df.empty:
            missing_dates.append(str(pd.Timestamp(game_date).date()))
            continue

        report_rows.append(summarize_injury_report(report_df, report_timestamp))

    if not report_rows:
        cached_path = get_cached_injury_dataset_path()
        if cached_path is not None:
            return {
                "status": "cached",
                "message": f"Using cached injury dataset from {cached_path.name}.",
                "rows": 0,
                "file": str(cached_path),
                "missing_dates": missing_dates,
            }

        return {
            "status": "no_reports",
            "message": "No valid official NBA injury reports were found for the predictor range.",
            "rows": 0,
            "file": None,
            "missing_dates": missing_dates,
        }

    injury_df = pd.concat(report_rows, ignore_index = True)
    start_season = OFFICIAL_INJURY_FIRST_SEASON
    end_season = int(eligible_games["season"].max())
    output_path = RAW_DIR / f"official_nba_injuries_by_team_{start_season}_{end_season}.csv"
    injury_df.to_csv(output_path, index = False)

    return {
        "status": "refreshed",
        "message": (
            f"Official injury data refreshed for seasons {start_season}-{end_season} "
            f"across {injury_df['game_date'].nunique()} game dates."
        ),
        "rows": int(len(injury_df)),
        "file": str(output_path),
        "missing_dates": missing_dates,
    }


def get_numeric_features(include_elo_features: bool = True) -> list[str]:
    features = list(BASE_NUMERIC_FEATURES)
    if include_elo_features:
        features.extend(ELO_NUMERIC_FEATURES)
    features.extend(INJURY_NUMERIC_FEATURES)
    return features


def _team_state() -> dict[str, object]:
    return {
        "games": 0,
        "wins": 0,
        "home_games": 0,
        "home_wins": 0,
        "away_games": 0,
        "away_wins": 0,
        "pts_for": 0,
        "pts_against": 0,
        "margin_sum": 0,
        "recent_wins": deque(maxlen = 5),
        "recent_margins": deque(maxlen = 5),
        "last_date": None,
    }


def _safe_pct(wins: int, games: int, default: float = 0.5) -> float:
    return wins / games if games else default


def _safe_avg(total: float, count: int, default: float = 0.0) -> float:
    return total / count if count else default


def _recent_avg(values: deque, default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _team_strength_score(win_pct: float, avg_margin: float) -> float:
    # Blend record and scoring margin into one pregame team-strength feature.
    normalized_margin = max(min(avg_margin / 15.0, 1.0), -1.0)
    margin_component = (normalized_margin + 1.0) / 2.0
    return win_pct * 0.7 + margin_component * 0.3


def _normalize_elo_rating(elo_rating: float) -> float:
    return 1.0 / (1.0 + exp(-(elo_rating - 1500.0) / 120.0))


def _blended_team_strength(team_strength: float, elo_rating: float) -> float:
    return team_strength * 0.65 + _normalize_elo_rating(elo_rating) * 0.35


def _update_states_from_game(states: dict[str, dict[str, object]], row) -> None:
    home_state = states[row.home_team]
    away_state = states[row.away_team]

    home_margin = row.home_pts - row.away_pts
    away_margin = -home_margin
    home_win = int(row.home_win)
    away_win = 1 - home_win

    home_state["games"] += 1
    home_state["wins"] += home_win
    home_state["home_games"] += 1
    home_state["home_wins"] += home_win
    home_state["pts_for"] += row.home_pts
    home_state["pts_against"] += row.away_pts
    home_state["margin_sum"] += home_margin
    home_state["recent_wins"].append(home_win)
    home_state["recent_margins"].append(home_margin)
    home_state["last_date"] = row.date

    away_state["games"] += 1
    away_state["wins"] += away_win
    away_state["away_games"] += 1
    away_state["away_wins"] += away_win
    away_state["pts_for"] += row.away_pts
    away_state["pts_against"] += row.home_pts
    away_state["margin_sum"] += away_margin
    away_state["recent_wins"].append(away_win)
    away_state["recent_margins"].append(away_margin)
    away_state["last_date"] = row.date


def _build_base_matchup_features(
    season: int,
    game_date: pd.Timestamp,
    home_team: str,
    away_team: str,
    home_state: dict[str, object],
    away_state: dict[str, object],
) -> dict[str, object]:
    home_rest_days = (
        (game_date - home_state["last_date"]).days - 1
        if home_state["last_date"] is not None
        else 7
    )
    away_rest_days = (
        (game_date - away_state["last_date"]).days - 1
        if away_state["last_date"] is not None
        else 7
    )

    home_win_pct = _safe_pct(home_state["wins"], home_state["games"])
    away_win_pct = _safe_pct(away_state["wins"], away_state["games"])
    home_home_win_pct = _safe_pct(home_state["home_wins"], home_state["home_games"])
    away_away_win_pct = _safe_pct(away_state["away_wins"], away_state["away_games"])
    home_avg_pts_for = _safe_avg(home_state["pts_for"], home_state["games"], default = 110.0)
    away_avg_pts_for = _safe_avg(away_state["pts_for"], away_state["games"], default = 110.0)
    home_avg_pts_against = _safe_avg(
        home_state["pts_against"], home_state["games"], default = 110.0
    )
    away_avg_pts_against = _safe_avg(
        away_state["pts_against"], away_state["games"], default = 110.0
    )
    home_avg_margin = _safe_avg(home_state["margin_sum"], home_state["games"])
    away_avg_margin = _safe_avg(away_state["margin_sum"], away_state["games"])
    home_recent_win_pct = _recent_avg(home_state["recent_wins"], default = 0.5)
    away_recent_win_pct = _recent_avg(away_state["recent_wins"], default = 0.5)
    home_recent_margin = _recent_avg(home_state["recent_margins"], default = 0.0)
    away_recent_margin = _recent_avg(away_state["recent_margins"], default = 0.0)
    home_team_strength = _team_strength_score(home_win_pct, home_avg_margin)
    away_team_strength = _team_strength_score(away_win_pct, away_avg_margin)

    return {
        "date": game_date,
        "season": season,
        "home_team": home_team,
        "away_team": away_team,
        "home_games_played": home_state["games"],
        "away_games_played": away_state["games"],
        "home_win_pct": home_win_pct,
        "away_win_pct": away_win_pct,
        "win_pct_diff": home_win_pct - away_win_pct,
        "home_home_win_pct": home_home_win_pct,
        "away_away_win_pct": away_away_win_pct,
        "venue_win_pct_diff": home_home_win_pct - away_away_win_pct,
        "home_avg_pts_for": home_avg_pts_for,
        "away_avg_pts_for": away_avg_pts_for,
        "offense_diff": home_avg_pts_for - away_avg_pts_for,
        "home_avg_pts_against": home_avg_pts_against,
        "away_avg_pts_against": away_avg_pts_against,
        "defense_diff": away_avg_pts_against - home_avg_pts_against,
        "home_avg_margin": home_avg_margin,
        "away_avg_margin": away_avg_margin,
        "margin_diff": home_avg_margin - away_avg_margin,
        "home_team_strength": home_team_strength,
        "away_team_strength": away_team_strength,
        "team_strength_diff": home_team_strength - away_team_strength,
        "home_elo_rating": 1500.0,
        "away_elo_rating": 1500.0,
        "elo_diff": 0.0,
        "elo_home_win_prob": 0.5,
        "home_blended_strength": home_team_strength,
        "away_blended_strength": away_team_strength,
        "blended_strength_diff": home_team_strength - away_team_strength,
        "home_recent_win_pct": home_recent_win_pct,
        "away_recent_win_pct": away_recent_win_pct,
        "recent_win_pct_diff": home_recent_win_pct - away_recent_win_pct,
        "home_recent_margin": home_recent_margin,
        "away_recent_margin": away_recent_margin,
        "recent_margin_diff": home_recent_margin - away_recent_margin,
        "home_rest_days": max(home_rest_days, 0),
        "away_rest_days": max(away_rest_days, 0),
        "rest_diff": max(home_rest_days, 0) - max(away_rest_days, 0),
    }


def build_pregame_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop = True).copy()
    season_states = defaultdict(lambda: defaultdict(_team_state))
    feature_rows = []

    # Rebuild each game's context using only information available before tipoff.
    for row in df.itertuples(index = False):
        season = row.season
        home_state = season_states[season][row.home_team]
        away_state = season_states[season][row.away_team]

        feature_row = _build_base_matchup_features(
            season = season,
            game_date = row.date,
            home_team = row.home_team,
            away_team = row.away_team,
            home_state = home_state,
            away_state = away_state,
        )
        feature_row["home_win"] = row.home_win
        feature_rows.append(feature_row)

        _update_states_from_game(season_states[season], row)

    feature_df = pd.DataFrame(feature_rows)
    elo_results, _ = run_elo(df)

    feature_df["home_elo_rating"] = elo_results["r_home_pre"].astype(float).values
    feature_df["away_elo_rating"] = elo_results["r_away_pre"].astype(float).values
    feature_df["elo_diff"] = feature_df["home_elo_rating"] - feature_df["away_elo_rating"]
    feature_df["elo_home_win_prob"] = elo_results["p_home_win"].astype(float).values
    feature_df["home_blended_strength"] = feature_df.apply(
        lambda row: _blended_team_strength(row["home_team_strength"], row["home_elo_rating"]),
        axis = 1,
    )
    feature_df["away_blended_strength"] = feature_df.apply(
        lambda row: _blended_team_strength(row["away_team_strength"], row["away_elo_rating"]),
        axis = 1,
    )
    feature_df["blended_strength_diff"] = (
        feature_df["home_blended_strength"] - feature_df["away_blended_strength"]
    )
    return feature_df


def merge_injury_features(feature_df: pd.DataFrame, injury_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    result = feature_df.copy()

    default_columns = {
        "home_out_count": 0.0,
        "away_out_count": 0.0,
        "home_doubtful_count": 0.0,
        "away_doubtful_count": 0.0,
        "home_questionable_count": 0.0,
        "away_questionable_count": 0.0,
        "home_probable_count": 0.0,
        "away_probable_count": 0.0,
        "home_available_count": 0.0,
        "away_available_count": 0.0,
        "home_weighted_injury_score": 0.0,
        "away_weighted_injury_score": 0.0,
        "home_reported_player_count": 0.0,
        "away_reported_player_count": 0.0,
        "home_estimated_absence_days_total": 0.0,
        "away_estimated_absence_days_total": 0.0,
        "home_estimated_absence_days_max": 0.0,
        "away_estimated_absence_days_max": 0.0,
        "home_injury_impact_score": 0.0,
        "away_injury_impact_score": 0.0,
        "home_long_term_absence_count": 0.0,
        "away_long_term_absence_count": 0.0,
        "home_injury_report_available": 0,
        "away_injury_report_available": 0,
        "out_count_diff": 0.0,
        "doubtful_count_diff": 0.0,
        "questionable_count_diff": 0.0,
        "probable_count_diff": 0.0,
        "weighted_injury_diff": 0.0,
        "estimated_absence_days_diff": 0.0,
        "max_absence_days_diff": 0.0,
        "injury_impact_diff": 0.0,
        "long_term_absence_diff": 0.0,
    }

    for column, value in default_columns.items():
        result[column] = value

    if injury_df is None or injury_df.empty:
        return result

    home_injuries = injury_df.rename(
        columns = {
            "game_date": "date",
            "team": "home_team",
            "latest_report_datetime": "home_latest_report_datetime",
            "out_count": "home_out_count",
            "doubtful_count": "home_doubtful_count",
            "questionable_count": "home_questionable_count",
            "probable_count": "home_probable_count",
            "available_count": "home_available_count",
            "weighted_injury_score": "home_weighted_injury_score",
            "reported_player_count": "home_reported_player_count",
            "estimated_absence_days_total": "home_estimated_absence_days_total",
            "estimated_absence_days_max": "home_estimated_absence_days_max",
            "injury_impact_score": "home_injury_impact_score",
            "long_term_absence_count": "home_long_term_absence_count",
        }
    )
    away_injuries = injury_df.rename(
        columns = {
            "game_date": "date",
            "team": "away_team",
            "latest_report_datetime": "away_latest_report_datetime",
            "out_count": "away_out_count",
            "doubtful_count": "away_doubtful_count",
            "questionable_count": "away_questionable_count",
            "probable_count": "away_probable_count",
            "available_count": "away_available_count",
            "weighted_injury_score": "away_weighted_injury_score",
            "reported_player_count": "away_reported_player_count",
            "estimated_absence_days_total": "away_estimated_absence_days_total",
            "estimated_absence_days_max": "away_estimated_absence_days_max",
            "injury_impact_score": "away_injury_impact_score",
            "long_term_absence_count": "away_long_term_absence_count",
        }
    )

    result = result.drop(columns = list(default_columns.keys()))
    result = result.merge(home_injuries, on = ["date", "home_team"], how = "left")
    result = result.merge(away_injuries, on = ["date", "away_team"], how = "left")

    fill_zero_columns = [
        "home_out_count",
        "away_out_count",
        "home_doubtful_count",
        "away_doubtful_count",
        "home_questionable_count",
        "away_questionable_count",
        "home_probable_count",
        "away_probable_count",
        "home_available_count",
        "away_available_count",
        "home_weighted_injury_score",
        "away_weighted_injury_score",
        "home_reported_player_count",
        "away_reported_player_count",
        "home_estimated_absence_days_total",
        "away_estimated_absence_days_total",
        "home_estimated_absence_days_max",
        "away_estimated_absence_days_max",
        "home_injury_impact_score",
        "away_injury_impact_score",
        "home_long_term_absence_count",
        "away_long_term_absence_count",
    ]
    for column in fill_zero_columns:
        result[column] = result[column].fillna(0.0)

    result["home_injury_report_available"] = result["home_latest_report_datetime"].notna().astype(int)
    result["away_injury_report_available"] = result["away_latest_report_datetime"].notna().astype(int)
    result["out_count_diff"] = result["away_out_count"] - result["home_out_count"]
    result["doubtful_count_diff"] = result["away_doubtful_count"] - result["home_doubtful_count"]
    result["questionable_count_diff"] = (
        result["away_questionable_count"] - result["home_questionable_count"]
    )
    result["probable_count_diff"] = result["away_probable_count"] - result["home_probable_count"]
    result["weighted_injury_diff"] = (
        result["away_weighted_injury_score"] - result["home_weighted_injury_score"]
    )
    result["estimated_absence_days_diff"] = (
        result["away_estimated_absence_days_total"] - result["home_estimated_absence_days_total"]
    )
    result["max_absence_days_diff"] = (
        result["away_estimated_absence_days_max"] - result["home_estimated_absence_days_max"]
    )
    result["injury_impact_diff"] = (
        result["away_injury_impact_score"] - result["home_injury_impact_score"]
    )
    result["long_term_absence_diff"] = (
        result["away_long_term_absence_count"] - result["home_long_term_absence_count"]
    )
    return result


def _apply_injury_row(
    frame: pd.DataFrame,
    side: str,
    injury_row: Optional[pd.Series],
) -> None:
    if injury_row is None:
        return

    for column in (
        "out_count",
        "doubtful_count",
        "questionable_count",
        "probable_count",
        "available_count",
        "weighted_injury_score",
        "reported_player_count",
        "estimated_absence_days_total",
        "estimated_absence_days_max",
        "injury_impact_score",
        "long_term_absence_count",
    ):
        frame.at[0, f"{side}_{column}"] = float(injury_row.get(column, 0.0) or 0.0)

    frame.at[0, f"{side}_injury_report_available"] = 1


def merge_latest_injury_snapshot(
    matchup_df: pd.DataFrame,
    injury_df: Optional[pd.DataFrame],
    game_date: pd.Timestamp,
) -> pd.DataFrame:
    frame = merge_injury_features(matchup_df, injury_df = None)
    if injury_df is None or injury_df.empty:
        return frame

    normalized_game_date = game_date.normalize()

    for side in ("home", "away"):
        team = frame.at[0, f"{side}_team"]
        team_rows = injury_df[
            (injury_df["team"] == team) & (injury_df["game_date"] <= normalized_game_date)
        ].sort_values(["game_date", "latest_report_datetime"])
        latest_row = team_rows.iloc[-1] if not team_rows.empty else None
        _apply_injury_row(frame, side, latest_row)

    frame["out_count_diff"] = frame["away_out_count"] - frame["home_out_count"]
    frame["doubtful_count_diff"] = frame["away_doubtful_count"] - frame["home_doubtful_count"]
    frame["questionable_count_diff"] = (
        frame["away_questionable_count"] - frame["home_questionable_count"]
    )
    frame["probable_count_diff"] = frame["away_probable_count"] - frame["home_probable_count"]
    frame["weighted_injury_diff"] = (
        frame["away_weighted_injury_score"] - frame["home_weighted_injury_score"]
    )
    frame["estimated_absence_days_diff"] = (
        frame["away_estimated_absence_days_total"] - frame["home_estimated_absence_days_total"]
    )
    frame["max_absence_days_diff"] = (
        frame["away_estimated_absence_days_max"] - frame["home_estimated_absence_days_max"]
    )
    frame["injury_impact_diff"] = (
        frame["away_injury_impact_score"] - frame["home_injury_impact_score"]
    )
    frame["long_term_absence_diff"] = (
        frame["away_long_term_absence_count"] - frame["home_long_term_absence_count"]
    )
    return frame


def _build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    categorical_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown = "ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers = [
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numeric_transformer, numeric_features),
        ]
    )


def train_prediction_models(
    games: pd.DataFrame,
    injury_df: Optional[pd.DataFrame] = None,
    logistic_include_elo_features: bool = True,
    random_forest_include_elo_features: bool = True,
) -> tuple[Pipeline, Pipeline, pd.DataFrame]:
    if games.empty:
        raise ValueError("At least one completed game is required to train the predictor.")

    feature_df = merge_injury_features(build_pregame_features(games), injury_df)
    if feature_df.empty:
        raise ValueError("Feature engineering produced no training rows.")

    y = feature_df["home_win"]
    logistic_numeric_features = get_numeric_features(
        include_elo_features = logistic_include_elo_features
    )
    random_forest_numeric_features = get_numeric_features(
        include_elo_features = random_forest_include_elo_features
    )
    logistic_columns = CATEGORICAL_FEATURES + logistic_numeric_features
    random_forest_columns = CATEGORICAL_FEATURES + random_forest_numeric_features

    logistic_regression = Pipeline(
        steps = [
            ("preprocessor", _build_preprocessor(logistic_numeric_features)),
            ("classifier", LogisticRegression(max_iter = 2000, random_state = 100)),
        ]
    )
    random_forest = Pipeline(
        steps = [
            ("preprocessor", _build_preprocessor(random_forest_numeric_features)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators = 300,
                    min_samples_leaf = 3,
                    random_state = 100,
                    n_jobs = -1,
                ),
            ),
        ]
    )

    logistic_regression.fit(feature_df[logistic_columns], y)
    random_forest.fit(feature_df[random_forest_columns], y)
    return logistic_regression, random_forest, feature_df


def evaluate_random_forest_elo_impact(
    games: pd.DataFrame,
    injury_df: Optional[pd.DataFrame] = None,
) -> dict[str, float | int | str]:
    if games.empty:
        raise ValueError("At least one completed game is required to evaluate the predictor.")

    feature_df = merge_injury_features(build_pregame_features(games), injury_df)
    feature_df = feature_df.sort_values("date").reset_index(drop = True)
    if len(feature_df) < 10:
        raise ValueError("Not enough completed games are available for a meaningful comparison.")

    latest_season = int(feature_df["season"].max())
    if feature_df["season"].nunique() > 1:
        train_df = feature_df[feature_df["season"] < latest_season].copy()
        test_df = feature_df[feature_df["season"] == latest_season].copy()
        split_label = f"train_before_{latest_season}_test_{latest_season}"
    else:
        split_index = max(int(len(feature_df) * 0.8), 1)
        train_df = feature_df.iloc[:split_index].copy()
        test_df = feature_df.iloc[split_index:].copy()
        split_label = "chronological_80_20"

    if train_df.empty or test_df.empty:
        raise ValueError("Evaluation split produced an empty train or test set.")

    y_train = train_df["home_win"]
    y_test = test_df["home_win"]
    base_numeric_features = get_numeric_features(include_elo_features = False)
    elo_numeric_features = get_numeric_features(include_elo_features = True)
    base_columns = CATEGORICAL_FEATURES + base_numeric_features
    elo_columns = CATEGORICAL_FEATURES + elo_numeric_features

    base_random_forest = Pipeline(
        steps = [
            ("preprocessor", _build_preprocessor(base_numeric_features)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators = 300,
                    min_samples_leaf = 3,
                    random_state = 100,
                    n_jobs = -1,
                ),
            ),
        ]
    )
    elo_random_forest = Pipeline(
        steps = [
            ("preprocessor", _build_preprocessor(elo_numeric_features)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators = 300,
                    min_samples_leaf = 3,
                    random_state = 100,
                    n_jobs = -1,
                ),
            ),
        ]
    )

    base_random_forest.fit(train_df[base_columns], y_train)
    elo_random_forest.fit(train_df[elo_columns], y_train)
    base_predictions = base_random_forest.predict(test_df[base_columns])
    elo_predictions = elo_random_forest.predict(test_df[elo_columns])
    base_accuracy = accuracy_score(y_test, base_predictions)
    elo_accuracy = accuracy_score(y_test, elo_predictions)

    return {
        "split": split_label,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "without_elo_accuracy": float(base_accuracy),
        "with_elo_accuracy": float(elo_accuracy),
        "accuracy_gain": float(elo_accuracy - base_accuracy),
    }


def _compute_state_snapshot(
    games: pd.DataFrame,
    season: int,
    game_date: pd.Timestamp,
    home_team: str,
    away_team: str,
) -> dict[str, object]:
    season_games = games[
        (games["season"] == season) & (pd.to_datetime(games["date"]) < game_date)
    ].sort_values("date")

    states = defaultdict(_team_state)
    for row in season_games.itertuples(index = False):
        _update_states_from_game(states, row)

    return _build_base_matchup_features(
        season = season,
        game_date = game_date,
        home_team = home_team,
        away_team = away_team,
        home_state = states[home_team],
        away_state = states[away_team],
    )


def _compute_elo_snapshot(games: pd.DataFrame, home_team: str, away_team: str) -> dict[str, float]:
    _, ratings = run_elo(games)
    home_elo_rating = float(ratings.get(home_team, 1500.0))
    away_elo_rating = float(ratings.get(away_team, 1500.0))
    home_team_strength = _normalize_elo_rating(home_elo_rating)
    away_team_strength = _normalize_elo_rating(away_elo_rating)

    return {
        "home_elo_rating": home_elo_rating,
        "away_elo_rating": away_elo_rating,
        "elo_diff": home_elo_rating - away_elo_rating,
        "elo_home_win_prob": float(win_prob(home_elo_rating, away_elo_rating)),
        "home_elo_strength": home_team_strength,
        "away_elo_strength": away_team_strength,
    }


def _apply_manual_injury_overrides(
    matchup_df: pd.DataFrame,
    overrides: dict[str, float],
) -> pd.DataFrame:
    frame = matchup_df.copy()

    # Convert simple front-end counts into the same injury feature family used in training.
    for side in ("home", "away"):
        out_count = float(overrides.get(f"{side}_out_count", frame.at[0, f"{side}_out_count"]))
        doubtful_count = float(
            overrides.get(f"{side}_doubtful_count", frame.at[0, f"{side}_doubtful_count"])
        )
        questionable_count = float(
            overrides.get(f"{side}_questionable_count", frame.at[0, f"{side}_questionable_count"])
        )
        probable_count = float(
            overrides.get(f"{side}_probable_count", frame.at[0, f"{side}_probable_count"])
        )

        frame.at[0, f"{side}_out_count"] = out_count
        frame.at[0, f"{side}_doubtful_count"] = doubtful_count
        frame.at[0, f"{side}_questionable_count"] = questionable_count
        frame.at[0, f"{side}_probable_count"] = probable_count
        frame.at[0, f"{side}_reported_player_count"] = (
            out_count + doubtful_count + questionable_count + probable_count
        )
        frame.at[0, f"{side}_weighted_injury_score"] = (
            out_count * 1.0
            + doubtful_count * 0.75
            + questionable_count * 0.5
            + probable_count * 0.25
        )
        frame.at[0, f"{side}_estimated_absence_days_total"] = (
            out_count * 10.0 + doubtful_count * 7.0 + questionable_count * 4.0 + probable_count * 2.0
        )
        frame.at[0, f"{side}_estimated_absence_days_max"] = (
            10.0 if out_count > 0 else 7.0 if doubtful_count > 0 else 4.0 if questionable_count > 0 else 2.0 if probable_count > 0 else 0.0
        )
        frame.at[0, f"{side}_injury_impact_score"] = (
            frame.at[0, f"{side}_weighted_injury_score"] * 7.0
        )
        frame.at[0, f"{side}_long_term_absence_count"] = out_count
        frame.at[0, f"{side}_injury_report_available"] = int(
            frame.at[0, f"{side}_reported_player_count"] > 0
        )

    frame["out_count_diff"] = frame["away_out_count"] - frame["home_out_count"]
    frame["doubtful_count_diff"] = frame["away_doubtful_count"] - frame["home_doubtful_count"]
    frame["questionable_count_diff"] = (
        frame["away_questionable_count"] - frame["home_questionable_count"]
    )
    frame["probable_count_diff"] = frame["away_probable_count"] - frame["home_probable_count"]
    frame["weighted_injury_diff"] = (
        frame["away_weighted_injury_score"] - frame["home_weighted_injury_score"]
    )
    frame["estimated_absence_days_diff"] = (
        frame["away_estimated_absence_days_total"] - frame["home_estimated_absence_days_total"]
    )
    frame["max_absence_days_diff"] = (
        frame["away_estimated_absence_days_max"] - frame["home_estimated_absence_days_max"]
    )
    frame["injury_impact_diff"] = (
        frame["away_injury_impact_score"] - frame["home_injury_impact_score"]
    )
    frame["long_term_absence_diff"] = (
        frame["away_long_term_absence_count"] - frame["home_long_term_absence_count"]
    )

    return frame


def build_matchup_features(
    games: pd.DataFrame,
    season: int,
    game_date: pd.Timestamp,
    home_team: str,
    away_team: str,
    injury_df: Optional[pd.DataFrame] = None,
    injury_overrides: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    snapshot = _compute_state_snapshot(games, season, game_date, home_team, away_team)
    matchup_df = pd.DataFrame([snapshot])
    elo_snapshot = _compute_elo_snapshot(games, home_team, away_team)
    matchup_df["home_elo_rating"] = elo_snapshot["home_elo_rating"]
    matchup_df["away_elo_rating"] = elo_snapshot["away_elo_rating"]
    matchup_df["elo_diff"] = elo_snapshot["elo_diff"]
    matchup_df["elo_home_win_prob"] = elo_snapshot["elo_home_win_prob"]
    matchup_df["home_blended_strength"] = matchup_df.apply(
        lambda row: _blended_team_strength(row["home_team_strength"], row["home_elo_rating"]),
        axis = 1,
    )
    matchup_df["away_blended_strength"] = matchup_df.apply(
        lambda row: _blended_team_strength(row["away_team_strength"], row["away_elo_rating"]),
        axis = 1,
    )
    matchup_df["blended_strength_diff"] = (
        matchup_df["home_blended_strength"] - matchup_df["away_blended_strength"]
    )
    matchup_df = merge_latest_injury_snapshot(matchup_df, injury_df, game_date)

    if injury_overrides:
        matchup_df = _apply_manual_injury_overrides(matchup_df, injury_overrides)

    return matchup_df


def predict_matchup_from_games(
    games: pd.DataFrame,
    home_team: str,
    away_team: str,
    season: int,
    game_date: pd.Timestamp,
    injury_overrides: Optional[dict[str, float]] = None,
) -> dict[str, object]:
    if home_team == away_team:
        raise ValueError("Home team and away team must be different.")

    if season not in set(int(value) for value in games["season"].unique()):
        raise ValueError(f"Season {season} is not present in the processed games dataset.")

    cutoff_games = games[pd.to_datetime(games["date"]) < game_date].copy()
    if cutoff_games.empty:
        raise ValueError("No completed games exist before the requested game date.")

    injury_df = load_optional_team_injuries(games)
    if injury_df is not None:
        injury_df = injury_df[injury_df["game_date"] <= game_date.normalize()].copy()

    # Train on completed games only, then score the requested future matchup snapshot.
    logistic_regression, random_forest, _ = train_prediction_models(cutoff_games, injury_df)
    matchup_df = build_matchup_features(
        games = cutoff_games,
        season = season,
        game_date = game_date,
        home_team = home_team,
        away_team = away_team,
        injury_df = injury_df,
        injury_overrides = injury_overrides,
    )
    logistic_input = matchup_df[list(logistic_regression.feature_names_in_)]
    random_forest_input = matchup_df[list(random_forest.feature_names_in_)]

    lr_home_prob = float(logistic_regression.predict_proba(logistic_input)[0][1])
    rf_home_prob = float(random_forest.predict_proba(random_forest_input)[0][1])
    lr_home_pred = int(logistic_regression.predict(logistic_input)[0])
    rf_home_pred = int(random_forest.predict(random_forest_input)[0])
    elo_home_prob = float(matchup_df.at[0, "elo_home_win_prob"])
    elo_home_pred = int(elo_home_prob >= 0.5)

    return {
        "matchup": f"{away_team} @ {home_team}",
        "season": season,
        "game_date": game_date.strftime("%Y-%m-%d"),
        "home_team": home_team,
        "away_team": away_team,
        "elo": {
            "home_win_probability": elo_home_prob,
            "home_win_prediction": elo_home_pred,
            "predicted_winner": home_team if elo_home_pred == 1 else away_team,
        },
        "logistic": {
            "home_win_probability": lr_home_prob,
            "home_win_prediction": lr_home_pred,
            "predicted_winner": home_team if lr_home_pred == 1 else away_team,
        },
        "random_forest": {
            "home_win_probability": rf_home_prob,
            "home_win_prediction": rf_home_pred,
            "predicted_winner": home_team if rf_home_pred == 1 else away_team,
            "feature_set": "elo_enhanced",
        },
        "context": {
            "home_win_pct": float(matchup_df.at[0, "home_win_pct"]),
            "away_win_pct": float(matchup_df.at[0, "away_win_pct"]),
            "home_team_strength": float(matchup_df.at[0, "home_team_strength"]),
            "away_team_strength": float(matchup_df.at[0, "away_team_strength"]),
            "team_strength_diff": float(matchup_df.at[0, "team_strength_diff"]),
            "home_elo_rating": float(matchup_df.at[0, "home_elo_rating"]),
            "away_elo_rating": float(matchup_df.at[0, "away_elo_rating"]),
            "elo_diff": float(matchup_df.at[0, "elo_diff"]),
            "home_blended_strength": float(matchup_df.at[0, "home_blended_strength"]),
            "away_blended_strength": float(matchup_df.at[0, "away_blended_strength"]),
            "blended_strength_diff": float(matchup_df.at[0, "blended_strength_diff"]),
            "home_rest_days": int(matchup_df.at[0, "home_rest_days"]),
            "away_rest_days": int(matchup_df.at[0, "away_rest_days"]),
            "home_out_count": float(matchup_df.at[0, "home_out_count"]),
            "away_out_count": float(matchup_df.at[0, "away_out_count"]),
            "home_injury_impact_score": float(matchup_df.at[0, "home_injury_impact_score"]),
            "away_injury_impact_score": float(matchup_df.at[0, "away_injury_impact_score"]),
        },
        "injury_data_used": injury_df is not None,
    }
