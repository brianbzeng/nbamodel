"""Home-win prediction helpers for the Flask app."""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cleaner import TEAM_MAP_BREF


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
REFERENCE_DIR = BASE_DIR / "data" / "reference"
OFFICIAL_INJURY_FIRST_SEASON = 2022
REPORT_HOURS = [17, 14, 13, 12, 11, 10, 9]
MAX_REPORT_FETCH_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2
MAX_REST_DAYS = 7
MAX_INJURY_LOOKBACK_DAYS = 3
LONG_TERM_REASON_KEYWORDS = (
    "acl",
    "achilles",
    "fracture",
    "surgery",
    "tear",
    "torn",
)
STATUS_IMPACT_MULTIPLIERS = {
    "OUT": 1.0,
    "DOUBTFUL": 0.75,
    "QUESTIONABLE": 0.45,
    "PROBABLE": 0.15,
    "AVAILABLE": 0.0,
}

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

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + INJURY_NUMERIC_FEATURES

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


def get_exact_injury_dataset_path(games: pd.DataFrame) -> Optional[Path]:
    if games.empty:
        return None

    start_season = int(games["season"].min())
    end_season = int(games["season"].max())
    exact_path = RAW_DIR / f"official_nba_injuries_by_team_{start_season}_{end_season}.csv"
    return exact_path if exact_path.exists() else None


def load_optional_team_injuries(games: pd.DataFrame) -> Optional[pd.DataFrame]:
    if games.empty:
        return None

    exact_path = get_exact_injury_dataset_path(games)
    injury_paths = [exact_path] if exact_path is not None else _injury_csv_candidates()
    if not injury_paths:
        return None

    frames = [pd.read_csv(path) for path in injury_paths]
    injury_df = pd.concat(frames, ignore_index = True)
    injury_df["game_date"] = pd.to_datetime(injury_df["game_date"]).dt.normalize()
    injury_df["latest_report_datetime"] = pd.to_datetime(injury_df["latest_report_datetime"])
    injury_df["team"] = injury_df["team"].map(TEAM_MAP_BREF).fillna(injury_df["team"])
    injury_df = injury_df.sort_values(
        ["game_date", "latest_report_datetime", "team"]
    ).drop_duplicates(
        subset = ["game_date", "team"], keep = "last"
    )
    return injury_df


def estimate_absence_days(reason: str) -> float:
    # Map common injury descriptions to rough expected absence lengths.
    reason_text = str(reason).strip().lower()
    if not reason_text or reason_text == "nan":
        return 5.0

    specific_patterns = [
        (("achilles", "repair"), 180.0),
        (("acl",), 240.0),
        (("reconstruction", "surgery"), 120.0),
        (("surgery",), 45.0),
        (("meniscus", "tear"), 30.0),
        (("tendon", "strain"), 21.0),
        (("stress", "fracture"), 45.0),
        (("fracture",), 25.0),
        (("hamstring", "strain"), 14.0),
        (("groin", "strain"), 10.0),
        (("ankle", "sprain"), 10.0),
        (("knee", "sprain"), 12.0),
        (("calf", "strain"), 10.0),
        (("shoulder", "strain"), 10.0),
        (("thumb", "sprain"), 8.0),
        (("injury", "recovery"), 7.0),
        (("injury", "management"), 4.0),
        (("concussion",), 6.0),
        (("illness",), 3.0),
    ]
    for keywords, days in specific_patterns:
        if all(keyword in reason_text for keyword in keywords):
            return days

    fallback_days = {
        "tear": 30.0,
        "surgery": 45.0,
        "repair": 120.0,
        "fracture": 25.0,
        "sprain": 10.0,
        "strain": 11.0,
        "soreness": 3.0,
        "tightness": 3.0,
        "discomfort": 3.0,
        "inflammation": 5.0,
        "contusion": 4.0,
        "management": 4.0,
        "reconditioning": 7.0,
        "recovery": 7.0,
        "illness": 3.0,
        "ankle": 8.0,
        "knee": 10.0,
        "hamstring": 11.0,
        "groin": 9.0,
        "calf": 9.0,
        "foot": 10.0,
        "back": 6.0,
        "hip": 10.0,
        "shoulder": 9.0,
        "elbow": 8.0,
        "wrist": 8.0,
        "thumb": 7.0,
        "finger": 6.0,
        "achilles": 90.0,
        "not with team": 5.0,
        "personal reasons": 3.0,
    }
    for keyword, days in fallback_days.items():
        if keyword in reason_text:
            return days

    return 5.0


def aggregate_team_injuries(injury_df: pd.DataFrame) -> pd.DataFrame:
    # Collapse detailed player rows into team-level features for each game date.
    if injury_df.empty:
        return pd.DataFrame(
            columns = [
                "game_date",
                "team",
                "latest_report_datetime",
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
            ]
        )

    latest_player_status = (
        injury_df.sort_values("report_datetime")
        .drop_duplicates(subset = ["game_date", "team", "player_name"], keep = "last")
        .copy()
    )

    status_series = latest_player_status["status"].str.upper()
    latest_player_status["out_flag"] = status_series.eq("OUT").astype(int)
    latest_player_status["doubtful_flag"] = status_series.eq("DOUBTFUL").astype(int)
    latest_player_status["questionable_flag"] = status_series.eq("QUESTIONABLE").astype(int)
    latest_player_status["probable_flag"] = status_series.eq("PROBABLE").astype(int)
    latest_player_status["available_flag"] = status_series.eq("AVAILABLE").astype(int)
    latest_player_status["reported_player_flag"] = latest_player_status["player_name"].ne("").astype(int)
    latest_player_status["estimated_absence_days"] = latest_player_status["reason"].map(
        estimate_absence_days
    )
    latest_player_status["status_impact_multiplier"] = status_series.map(
        STATUS_IMPACT_MULTIPLIERS
    ).fillna(0.25)
    latest_player_status["weighted_status_score"] = (
        latest_player_status["out_flag"] * 1.0
        + latest_player_status["doubtful_flag"] * 0.75
        + latest_player_status["questionable_flag"] * 0.5
        + latest_player_status["probable_flag"] * 0.25
    )
    latest_player_status["injury_impact_score"] = (
        latest_player_status["estimated_absence_days"]
        * latest_player_status["status_impact_multiplier"]
    )
    latest_player_status["long_term_absence_flag"] = (
        (latest_player_status["estimated_absence_days"] >= 14.0)
        & status_series.ne("AVAILABLE")
    ).astype(int)

    return (
        latest_player_status.groupby(["game_date", "team"], as_index = False)
        .agg(
            latest_report_datetime = ("report_datetime", "max"),
            out_count = ("out_flag", "sum"),
            doubtful_count = ("doubtful_flag", "sum"),
            questionable_count = ("questionable_flag", "sum"),
            probable_count = ("probable_flag", "sum"),
            available_count = ("available_flag", "sum"),
            weighted_injury_score = ("weighted_status_score", "sum"),
            reported_player_count = ("reported_player_flag", "sum"),
            estimated_absence_days_total = ("estimated_absence_days", "sum"),
            estimated_absence_days_max = ("estimated_absence_days", "max"),
            injury_impact_score = ("injury_impact_score", "sum"),
            long_term_absence_count = ("long_term_absence_flag", "sum"),
        )
        .sort_values(["game_date", "team"])
        .reset_index(drop = True)
    )


def find_latest_report_timestamp_for_date(game_date: pd.Timestamp) -> Optional[datetime]:
    # Keep one latest valid report per game date to match the original project workflow.
    try:
        from nbainjuries.injury import check_reportvalid
    except ImportError:
        return None

    for hour in REPORT_HOURS:
        timestamp = datetime(game_date.year, game_date.month, game_date.day, hour, 0)
        try:
            if check_reportvalid(timestamp):
                return timestamp
        except Exception:
            continue
    return None


def fetch_report_with_retries(timestamp: datetime) -> Optional[pd.DataFrame]:
    try:
        from nbainjuries.injury import get_reportdata
    except ImportError:
        return None

    last_error = None
    for attempt in range(1, MAX_REPORT_FETCH_ATTEMPTS + 1):
        try:
            return get_reportdata(timestamp, return_df = True)
        except Exception as exc:
            last_error = exc
            if attempt < MAX_REPORT_FETCH_ATTEMPTS:
                import time

                time.sleep(RETRY_DELAY_SECONDS)

    if last_error is not None:
        print(f"Giving up on injury report {timestamp}: {last_error}")
    return None


def get_latest_available_report_timestamp(
    reference_timestamp: Optional[datetime] = None,
    lookback_days: int = 30,
) -> Optional[datetime]:
    # Walk backward from the reference time to find the latest report that actually exists.
    reference = reference_timestamp or datetime.now()
    for day_offset in range(lookback_days + 1):
        candidate_date = pd.Timestamp(reference.date()) - pd.Timedelta(days = day_offset)
        report_timestamp = find_latest_report_timestamp_for_date(candidate_date)
        if report_timestamp is not None:
            return report_timestamp
    return None


def scrape_injury_reports_for_dates(game_dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, list[str]]:
    return _fetch_injury_reports_concurrent(game_dates)


def _fetch_injury_report_for_date(game_date: pd.Timestamp) -> tuple[Optional[pd.DataFrame], bool]:
    """Fetch and normalize a single game-date injury report.

    Returns (report_df, success). report_df is None when no valid report was
    found; success is False when the date should be recorded as missing.
    """
    report_timestamp = find_latest_report_timestamp_for_date(pd.Timestamp(game_date))
    if report_timestamp is None:
        return None, False

    report_df = fetch_report_with_retries(report_timestamp)
    if report_df is None or report_df.empty:
        return None, False

    report_df = report_df.copy().rename(
        columns = {
            "Game Date": "game_date",
            "Game Time": "game_time",
            "Matchup": "matchup",
            "Team": "team",
            "Player Name": "player_name",
            "Current Status": "status",
            "Reason": "reason",
        }
    )
    report_df["game_date"] = pd.to_datetime(report_df["game_date"]).dt.normalize()
    report_df["team"] = report_df["team"].map(TEAM_MAP_BREF).fillna(report_df["team"])
    report_df["player_name"] = report_df["player_name"].astype(str).str.strip()
    report_df["status"] = report_df["status"].astype(str).str.strip()
    report_df["reason"] = report_df["reason"].astype(str).str.strip()
    report_df["report_datetime"] = pd.Timestamp(report_timestamp)
    return report_df, True


# Cap concurrent injury-report HTTP requests. Each report makes up to 7
# validity checks, so this keeps total in-flight requests reasonable.
MAX_CONCURRENT_INJURY_FETCHES = 20


def _fetch_injury_reports_concurrent(
    game_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch injury reports for many dates concurrently using a thread pool."""
    detail_frames = []
    missing_dates = []

    sorted_dates = sorted(pd.to_datetime(game_dates).normalize().unique())

    with ThreadPoolExecutor(max_workers = MAX_CONCURRENT_INJURY_FETCHES) as executor:
        results = executor.map(_fetch_injury_report_for_date, [pd.Timestamp(d) for d in sorted_dates])

    for game_date, (report_df, success) in zip(sorted_dates, results):
        if success and report_df is not None:
            detail_frames.append(report_df)
        else:
            missing_dates.append(str(pd.Timestamp(game_date).date()))

    if not detail_frames:
        return pd.DataFrame(), missing_dates

    return pd.concat(detail_frames, ignore_index = True), missing_dates


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
    detail_frames = []
    missing_dates = []

    detail_df, missing_dates = _fetch_injury_reports_concurrent(list(game_dates))
    if not detail_df.empty:
        detail_frames.append(detail_df)

    if not detail_frames:
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

    detailed_injury_df = pd.concat(detail_frames, ignore_index = True)
    injury_df = aggregate_team_injuries(detailed_injury_df)
    start_season = OFFICIAL_INJURY_FIRST_SEASON
    end_season = int(eligible_games["season"].max())
    detail_output_path = RAW_DIR / f"official_nba_injuries_detailed_{start_season}_{end_season}.csv"
    output_path = RAW_DIR / f"official_nba_injuries_by_team_{start_season}_{end_season}.csv"
    detailed_injury_df.to_csv(detail_output_path, index = False)
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


def get_numeric_features() -> list[str]:
    return list(NUMERIC_FEATURES)


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


def _bounded_rest_days(last_date: Optional[pd.Timestamp], game_date: pd.Timestamp) -> int:
    if last_date is None:
        return MAX_REST_DAYS

    raw_rest_days = (game_date - last_date).days - 1
    return max(0, min(raw_rest_days, MAX_REST_DAYS))


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
    home_rest_days = _bounded_rest_days(home_state["last_date"], game_date)
    away_rest_days = _bounded_rest_days(away_state["last_date"], game_date)

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
        "home_recent_win_pct": home_recent_win_pct,
        "away_recent_win_pct": away_recent_win_pct,
        "recent_win_pct_diff": home_recent_win_pct - away_recent_win_pct,
        "home_recent_margin": home_recent_margin,
        "away_recent_margin": away_recent_margin,
        "recent_margin_diff": home_recent_margin - away_recent_margin,
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "rest_diff": home_rest_days - away_rest_days,
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

    return pd.DataFrame(feature_rows)


def merge_injury_features(feature_df: pd.DataFrame, injury_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    result = feature_df.copy()

    # Injury feature column families added by the merge. Keeping these as
    # module-level constants avoids rebuilding the dict on every call.
    INJURY_FILL_ZERO_COLUMNS = [
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
    INJURY_DIFF_COLUMNS = (
        ("out_count_diff", "away_out_count", "home_out_count"),
        ("doubtful_count_diff", "away_doubtful_count", "home_doubtful_count"),
        ("questionable_count_diff", "away_questionable_count", "home_questionable_count"),
        ("probable_count_diff", "away_probable_count", "home_probable_count"),
        ("weighted_injury_diff", "away_weighted_injury_score", "home_weighted_injury_score"),
        ("estimated_absence_days_diff", "away_estimated_absence_days_total", "home_estimated_absence_days_total"),
        ("max_absence_days_diff", "away_estimated_absence_days_max", "home_estimated_absence_days_max"),
        ("injury_impact_diff", "away_injury_impact_score", "home_injury_impact_score"),
        ("long_term_absence_diff", "away_long_term_absence_count", "home_long_term_absence_count"),
    )

    if injury_df is None or injury_df.empty:
        # Assign all default columns in one shot instead of looping.
        defaults = {col: 0.0 for col in INJURY_FILL_ZERO_COLUMNS}
        defaults["home_injury_report_available"] = 0
        defaults["away_injury_report_available"] = 0
        defaults.update({name: 0.0 for name, _, _ in INJURY_DIFF_COLUMNS})
        for column, value in defaults.items():
            result[column] = value
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

    result = result.merge(home_injuries, on = ["date", "home_team"], how = "left")
    result = result.merge(away_injuries, on = ["date", "away_team"], how = "left")

    # Batch fillna on all numeric injury columns at once.
    result[INJURY_FILL_ZERO_COLUMNS] = result[INJURY_FILL_ZERO_COLUMNS].fillna(0.0)

    result["home_injury_report_available"] = result["home_latest_report_datetime"].notna().astype(int)
    result["away_injury_report_available"] = result["away_latest_report_datetime"].notna().astype(int)
    for diff_name, away_col, home_col in INJURY_DIFF_COLUMNS:
        result[diff_name] = result[away_col] - result[home_col]
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
        if latest_row is not None:
            report_age_days = (normalized_game_date - latest_row["game_date"]).days
            if report_age_days > MAX_INJURY_LOOKBACK_DAYS:
                latest_row = None
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
) -> tuple[Pipeline, Pipeline, pd.DataFrame]:
    if games.empty:
        raise ValueError("At least one completed game is required to train the predictor.")

    feature_df = merge_injury_features(build_pregame_features(games), injury_df)
    if feature_df.empty:
        raise ValueError("Feature engineering produced no training rows.")

    y = feature_df["home_win"]
    numeric_features = get_numeric_features()
    model_columns = CATEGORICAL_FEATURES + numeric_features

    logistic_regression = Pipeline(
        steps = [
            ("preprocessor", _build_preprocessor(numeric_features)),
            ("classifier", LogisticRegression(max_iter = 2000, random_state = 100)),
        ]
    )
    random_forest = Pipeline(
        steps = [
            ("preprocessor", _build_preprocessor(numeric_features)),
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

    logistic_regression.fit(feature_df[model_columns], y)
    random_forest.fit(feature_df[model_columns], y)
    return logistic_regression, random_forest, feature_df


def evaluate_prediction_models(
    games: pd.DataFrame,
    injury_df: Optional[pd.DataFrame] = None,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame]:
    if games.empty:
        raise ValueError("At least one completed game is required to evaluate the predictor.")

    ordered_games = games.sort_values(["date", "home_team", "away_team"]).reset_index(drop = True)
    feature_df = merge_injury_features(build_pregame_features(ordered_games), injury_df)
    feature_df = feature_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop = True)

    if len(feature_df) < 10:
        raise ValueError("Not enough completed games are available for a meaningful evaluation.")

    latest_season = int(feature_df["season"].max())
    if feature_df["season"].nunique() > 1:
        train_games = ordered_games[ordered_games["season"] < latest_season].copy()
        test_df = feature_df[feature_df["season"] == latest_season].copy()
        split_label = f"train_before_{latest_season}_test_{latest_season}"
        test_season = latest_season
    else:
        split_index = max(int(len(feature_df) * 0.8), 1)
        train_games = ordered_games.iloc[:split_index].copy()
        test_df = feature_df.iloc[split_index:].copy()
        split_label = "chronological_80_20"
        test_season = int(test_df["season"].max())

    if train_games.empty or test_df.empty:
        raise ValueError("Evaluation split produced an empty train or test set.")

    logistic_regression, random_forest, _ = train_prediction_models(train_games, injury_df)
    model_columns = list(logistic_regression.feature_names_in_)

    logistic_home_probs = logistic_regression.predict_proba(test_df[model_columns])[:, 1]
    random_forest_home_probs = random_forest.predict_proba(test_df[model_columns])[:, 1]
    logistic_predictions = (logistic_home_probs >= 0.5).astype(int)
    random_forest_predictions = (random_forest_home_probs >= 0.5).astype(int)
    actual_home_wins = test_df["home_win"].astype(int).to_numpy()
    home_baseline_predictions = [1] * len(test_df)

    summary = {
        "split": split_label,
        "train_rows": int(len(train_games)),
        "test_rows": int(len(test_df)),
        "test_season": int(test_season),
        "injury_data_used": bool(injury_df is not None and not injury_df.empty),
        "home_baseline_accuracy": float(accuracy_score(actual_home_wins, home_baseline_predictions)),
        "logistic_accuracy": float(accuracy_score(actual_home_wins, logistic_predictions)),
        "random_forest_accuracy": float(accuracy_score(actual_home_wins, random_forest_predictions)),
    }
    summary["best_model"] = max(
        (
            ("Home baseline", summary["home_baseline_accuracy"]),
            ("Logistic regression", summary["logistic_accuracy"]),
            ("Random forest", summary["random_forest_accuracy"]),
        ),
        key = lambda item: item[1],
    )[0]
    summary["random_forest_vs_logistic_gain"] = (
        summary["random_forest_accuracy"] - summary["logistic_accuracy"]
    )
    summary["logistic_vs_baseline_gain"] = (
        summary["logistic_accuracy"] - summary["home_baseline_accuracy"]
    )
    summary["random_forest_vs_baseline_gain"] = (
        summary["random_forest_accuracy"] - summary["home_baseline_accuracy"]
    )

    export_df = test_df[
        [
            "date",
            "season",
            "home_team",
            "away_team",
            "home_win",
            "home_win_pct",
            "away_win_pct",
            "home_team_strength",
            "away_team_strength",
            "home_out_count",
            "away_out_count",
            "home_injury_impact_score",
            "away_injury_impact_score",
        ]
    ].copy()
    export_df["matchup"] = export_df["away_team"] + " @ " + export_df["home_team"]
    # Vectorized winner selection: avoid per-row Python via .apply(axis=1).
    home_teams_col = export_df["home_team"].to_numpy()
    away_teams_col = export_df["away_team"].to_numpy()
    home_win_col = export_df["home_win"].to_numpy()
    export_df["true_winner"] = np.where(home_win_col == 1, home_teams_col, away_teams_col)
    export_df["logistic_home_win_probability"] = logistic_home_probs
    export_df["logistic_predicted_winner"] = np.where(
        logistic_home_probs >= 0.5, home_teams_col, away_teams_col
    )
    export_df["random_forest_home_win_probability"] = random_forest_home_probs
    export_df["random_forest_predicted_winner"] = np.where(
        random_forest_home_probs >= 0.5, home_teams_col, away_teams_col
    )
    export_df = export_df[
        [
            "date",
            "season",
            "matchup",
            "home_team",
            "away_team",
            "true_winner",
            "home_win",
            "logistic_predicted_winner",
            "logistic_home_win_probability",
            "random_forest_predicted_winner",
            "random_forest_home_win_probability",
            "home_win_pct",
            "away_win_pct",
            "home_team_strength",
            "away_team_strength",
            "home_out_count",
            "away_out_count",
            "home_injury_impact_score",
            "away_injury_impact_score",
        ]
    ]

    return summary, export_df


def _compute_state_snapshot(
    games: pd.DataFrame,
    season: int,
    game_date: pd.Timestamp,
    home_team: str,
    away_team: str,
) -> dict[str, object]:
    # Normalize once instead of calling pd.to_datetime per-row in the mask.
    dates = pd.to_datetime(games["date"])
    season_games = games[
        (games["season"] == season) & (dates < game_date)
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

    available_seasons = frozenset(int(value) for value in games["season"].unique())
    if season not in available_seasons:
        raise ValueError(f"Season {season} is not present in the processed games dataset.")

    dates = pd.to_datetime(games["date"])
    cutoff_games = games[dates < game_date].copy()
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
    injury_data_used = bool(
        matchup_df.at[0, "home_injury_report_available"]
        or matchup_df.at[0, "away_injury_report_available"]
    )

    return {
        "matchup": f"{away_team} @ {home_team}",
        "season": season,
        "game_date": game_date.strftime("%Y-%m-%d"),
        "home_team": home_team,
        "away_team": away_team,
        "logistic": {
            "home_win_probability": lr_home_prob,
            "home_win_prediction": lr_home_pred,
            "predicted_winner": home_team if lr_home_pred == 1 else away_team,
        },
        "random_forest": {
            "home_win_probability": rf_home_prob,
            "home_win_prediction": rf_home_pred,
            "predicted_winner": home_team if rf_home_pred == 1 else away_team,
        },
        "context": {
            "home_win_pct": float(matchup_df.at[0, "home_win_pct"]),
            "away_win_pct": float(matchup_df.at[0, "away_win_pct"]),
            "home_team_strength": float(matchup_df.at[0, "home_team_strength"]),
            "away_team_strength": float(matchup_df.at[0, "away_team_strength"]),
            "team_strength_diff": float(matchup_df.at[0, "team_strength_diff"]),
            "home_rest_days": int(matchup_df.at[0, "home_rest_days"]),
            "away_rest_days": int(matchup_df.at[0, "away_rest_days"]),
            "home_out_count": float(matchup_df.at[0, "home_out_count"]),
            "away_out_count": float(matchup_df.at[0, "away_out_count"]),
            "home_injury_impact_score": float(matchup_df.at[0, "home_injury_impact_score"]),
            "away_injury_impact_score": float(matchup_df.at[0, "away_injury_impact_score"]),
        },
        "injury_data_used": injury_data_used,
    }
