import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from nbainjuries.injury import check_reportvalid, get_reportdata


RAW_DIR = Path("data/raw")
REPORT_HOURS = [17, 14, 13, 12, 11, 10, 9]
MAX_REPORT_FETCH_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2

STATUS_IMPACT_MULTIPLIERS = {
    "OUT": 1.0,
    "DOUBTFUL": 0.75,
    "QUESTIONABLE": 0.45,
    "PROBABLE": 0.15,
    "AVAILABLE": 0.0,
}


def prompt_for_season_range() -> tuple[int, int]:
    print("\n=== NBA Injury Scraper (nbainjuries) ===")
    start_season = int(input("Start season ending year (e.g. 2022): ").strip())
    end_season = int(input(f"End season ending year (e.g. {start_season}): ").strip())

    if start_season > end_season:
        raise ValueError("Start season ending year must be less than or equal to end season ending year.")

    return start_season, end_season


def find_source_game_csv(start_season: int, end_season: int) -> Path:
    # Reuse a compatible game CSV so that any injury data matches with game dates already in our pipeline
    exact_path = RAW_DIR / f"bref_games_{start_season}_{end_season}.csv"
    if exact_path.exists():
        return exact_path

    candidate_files = sorted(RAW_DIR.glob("bref_games*.csv"))
    if not candidate_files:
        raise FileNotFoundError("No bref_games CSV files were found in data/raw.")

    compatible_files = []
    for path in candidate_files:
        try:
            df = pd.read_csv(path, usecols = ["season"])
        except Exception:
            continue

        if df.empty:
            continue

        file_start = int(df["season"].min())
        file_end = int(df["season"].max())
        if file_start <= start_season and file_end >= end_season:
            compatible_files.append((file_end - file_start, path))

    if not compatible_files:
        raise FileNotFoundError(
            f"No bref_games CSV in data/raw covers seasons {start_season}-{end_season}."
        )

    compatible_files.sort(key = lambda item: (item[0], item[1].name))
    return compatible_files[0][1]


def load_game_dates_for_range(game_csv_path: Path, start_season: int, end_season: int) -> pd.DataFrame:
    df = pd.read_csv(game_csv_path, parse_dates = ["date"])
    df = df[(df["season"] >= start_season) & (df["season"] <= end_season)].copy()
    df = df.sort_values("date").reset_index(drop = True)

    if df.empty:
        raise ValueError(
            f"No games found in {game_csv_path.name} for seasons {start_season}-{end_season}."
        )

    return df


def find_latest_report_timestamp_for_date(game_date: pd.Timestamp) -> Optional[datetime]:
    # Search for and keep the latest valid injury reports
    for hour in REPORT_HOURS:
        ts = datetime(game_date.year, game_date.month, game_date.day, hour, 0)
        try:
            if check_reportvalid(ts):
                return ts
        except Exception:
            continue
    return None


def fetch_report_with_retries(timestamp: datetime) -> Optional[pd.DataFrame]:
    # Retry fetch failures instead of losing the entire scrape
    last_error = None

    for attempt in range(1, MAX_REPORT_FETCH_ATTEMPTS + 1):
        try:
            return get_reportdata(timestamp, return_df = True)
        except Exception as exc:
            last_error = exc
            print(
                f"Attempt {attempt}/{MAX_REPORT_FETCH_ATTEMPTS} failed for "
                f"{timestamp}: {exc}"
            )
            if attempt < MAX_REPORT_FETCH_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)

    print(f"Giving up on report {timestamp}: {last_error}")
    return None


def scrape_reports_for_game_dates(game_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Pull one injury report for each unique game date in the range
    unique_dates = sorted(game_df["date"].dt.normalize().unique())
    frames = []
    missing_dates = []

    for game_date in unique_dates:
        timestamp = find_latest_report_timestamp_for_date(pd.Timestamp(game_date))
        if timestamp is None:
            missing_dates.append(pd.Timestamp(game_date))
            print(f"No report found for {pd.Timestamp(game_date).date()}")
            continue

        print(f"Using report {timestamp} for game date {pd.Timestamp(game_date).date()}")
        report_df = fetch_report_with_retries(timestamp)
        if report_df is None:
            missing_dates.append(pd.Timestamp(game_date))
            continue

        report_df = report_df.copy()
        report_df["report_datetime"] = pd.Timestamp(timestamp)
        frames.append(report_df)

    if not frames:
        return pd.DataFrame(), pd.DataFrame({"missing_game_date": missing_dates})

    combined = pd.concat(frames, ignore_index = True)
    combined["Game Date"] = pd.to_datetime(combined["Game Date"])
    combined = combined.rename(
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
    combined["game_date"] = combined["game_date"].dt.normalize()
    combined["team"] = combined["team"].astype(str).str.strip()
    combined["player_name"] = combined["player_name"].astype(str).str.strip()
    combined["status"] = combined["status"].astype(str).str.strip()

    missing_df = pd.DataFrame({"missing_game_date": missing_dates})
    return combined, missing_df


def estimate_absence_days(reason: str) -> float:
    # Map common injury descriptions to a rough expected absence length (in days)
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
    # Aggregate injury features for modeling
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
    latest_player_status["estimated_absence_days"] = latest_player_status["reason"].map(estimate_absence_days)
    latest_player_status["status_impact_multiplier"] = status_series.map(STATUS_IMPACT_MULTIPLIERS).fillna(0.25)
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


def build_output_paths(start_season: int, end_season: int) -> tuple[Path, Path, Path]:
    suffix = f"{start_season}_{end_season}"
    detail_path = RAW_DIR / f"official_nba_injuries_detailed_{suffix}.csv"
    team_path = RAW_DIR / f"official_nba_injuries_by_team_{suffix}.csv"
    missing_path = RAW_DIR / f"official_nba_injuries_missing_dates_{suffix}.csv"
    return detail_path, team_path, missing_path


def main() -> None:
    start_season, end_season = prompt_for_season_range()
    game_csv_path = find_source_game_csv(start_season, end_season)
    game_df = load_game_dates_for_range(game_csv_path, start_season, end_season)
    print(
        f"Using {game_csv_path.name} as the game-date source for seasons "
        f"{start_season}-{end_season}"
    )

    detail_df, missing_df = scrape_reports_for_game_dates(game_df)
    team_df = aggregate_team_injuries(detail_df)
    detail_path, team_path, missing_path = build_output_paths(start_season, end_season)

    RAW_DIR.mkdir(parents = True, exist_ok = True)
    detail_df.to_csv(detail_path, index = False)
    team_df.to_csv(team_path, index = False)
    missing_df.to_csv(missing_path, index = False)

    print(f"\nSaved {len(detail_df):,} detailed injury rows to {detail_path}")
    print(f"Saved {len(team_df):,} team-level injury rows to {team_path}")
    print(f"Saved {len(missing_df):,} missing game dates to {missing_path}")
    if not detail_df.empty:
        print(
            "Injury report game-date range: "
            f"{detail_df['game_date'].min().date()} to {detail_df['game_date'].max().date()}"
        )


if __name__ == "__main__":
    main()
