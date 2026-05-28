# updater.py  (daily manual update tool)

from pathlib import Path
import pandas as pd

from model import run_elo

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Baseline processed games: 2016–2025 only, created by main.py
BASE_PROCESSED = PROCESSED_DIR / "games_2016_2025_normalized.csv"

# Per-day game inputs (one file per NBA calendar day)
DAILY_GAMES_DIR = DATA_DIR / "daily_updates"

# Extended dataset: baseline + all daily games
EXTENDED_PROCESSED = PROCESSED_DIR / "games_extended_daily.csv"

# Where to store *ratings* snapshots per day
DAILY_RATINGS_DIR = RESULTS_DIR / "daily_ratings"


def add_changes(prev_standings: pd.DataFrame,
                curr_standings: pd.DataFrame) -> pd.DataFrame:
    prev = prev_standings.reset_index(drop=True).copy()
    curr = curr_standings.reset_index(drop=True).copy()

    prev["prev_rank"] = prev.index + 1
    curr["curr_rank"] = curr.index + 1

    merged = curr.merge(
        prev[["team", "rating", "prev_rank"]].rename(
            columns={"rating": "prev_rating"}
        ),
        on="team",
        how="left",
    )

    merged["delta_rating"] = merged["rating"] - merged["prev_rating"]
    merged["delta_rank"] = merged["prev_rank"] - merged["curr_rank"]

    return merged[["team", "rating", "delta_rating", "delta_rank"]]


def load_baseline_games() -> pd.DataFrame:
    if not BASE_PROCESSED.exists():
        raise FileNotFoundError(
            f"Baseline processed games not found at {BASE_PROCESSED}.\n"
            f"Run main.py once to create it."
        )
    df = pd.read_csv(BASE_PROCESSED)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_all_daily_games() -> pd.DataFrame:
    cols = [
        "date",
        "season",
        "home_team",
        "away_team",
        "home_pts",
        "away_pts",
        "home_win",
        "margin",
    ]

    if not DAILY_GAMES_DIR.exists():
        return pd.DataFrame(columns=cols)

    frames = []
    for f in sorted(DAILY_GAMES_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=cols)

    return pd.concat(frames, ignore_index=True)[cols]


def prompt_day_games():
    print("\n=== Enter ONE NBA day of games ===")
    print("You will enter the DATE once, then multiple games for that date.\n")
    print("Format:")
    print("  Date       : MM/DD/YYYY")
    print("  Season     : YYYY (season end year, e.g. 2026 for 2025–26)")
    print("  Home team  : 3-letter code (GSW, DEN, BOS, ...)")
    print("  Away team  : 3-letter code")
    print("  Home points: integer")
    print("  Away points: integer\n")

    date_str = input("Date for this batch (MM/DD/YYYY) [blank to cancel]: ").strip()
    if date_str == "":
        print("No date entered, cancelling update.")
        return None, None