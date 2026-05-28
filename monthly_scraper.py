from pathlib import Path
import pandas as pd

from model import run_elo
from scraper import scrape_bref_month_games
from cleaner import clean_games, normalize_team_names

# ---------- Paths ----------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Baseline processed games: 2016–2025 only, created by main.py
BASE_PROCESSED = PROCESSED_DIR / "games_2016_2025_normalized.csv"

# Where we store per-month game data (real + simulated) for 2026+ seasons
MONTHLY_UPDATES_DIR = DATA_DIR / "monthly_updates"

# Extended dataset: baseline + all monthly updates
EXTENDED_PROCESSED = PROCESSED_DIR / "games_extended_monthly.csv"

# Where we store monthly Elo standings snapshots
MONTHLY_RATINGS_DIR = RESULTS_DIR / "monthly_ratings"


# ---------- Duplicate warning helper ----------

def warn_and_filter_duplicates(new_df: pd.DataFrame,
                               all_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Warn about games in new_df that already exist in all_monthly,
    based on (date, home_team, away_team), and drop them from new_df.

    Returns a filtered copy of new_df with duplicates removed.
    """
    if all_monthly is None or all_monthly.empty or new_df.empty:
        return new_df

    key_cols = ["date", "home_team", "away_team"]

    all_monthly = all_monthly.copy()
    all_monthly["date"] = pd.to_datetime(all_monthly["date"])
    new_df = new_df.copy()
    new_df["date"] = pd.to_datetime(new_df["date"])

    merged = new_df.merge(
        all_monthly[key_cols].drop_duplicates(),
        on=key_cols,
        how="left",
        indicator=True,
    )

    dup_mask = merged["_merge"] == "both"
    num_dups = dup_mask.sum()

    if num_dups > 0:
        print(f"\n⚠️  WARNING: {num_dups} duplicate game(s) detected in this month import.")
        dups = merged.loc[dup_mask, key_cols].sort_values(["date", "home_team"])
        print("These games already exist and will be skipped:\n")
        print(dups.to_string(index=False))
        print()

    filtered = merged.loc[~dup_mask, new_df.columns]
    return filtered.reset_index(drop=True)


# ---------- Core loaders ----------

def load_baseline_games() -> pd.DataFrame:
    if not BASE_PROCESSED.exists():
        raise FileNotFoundError(
            f"Baseline processed games not found at {BASE_PROCESSED}.\n"
            f"Run main.py once to create it."
        )
    df = pd.read_csv(BASE_PROCESSED)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_all_monthly_updates() -> pd.DataFrame:
    """
    Load all monthly update CSVs from data/monthly_updates into one DataFrame.
    Handles both real and simulated months (2026+).
    """
    cols = [
        "date",
        "season",
        "home_team",
        "away_team",
        "home_pts",
        "away_pts",
        "home_win",
        "margin",
        "source",  # "real" or "sim"
    ]

    if not MONTHLY_UPDATES_DIR.exists():
        return pd.DataFrame(columns=cols)