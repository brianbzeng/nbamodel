# main.py

from pathlib import Path
import pandas as pd

from cleaner import clean_games, normalize_team_names
from model import run_elo

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

RAW_FILE = RAW_DIR / "bref_games_2016_2025.csv"
PROCESSED_FILE = PROCESSED_DIR / "games_2016_2025_normalized.csv"


def load_or_build_processed():
    if PROCESSED_FILE.exists():
        print("Loading processed games...")
        return pd.read_csv(PROCESSED_FILE)

    print("Loading raw games and processing...")
    df = pd.read_csv(RAW_FILE, parse_dates=["date"])
    df = clean_games(df)
    df = normalize_team_names(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"Saved processed games to {PROCESSED_FILE}")
    return df


if __name__ == "__main__":
    games = load_or_build_processed()
    print(games.head())

    # Optional: compute a baseline Elo for 2016–2025 (for analysis only)
    results, final_ratings = run_elo(games)

    print("\nBaseline Elo performance across seasons 2016–2025:")
    print("Log loss:", results["logloss"].mean())
    print("Brier score:", results["brier"].mean())

    last_season = games["season"].max()
    last_teams = sorted(
        set(games.loc[games["season"] == last_season, "home_team"])
        .union(games.loc[games["season"] == last_season, "away_team"])
    )
    baseline_final = (
        pd.DataFrame(
            {
                "team": last_teams,
                "rating": [final_ratings[t] for t in last_teams],
            }
        )
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_final.to_csv(
        RESULTS_DIR / "elo_baseline_2016_2025_final_ratings.csv", index=False
    )
    results.to_csv(
        RESULTS_DIR / "elo_baseline_2016_2025_game_results.csv", index=False
    )

    print("Saved baseline Elo results:")
    print(" - elo_baseline_2016_2025_final_ratings.csv")
    print(" - elo_baseline_2016_2025_game_results.csv")
