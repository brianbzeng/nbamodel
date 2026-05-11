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

    try:
        day_date = pd.to_datetime(date_str, format="%m/%d/%Y")
    except ValueError as e:
        print(f"Invalid date: {e}. Cancelling.")
        return None, None

    season_str = input("Season end year (YYYY): ").strip()
    try:
        season = int(season_str)
    except ValueError:
        print("Invalid season year. Cancelling.")
        return None, None

    print("\nNow enter EACH GAME for that date.")
    print("Leave HOME TEAM blank and press Enter when you are finished.\n")

    rows = []
    while True:
        home_team = input("Home team (e.g. GSW, blank to finish): ").strip().upper()
        if home_team == "":
            break

        away_team = input("Away team (e.g. LAL): ").strip().upper()
        home_pts_str = input("Home points: ").strip()
        away_pts_str = input("Away points: ").strip()

        try:
            home_pts = int(home_pts_str)
            away_pts = int(away_pts_str)
        except ValueError:
            print("  !! Points must be integers. Try that game again.\n")
            continue

        home_win = int(home_pts > away_pts)
        margin = abs(home_pts - away_pts)

        rows.append(
            {
                "date": day_date,
                "season": season,
                "home_team": home_team,
                "away_team": away_team,
                "home_pts": home_pts,
                "away_pts": away_pts,
                "home_win": home_win,
                "margin": margin,
            }
        )

        print("  Added game.\n")

    if not rows:
        print("No games entered for this date.")
        return None, None

    day_df = pd.DataFrame(rows)
    return day_date, day_df


def save_day_games(day_date: pd.Timestamp, day_df: pd.DataFrame):
    DAILY_GAMES_DIR.mkdir(parents=True, exist_ok=True)
    fname = DAILY_GAMES_DIR / f"{day_date.strftime('%Y-%m-%d')}.csv"
    day_df.to_csv(fname, index=False)
    print(f"\nSaved today's games to {fname}")


def compute_season_start_standings(baseline_games: pd.DataFrame,
                                   base_rating=1500.0,
                                   reg_factor=0.75):
    _, final_ratings = run_elo(baseline_games,
                               base_rating=base_rating,
                               k=20.0,
                               hca=50.0,
                               scale=400.0,
                               use_margin=True,
                               season_regress=True,
                               reg_factor=reg_factor)

    start_ratings = {
        team: reg_factor * r + (1 - reg_factor) * base_rating
        for team, r in final_ratings.items()
    }

    df = (
        pd.DataFrame(
            {
                "team": list(start_ratings.keys()),
                "rating": list(start_ratings.values()),
            }
        )
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    return df


def compute_standings_for_games(games: pd.DataFrame) -> pd.DataFrame:
    results, final_ratings = run_elo(
        games,
        base_rating=1500.0,
        k=20.0,
        hca=50.0,
        scale=400.0,
        use_margin=True,
        season_regress=True,
        reg_factor=0.75,
    )

    last_season = games["season"].max()
    last_teams = sorted(
        set(games.loc[games["season"] == last_season, "home_team"])
        .union(games.loc[games["season"] == last_season, "away_team"])
    )

    final_table = (
        pd.DataFrame(
            {
                "team": last_teams,
                "rating": [final_ratings[t] for t in last_teams],
            }
        )
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )

    return final_table, results


def save_daily_ratings_snapshot(day_date: pd.Timestamp, standings: pd.DataFrame):
    DAILY_RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    fname = DAILY_RATINGS_DIR / f"ratings_{day_date.strftime('%Y-%m-%d')}.csv"
    standings.to_csv(fname, index=False)
    print(f"Saved today's Elo standings snapshot to {fname}")


if __name__ == "__main__":
    baseline_games = load_baseline_games()

    day_date, day_df = prompt_day_games()
    if day_df is not None:
        save_day_games(day_date, day_df)

    all_daily_games = load_all_daily_games()

    if day_df is not None:
        prev_daily_games = all_daily_games[all_daily_games["date"] < day_date]
        curr_daily_games = all_daily_games[all_daily_games["date"] <= day_date]
    else:
        prev_daily_games = all_daily_games.copy()
        curr_daily_games = all_daily_games.copy()

    prev_games = (
        pd.concat([baseline_games, prev_daily_games], ignore_index=True)
        if not prev_daily_games.empty
        else baseline_games.copy()
    )
    prev_games = prev_games.sort_values("date").reset_index(drop=True)

    curr_games = (
        pd.concat([baseline_games, curr_daily_games], ignore_index=True)
        if not curr_daily_games.empty
        else baseline_games.copy()
    )
    curr_games = curr_games.sort_values("date").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    curr_games.to_csv(EXTENDED_PROCESSED, index=False)
    print(f"\nSaved extended daily dataset to {EXTENDED_PROCESSED}")

    season_start_standings = compute_season_start_standings(baseline_games)
    prev_standings, _ = compute_standings_for_games(prev_games)
    curr_standings, _ = compute_standings_for_games(curr_games)

    curr_with_changes = add_changes(prev_standings, curr_standings)

    print("\n=== SEASON START STANDINGS (next season) ===")
    print(season_start_standings)

    print("\n=== PREVIOUS DAY STANDINGS ===")
    print(prev_standings)

    print("\n=== CURRENT DAY STANDINGS (after today's games) ===")
    print(curr_with_changes)

    if day_df is not None:
        save_daily_ratings_snapshot(day_date, curr_with_changes)
