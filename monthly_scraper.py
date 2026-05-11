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

    frames = []
    for f in sorted(MONTHLY_UPDATES_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=cols)

    merged = pd.concat(frames, ignore_index=True)

    # In case old files don’t have 'source'
    if "source" not in merged.columns:
        merged["source"] = "unknown"

    return merged[cols]


# ---------- Movement helper ----------

def add_changes(prev_standings: pd.DataFrame,
                curr_standings: pd.DataFrame) -> pd.DataFrame:
    """
    Take previous and current standings (both sorted by rating desc),
    and return current standings with:

      - delta_rating: current_rating - previous_rating
      - delta_rank  : previous_rank - current_rank  (positive = moved up)
    """
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


# ---------- Elo wrappers ----------

def compute_season_start_standings(baseline_games: pd.DataFrame,
                                   base_rating=1500.0,
                                   reg_factor=0.75) -> pd.DataFrame:
    """
    Run Elo on baseline data and compute "start of next season" ratings
    by regressing once toward the mean.
    """
    _, final_ratings = run_elo(
        baseline_games,
        base_rating=base_rating,
        k=20.0,
        hca=50.0,
        scale=400.0,
        use_margin=True,
        season_regress=True,
        reg_factor=reg_factor,
    )

    start_ratings = {
        team: reg_factor * r + (1 - reg_factor) * base_rating
        for team, r in final_ratings.items()
    }

    df = (
        pd.DataFrame(
            {"team": list(start_ratings.keys()), "rating": list(start_ratings.values())}
        )
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    return df


def compute_standings_for_games(games: pd.DataFrame) -> pd.DataFrame:
    """
    Run Elo on the provided games and return standings for the most recent season.
    """
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


def save_monthly_ratings_snapshot(label: str,
                                  standings_with_changes: pd.DataFrame):
    """
    Save the current month's standings (with deltas) to:
      data/results/monthly_ratings/ratings_<label>.csv
    where label is like '2026-10' or '2026-11'.
    """
    MONTHLY_RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    fname = MONTHLY_RATINGS_DIR / f"ratings_{label}.csv"
    standings_with_changes.to_csv(fname, index=False)
    print(f"Saved monthly Elo standings snapshot to {fname}")


# ---------- Menu + import / simulate ----------

def menu_choice() -> str:
    print("\n=== NBA Elo Monthly Updater ===")
    print("0) Exit")
    print("1) Import REAL month from Basketball-Reference")
    print("2) SIMULATE a month of games "
          "(WARNING: this will skew Elo away from real history)")
    choice = input("Select an option [0/1/2]: ").strip()
    return choice


def prompt_season_and_month():
    season_str = input("Season END year (YYYY), e.g. 2026 for 2025–26: ").strip()
    month_str = input("Month number (10=Oct, 11=Nov, 12=Dec, 1=Jan, 2=Feb, ...): ").strip()

    try:
        season = int(season_str)
        month = int(month_str)
    except ValueError:
        raise ValueError("Season and month must be integers.")

    if season <= 2025:
        raise ValueError(
            "Season end year must be >= 2026. "
            "Baseline already includes seasons through 2025."
        )

    return season, month


def import_real_month(season: int, month: int, all_monthly_before: pd.DataFrame) -> pd.DataFrame:
    """
    Scrape a real month of games from Basketball-Reference and save to monthly_updates.
    Includes duplicate warning and FULL normalization so it matches the baseline.
    """
    df_raw = scrape_bref_month_games(season, month)
    if df_raw.empty:
        print("No games found for that month (maybe schedule not complete yet?).")
        return df_raw

    # 1) Clean: add home_win, margin, canonical date (no timestamp)
    df_clean = clean_games(df_raw)

    # 2) Normalize team names to 3-letter abbreviations (ATL, GSW, ...)
    df_norm = normalize_team_names(df_clean)

    # 3) Tag source
    df_norm["source"] = "real"

    # 4) Duplicate warning vs all previous monthly updates
    df_norm = warn_and_filter_duplicates(df_norm, all_monthly_before)

    if df_norm.empty:
        print("After removing duplicates, no new games to add for this month.")
        return df_norm

    # 5) Save month CSV, same column order as baseline + 'source'
    MONTHLY_UPDATES_DIR.mkdir(parents=True, exist_ok=True)
    fname = MONTHLY_UPDATES_DIR / f"games_{season}-{month:02d}.csv"
    df_norm.to_csv(fname, index=False)
    print(f"Saved real month games to {fname}")

    return df_norm


def simulate_month(season: int, month: int) -> pd.DataFrame:
    """
    Let user manually input a month worth of games (simulated).
    Note: all games are tagged as 'sim' and will skew Elo.
    Here we assume the user enters 3-letter team codes already (ATL, GSW, ...),
    so we DO NOT normalize names again.
    """
    print("\n=== SIMULATE a month of games ===")
    print("All games will be stored as simulated and included in Elo.")
    print("You can clear or ignore them later if needed.\n")

    rows = []

    while True:
        date_str = input("Game date (MM/DD/YYYY, blank to finish): ").strip()
        if date_str == "":
            break

        try:
            date = pd.to_datetime(date_str, format="%m/%d/%Y")
        except ValueError as e:
            print(f"  Invalid date: {e}. Try again.\n")
            continue

        home_team = input("Home team (3-letter code, blank to cancel this game): ").strip().upper()
        if home_team == "":
            continue
        away_team = input("Away team (3-letter code): ").strip().upper()
        home_pts_str = input("Home points: ").strip()
        away_pts_str = input("Away points: ").strip()

        try:
            home_pts = int(home_pts_str)
            away_pts = int(away_pts_str)
        except ValueError:
            print("  Points must be integers. Try again.\n")
            continue

        home_win = int(home_pts > away_pts)
        margin = abs(home_pts - away_pts)

        rows.append(
            {
                "date": date,
                "season": season,
                "home_team": home_team,
                "away_team": away_team,
                "home_pts": home_pts,
                "away_pts": away_pts,
                "home_win": home_win,
                "margin": margin,
                "source": "sim",
            }
        )

        print("  Added simulated game.\n")

    if not rows:
        print("No simulated games entered.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    MONTHLY_UPDATES_DIR.mkdir(parents=True, exist_ok=True)
    fname = MONTHLY_UPDATES_DIR / f"games_sim_{season}-{month:02d}.csv"
    df.to_csv(fname, index=False)
    print(f"Saved simulated month games to {fname}")

    return df


# ---------- Main flow ----------

if __name__ == "__main__":
    choice = menu_choice()

    if choice == "0":
        print("Exiting without importing or simulating any month.")
        raise SystemExit(0)

    try:
        season, month = prompt_season_and_month()
    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(1)

    month_label = f"{season}-{month:02d}"

    baseline_games = load_baseline_games()
    all_monthly_before = load_all_monthly_updates()

    # --- 1) Import or simulate this month ---
    if choice == "1":
        new_df = import_real_month(season, month, all_monthly_before)
        if new_df.empty:
            print("Nothing imported; exiting.")
            raise SystemExit(0)
    elif choice == "2":
        print("WARNING: including simulated games will make Elo diverge from real history.\n")
        new_df = simulate_month(season, month)
        if new_df.empty:
            print("No simulated games; exiting.")
            raise SystemExit(0)
    else:
        print("Invalid choice. Exiting.")
        raise SystemExit(1)

    # --- 2) Reload ALL monthly updates (baseline format now) ---
    all_monthly = load_all_monthly_updates()
    if all_monthly.empty:
        print("No monthly updates found after import; something went wrong.")
        raise SystemExit(1)

    # Ensure consistent dtypes
    all_monthly["date"] = pd.to_datetime(all_monthly["date"])
    all_monthly["season"] = all_monthly["season"].astype(int)

    # --- 3) Build extended dataset (baseline + all monthly updates) ---
    extended = pd.concat([baseline_games, all_monthly], ignore_index=True)
    extended = (
        extended.sort_values("date")
        .drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
        .reset_index(drop=True)
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    extended.to_csv(EXTENDED_PROCESSED, index=False)
    print(f"\nSaved extended monthly dataset to {EXTENDED_PROCESSED}")

    # --- 4) Figure out which months we’re comparing (labels for prints) ---
    # Games in the current month we just imported/simulated
    current_month_games = all_monthly[
        (all_monthly["season"] == season)
        & (all_monthly["date"].dt.month == month)
    ]

    if current_month_games.empty:
        print("No games found in all_monthly for the current month; exiting.")
        raise SystemExit(1)

    month_start = current_month_games["date"].min()
    month_end = current_month_games["date"].max()

    # All monthly updates strictly before this month’s first game
    prev_updates = all_monthly[all_monthly["date"] < month_start]

    # Label for previous cutoff
    if prev_updates.empty:
        prev_label = "baseline only (through 2025)"
    else:
        prev_months = (
            prev_updates.assign(month=prev_updates["date"].dt.month)[["season", "month"]]
            .drop_duplicates()
            .sort_values(["season", "month"])
        )
        last_prev = prev_months.iloc[-1]
        prev_label = f"{int(last_prev['season'])}-{int(last_prev['month']):02d}"

    curr_label = month_label  # current month we just processed

    # --- 5) Build previous & current extended datasets ---
    prev_extended = (
        pd.concat([baseline_games, prev_updates], ignore_index=True)
        if not prev_updates.empty
        else baseline_games.copy()
    )
    prev_extended = prev_extended.sort_values("date").reset_index(drop=True)

    curr_updates = all_monthly[all_monthly["date"] <= month_end]
    curr_extended = (
        pd.concat([baseline_games, curr_updates], ignore_index=True)
        if not curr_updates.empty
        else baseline_games.copy()
    )
    curr_extended = curr_extended.sort_values("date").reset_index(drop=True)

    # --- 6) Compute standings ---

    # Season start (after regressing baseline Elo once)
    season_start_standings = compute_season_start_standings(baseline_games)

    # Previous cutoff standings & current month standings
    prev_standings, _ = compute_standings_for_games(prev_extended)
    curr_standings, _ = compute_standings_for_games(curr_extended)

    curr_with_changes = add_changes(prev_standings, curr_standings)

    # --- 7) Print with clear labels so user knows what they’re looking at ---

    print(f"\n=== SEASON START STANDINGS (ratings entering {season}-{month:02d} season) ===")
    print(season_start_standings)

    print(f"\n=== PREVIOUS STANDINGS (through {prev_label}) ===")
    print(prev_standings)

    print(f"\n=== CURRENT STANDINGS (after games in {curr_label}) ===")
    print(curr_with_changes)

    # --- 8) Save the current month standings snapshot with deltas ---
    save_monthly_ratings_snapshot(curr_label, curr_with_changes)
