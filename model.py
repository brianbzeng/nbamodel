# model.py

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def win_prob(r_home, r_away, hca=50, scale=400):
    """
    Compute probability home team wins given ratings and home-court advantage.
    """
    return sigmoid((r_home - r_away + hca) / scale)


def update_ratings(ratings, home, away, y, k=20, hca=50, scale=400, margin=None):
    """
    One-game Elo-style update.
    ratings: dict[team -> rating]
    home, away: team codes (e.g. 'GSW')
    y: 1 if home won, 0 if home lost
    margin: optional point differential magnitude (int)
    """
    rh, ra = ratings[home], ratings[away]
    p = win_prob(rh, ra, hca=hca, scale=scale)

    mult = 1.0
    if margin is not None:
        # margin-of-victory multiplier
        mult = np.log(margin + 1)

    delta = k * mult * (y - p)
    ratings[home] += delta
    ratings[away] -= delta
    return p, delta


def brier(y, p):
    return (y - p) ** 2


def logloss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run_elo(
    games: pd.DataFrame,
    base_rating: float = 1500.0,
    k: float = 20.0,
    hca: float = 50.0,
    scale: float = 400.0,
    use_margin: bool = True,
    season_regress: bool = True,
    reg_factor: float = 0.75,
):
    """
    Run Elo-like updates over multiple seasons of games.

    games: DataFrame with columns:
        ['date', 'season', 'home_team', 'away_team', 'home_pts', 'away_pts',
         'home_win', 'margin']

    returns:
        results_df: per-game DataFrame with predictions & metrics
        final_ratings: dict[team -> rating] after last game
    """
    if games.empty:
        raise ValueError("run_elo requires at least one game.")

    # Copy only the columns we need, in sorted order. Sorting a narrow frame is
    # cheaper than sorting the full DataFrame the caller hands us.
    needed = ["date", "season", "home_team", "away_team", "home_pts", "away_pts", "home_win", "margin"]
    missing = [c for c in needed if c not in games.columns]
    if missing:
        raise ValueError(f"run_elo is missing required columns: {missing}")

    df = games[needed].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Pre-extract columns as numpy arrays for fast per-row access.
    seasons = df["season"].to_numpy()
    home_teams = df["home_team"].to_numpy()
    away_teams = df["away_team"].to_numpy()
    home_pts = df["home_pts"].to_numpy()
    away_pts = df["away_pts"].to_numpy()
    home_wins = df["home_win"].to_numpy()
    margins = df["margin"].to_numpy()
    dates = df["date"].to_numpy()

    # Initialize ratings for all teams.
    teams = sorted(set(home_teams.tolist()).union(away_teams.tolist()))
    ratings = {team: base_rating for team in teams}

    # Pre-allocate record arrays to avoid per-row dict allocation and list
    # growth during the hot loop. Building a DataFrame from aligned arrays at
    # the end is much faster than from a list of dicts.
    n = len(df)
    record_r_home = np.empty(n)
    record_r_away = np.empty(n)
    record_p_home = np.empty(n)
    record_delta = np.empty(n)
    record_logloss = np.empty(n)
    record_brier = np.empty(n)
    record_margin = np.empty(n, dtype=int)

    current_season = None

    for i in range(n):
        season = int(seasons[i])

        # At the start of a new season, regress ratings toward the mean
        if current_season is None:
            current_season = season
        elif season != current_season:
            if season_regress:
                for t in ratings:
                    ratings[t] = reg_factor * ratings[t] + (1.0 - reg_factor) * base_rating
            current_season = season

        home = home_teams[i]
        away = away_teams[i]
        y = int(home_wins[i])
        margin_value = margins[i]
        margin = 0 if pd.isna(margin_value) else int(margin_value)

        # Ratings before this game.
        r_home_pre = ratings[home]
        r_away_pre = ratings[away]

        # Update
        p_home, delta = update_ratings(
            ratings,
            home,
            away,
            y,
            k=k,
            hca=hca,
            scale=scale,
            margin=margin if use_margin else None,
        )

        # Metrics
        ll = logloss(y, p_home)
        br = brier(y, p_home)

        record_r_home[i] = r_home_pre
        record_r_away[i] = r_away_pre
        record_p_home[i] = p_home
        record_delta[i] = delta
        record_logloss[i] = ll
        record_brier[i] = br
        record_margin[i] = margin

    results_df = pd.DataFrame(
        {
            "date": dates,
            "season": seasons,
            "home_team": home_teams,
            "away_team": away_teams,
            "home_pts": home_pts,
            "away_pts": away_pts,
            "home_win": home_wins,
            "margin": record_margin,
            "r_home_pre": record_r_home,
            "r_away_pre": record_r_away,
            "p_home_win": record_p_home,
            "delta": record_delta,
            "logloss": record_logloss,
            "brier": record_brier,
        }
    )
    final_ratings = dict(ratings)  # copy

    return results_df, final_ratings
