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
    One-game Elo/Bayesian-style update.
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
    df = games.sort_values("date").reset_index(drop=True).copy()

    # Initialize ratings for all teams
    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    ratings = {team: base_rating for team in teams}

    records = []

    current_season = None

    for _, row in df.iterrows():
        season = int(row["season"])

        # At the start of a new season, regress ratings toward the mean
        if current_season is None:
            current_season = season
        elif season != current_season:
            if season_regress:
                for t in ratings:
                    ratings[t] = reg_factor * ratings[t] + (1 - reg_factor) * base_rating
            current_season = season

        home = row["home_team"]
        away = row["away_team"]
        y = int(row["home_win"])
        margin = int(row["margin"])

        # Ratings before this game
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

        records.append(
            {
                "date": row["date"],
                "season": season,
                "home_team": home,
                "away_team": away,
                "home_pts": row["home_pts"],
                "away_pts": row["away_pts"],
                "home_win": y,
                "margin": margin,
                "r_home_pre": r_home_pre,
                "r_away_pre": r_away_pre,
                "p_home_win": p_home,
                "delta": delta,
                "logloss": ll,
                "brier": br,
            }
        )

    results_df = pd.DataFrame(records)
    final_ratings = dict(ratings)  # copy

    return results_df, final_ratings
