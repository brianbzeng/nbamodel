"""Compute the Elo gap available before each training or prediction matchup."""

from collections import defaultdict

import pandas as pd

from game_updates import GAME_KEY
from model import ELO_BASE_RATING, ELO_SEASON_RETENTION, update_ratings


ELO_DIFF_COLUMN = "elo_diff"


def build_elo_features(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], int | None]:
    """Capture ratings before each date and advance them using that day's results."""
    if games.empty:
        return pd.DataFrame(columns=[*GAME_KEY, ELO_DIFF_COLUMN]), {}, None

    completed = games[[*GAME_KEY, "home_win", "margin"]].copy()
    completed["date"] = pd.to_datetime(completed["date"]).dt.normalize()
    completed = completed.sort_values(GAME_KEY, ignore_index=True)
    ratings = defaultdict(lambda: ELO_BASE_RATING)
    current_season = None
    gaps = []
    for _, slate in completed.groupby("date", sort=False):
        season = int(slate["season"].iat[0])
        if not slate["season"].eq(season).all():
            raise ValueError("Completed games on one date must belong to the same season.")
        if current_season is not None and current_season != season:
            for team, rating in ratings.items():
                ratings[team] = (
                    ELO_SEASON_RETENTION * rating + (1.0 - ELO_SEASON_RETENTION) * ELO_BASE_RATING
                )
        current_season = season

        # The ledger contains dates, so no other result on the date enters a historical feature.
        start_of_day = ratings.copy()
        for game in slate.itertuples(index=False):
            gaps.append(start_of_day[game.home_team] - start_of_day[game.away_team])
            margin = 0 if pd.isna(game.margin) else int(game.margin)
            update_ratings(ratings, game.home_team, game.away_team, int(game.home_win), margin=margin)

    return completed[GAME_KEY].assign(**{ELO_DIFF_COLUMN: gaps}), dict(ratings), current_season


def slate_elo_differences(
    final_ratings: dict[str, float], final_season: int | None, matchups: list[dict[str, object]],
) -> list[float]:
    """Score matchups from known finals and regress at an unplayed season opener."""
    gaps = []
    for matchup in matchups:
        season = int(matchup["season"])
        if final_season is not None and season < final_season:
            raise ValueError("Cannot compute pregame Elo from a later season's results.")
        home = final_ratings.get(matchup["home_team"], ELO_BASE_RATING)
        away = final_ratings.get(matchup["away_team"], ELO_BASE_RATING)
        retention = ELO_SEASON_RETENTION if final_season is not None and season > final_season else 1.0
        gaps.append((home - away) * retention)
    return gaps
