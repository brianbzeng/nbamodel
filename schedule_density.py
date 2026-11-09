"""Count recent completed games available before each matchup date."""

from bisect import bisect_left
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd

from game_updates import GAME_KEY


DENSITY_COLUMNS = [
    "home_games_prior_4_days",
    "home_games_prior_7_days",
    "away_games_prior_4_days",
    "away_games_prior_7_days",
]
WINDOWS = (timedelta(days=4), timedelta(days=7))


def _matchup_counts(
    appearances: dict[tuple[int, str], list[date]], game_date: date, season: int, home: str, away: str,
) -> list[int]:
    counts = []
    for team in (home, away):
        team_dates = appearances.get((season, team), [])
        end = bisect_left(team_dates, game_date)
        for window in WINDOWS:
            start = bisect_left(team_dates, game_date - window, hi=end)
            counts.append(end - start)
    return counts


def build_schedule_density_features(
    games: pd.DataFrame, matchups: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build date-safe historical and slate features from the supplied finals."""
    completed = games[GAME_KEY].copy()
    completed["date"] = pd.to_datetime(completed["date"]).dt.normalize()
    completed = completed.sort_values(GAME_KEY, ignore_index=True)
    appearances: dict[tuple[int, str], list[date]] = defaultdict(list)
    for game in completed.itertuples(index=False):
        for team in (game.home_team, game.away_team):
            appearances[game.season, team].append(game.date.date())

    # A date cutoff excludes every result on the target date, regardless of input order.
    historical = [
        _matchup_counts(appearances, game.date.date(), game.season, game.home_team, game.away_team)
        for game in completed.itertuples(index=False)
    ]
    slate = [
        _matchup_counts(
            appearances, pd.Timestamp(game["game_date"]).date(),
            game["season"], game["home_team"], game["away_team"],
        )
        for game in (matchups or [])
    ]
    return (
        completed.join(pd.DataFrame(historical, columns=DENSITY_COLUMNS, dtype=int)),
        pd.DataFrame(slate, columns=DENSITY_COLUMNS, dtype=int),
    )
