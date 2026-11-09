"""Build pregame scoring margins that decay by team games played."""

import math

import pandas as pd

from game_updates import GAME_KEY


RECENT_MARGIN_COLUMNS = ("home_recent_margin", "away_recent_margin", "recent_margin_diff")
FORM_HALF_LIFE = 7


def build_recent_form_features(
    games: pd.DataFrame,
    matchups: list[dict[str, object]] | None = None,
    *,
    half_life: float = FORM_HALF_LIFE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use earlier dates for training and supplied known finals for the slate."""
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError("Recent form half-life must be positive and finite.")

    decay = 0.5 ** (1 / half_life)
    state: dict[tuple[int, str], tuple[float, float]] = {}
    history = []

    def margin_for(season: int, team: str) -> float:
        total, weight = state.get((season, team), (0.0, 0.0))
        return total / weight if weight else 0.0

    def matchup_margins(season: int, home: str, away: str) -> tuple[float, float, float]:
        home_margin = margin_for(season, home)
        away_margin = margin_for(season, away)
        return home_margin, away_margin, home_margin - away_margin

    if not games.empty:
        completed = games[[*GAME_KEY, "home_pts", "away_pts"]].copy()
        completed["date"] = pd.to_datetime(completed["date"]).dt.normalize()
        completed = completed.sort_values(GAME_KEY, ignore_index=True)
        for date, day in completed.groupby("date", sort=False):
            for game in day.itertuples(index=False):
                history.append((
                    date, game.season, game.home_team, game.away_team,
                    *matchup_margins(game.season, game.home_team, game.away_team),
                ))
            for game in day.itertuples(index=False):
                margin = game.home_pts - game.away_pts
                for team, signed_margin in ((game.home_team, margin), (game.away_team, -margin)):
                    key = (game.season, team)
                    total, weight = state.get(key, (0.0, 0.0))
                    state[key] = (decay * total + signed_margin, decay * weight + 1)

    slate = [
        matchup_margins(int(game["season"]), str(game["home_team"]), str(game["away_team"]))
        for game in (matchups or [])
    ]
    return (
        pd.DataFrame(history, columns=[*GAME_KEY, *RECENT_MARGIN_COLUMNS]),
        pd.DataFrame(slate, columns=RECENT_MARGIN_COLUMNS),
    )
