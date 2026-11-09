import pandas as pd
import pytest

from elo_features import ELO_DIFF_COLUMN, build_elo_features, slate_elo_differences
from game_updates import GAME_KEY
from model import ELO_BASE_RATING, ELO_SEASON_RETENTION, update_ratings


def test_historical_elo_captures_all_matchups_before_same_date_results():
    games = pd.DataFrame(
        [
            ("2026-11-10", 2027, "BOS", "DAL", 1, 5),
            ("2026-11-09", 2027, "BOS", "NYK", 1, 20),
            ("2026-11-09", 2027, "BOS", "LAL", 1, 8),
        ],
        columns=[*GAME_KEY, "home_win", "margin"],
    )

    features, _, _ = build_elo_features(games)
    ordered = features.set_index("away_team")[ELO_DIFF_COLUMN]
    assert ordered["LAL"] == 0
    assert ordered["NYK"] == 0

    ratings = dict.fromkeys(["BOS", "NYK", "LAL"], ELO_BASE_RATING)
    update_ratings(ratings, "BOS", "LAL", 1, margin=8)
    update_ratings(ratings, "BOS", "NYK", 1, margin=20)
    assert ordered["DAL"] == pytest.approx(ratings["BOS"] - ELO_BASE_RATING)

    changed = games.copy()
    changed.loc[1, "home_win"] = 0
    changed_features, _, _ = build_elo_features(changed)
    assert changed_features[ELO_DIFF_COLUMN].iloc[:2].tolist() == [0, 0]
    assert changed_features[ELO_DIFF_COLUMN].iloc[-1] != ordered["DAL"]


def test_live_elo_uses_supplied_finals_and_regresses_once_at_season_opener():
    games = pd.DataFrame(
        [("2026-04-12", 2026, "BOS", "LAL", 1, 8)],
        columns=[*GAME_KEY, "home_win", "margin"],
    )
    _, ratings, season = build_elo_features(games)
    same_season = {"season": 2026, "home_team": "BOS", "away_team": "LAL"}
    opener = {**same_season, "season": 2027}
    original_ratings = ratings.copy()

    gap, opener_gap = slate_elo_differences(ratings, season, [same_season, opener])

    assert gap > 0
    assert opener_gap == pytest.approx(gap * ELO_SEASON_RETENTION)
    assert ratings == original_ratings

    new_season = pd.concat([
        games,
        pd.DataFrame([{"date": "2026-10-20", **opener, "home_win": 1, "margin": 5}]),
    ])
    features, ratings, season = build_elo_features(new_season)
    assert features[ELO_DIFF_COLUMN].iloc[-1] == pytest.approx(opener_gap)
    assert slate_elo_differences(ratings, season, [opener])[0] > opener_gap
