import pandas as pd
import pytest

from game_updates import GAME_KEY
from recent_form import RECENT_MARGIN_COLUMNS, build_recent_form_features


def test_recent_form_weights_signed_margins_and_resets_each_season():
    games = pd.DataFrame(
        [
            ("2026-11-01", 2027, "BOS", "LAL", 110, 100),
            ("2026-11-03", 2027, "NYK", "BOS", 105, 100),
            ("2026-11-04", 2027, "BOS", "LAL", 130, 100),
            ("2027-10-20", 2028, "BOS", "LAL", 100, 95),
        ],
        columns=[*GAME_KEY, "home_pts", "away_pts"],
    )

    history, _ = build_recent_form_features(games, half_life=1)

    assert tuple(history.columns[-3:]) == RECENT_MARGIN_COLUMNS
    assert history.loc[1, "away_recent_margin"] == 10
    assert history.loc[2, "home_recent_margin"] == 0
    assert history.loc[2, "away_recent_margin"] == -10
    assert history.loc[2, "recent_margin_diff"] == 10
    assert history.loc[3, "recent_margin_diff"] == 0


def test_same_date_finals_are_excluded_from_historical_rows_but_known_to_live_slate():
    games = pd.DataFrame(
        [
            ("2026-11-05", 2027, "BOS", "NYK", 100, 110),
            ("2026-11-05", 2027, "BOS", "LAL", 120, 100),
        ],
        columns=[*GAME_KEY, "home_pts", "away_pts"],
    )
    matchup = {"season": 2027, "home_team": "BOS", "away_team": "NYK"}

    history, slate = build_recent_form_features(games, [matchup, {**matchup, "season": 2028}], half_life=1)

    assert history["recent_margin_diff"].tolist() == [0, 0]
    assert slate.loc[0, "home_recent_margin"] == pytest.approx(0)
    assert slate.loc[0, "recent_margin_diff"] == pytest.approx(-10)
    assert slate.loc[1, "recent_margin_diff"] == 0
