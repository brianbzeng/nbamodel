import pandas as pd

from game_updates import GAME_KEY
from schedule_density import DENSITY_COLUMNS, build_schedule_density_features


def test_historical_density_counts_exact_window_and_excludes_target_date():
    games = pd.DataFrame(
        [
            ("2026-11-08", 2027, "BOS", "DAL"),
            ("2026-11-08", 2027, "BOS", "CHI"),
            ("2026-11-04", 2027, "DET", "BOS"),
            ("2026-11-01", 2027, "BOS", "DAL"),
            ("2026-10-31", 2027, "BOS", "PHI"),
        ],
        columns=GAME_KEY,
    )

    features, _ = build_schedule_density_features(games)
    november_8 = features.loc[features["date"].eq(pd.Timestamp("2026-11-08"))].set_index("away_team")

    assert november_8.loc["CHI", DENSITY_COLUMNS].tolist() == [1, 2, 0, 0]
    assert november_8.loc["DAL", DENSITY_COLUMNS].tolist() == [1, 2, 0, 1]


def test_slate_density_excludes_same_day_and_future_finals_and_resets_each_season():
    games = pd.DataFrame(
        [
            ("2026-10-18", 2026, "BOS", "DAL"),
            ("2026-10-19", 2027, "BOS", "CHI"),
            ("2026-10-20", 2027, "DAL", "DET"),
            ("2026-10-21", 2027, "BOS", "DAL"),
        ],
        columns=GAME_KEY,
    )
    matchups = [
        {"game_date": pd.Timestamp(day), "season": 2027, "home_team": "BOS", "away_team": "DAL"}
        for day in ("2026-10-20", "2026-10-19", "2026-10-22")
    ]

    _, slate = build_schedule_density_features(games, matchups)

    assert slate.values.tolist() == [[1, 1, 0, 0], [0, 0, 0, 0], [2, 2, 2, 2]]
