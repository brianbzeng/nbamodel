import numpy as np
import pandas as pd

import predictor


def test_daily_slate_fits_and_scores_once_using_only_prior_games(monkeypatch):
    games = pd.DataFrame(
        [
            ("2026-11-06", 2027, "BOS", "LAL", 100, 90, 1, 10),
            ("2026-11-07", 2027, "DAL", "GSW", 90, 100, 0, 10),
            ("2026-11-08", 2027, "BOS", "GSW", 130, 100, 1, 30),
            ("2026-11-09", 2027, "DAL", "LAL", 130, 100, 1, 30),
        ],
        columns=["date", "season", "home_team", "away_team", "home_pts", "away_pts", "home_win", "margin"],
    )
    games["date"] = pd.to_datetime(games["date"])
    matchups = [
        {"game_id": "first", "season": 2027, "home_team": "BOS", "away_team": "GSW"},
        {"game_id": "second", "season": 2027, "home_team": "DAL", "away_team": "LAL"},
    ]
    calls = []

    class Model:
        feature_names_in_ = ["home_games_played", "away_games_played"]

        def predict_proba(self, features):
            calls.append("score")
            assert features.to_numpy().tolist() == [[1, 1], [1, 1]]
            return np.array([[0.25, 0.75], [0.6, 0.4]])

    def fit(training):
        calls.append("fit")
        assert training["date"].tolist() == games["date"].tolist()[:2]
        return Model()

    monkeypatch.setattr(predictor, "_fit_logistic_regression", fit)

    assert predictor.predict_slate(games, matchups, None, pd.Timestamp("2026-11-08")) == {
        "first": 0.75,
        "second": 0.4,
    }
    assert calls == ["fit", "score"]
