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


def test_daily_slate_uses_verified_finals_and_latest_matchup_reports(monkeypatch):
    day = pd.Timestamp("2026-11-08")
    prior_day = pd.Timestamp("2026-11-07")
    next_day = pd.Timestamp("2026-11-09")
    columns = ["date", "season", "home_team", "away_team", "home_pts", "away_pts", "home_win", "margin"]
    games = pd.DataFrame(
        [
            (prior_day, 2027, "BOS", "LAL", 100, 90, 1, 10),
            (prior_day, 2027, "DAL", "GSW", 90, 100, 0, 10),
            (day, 2027, "NYK", "PHI", 100, 90, 1, 10),
        ],
        columns=columns,
    )
    finals = pd.DataFrame(
        [
            (day, 2027, "BOS", "DET", 101, 100, 1, 1),
            (day, 2027, "BOS", "DET", 90, 110, 0, 20),
            (day, 2027, "BOS", "LAL", 110, 100, 1, 10),
            (next_day, 2027, "DAL", "LAL", 110, 100, 1, 10),
        ],
        columns=columns,
    )
    reports = pd.DataFrame(
        [
            (prior_day, "BOS", "Boston player", "Out", "Illness", pd.Timestamp("2026-11-07 15:00")),
            (day, "LAL", "Lakers player", "Questionable", "Illness", pd.Timestamp("2026-11-08 15:00")),
            (day, "LAL", "Lakers player", "Out", "Illness", pd.Timestamp("2026-11-08 17:00")),
        ],
        columns=["game_date", "team", "player_name", "status", "reason", "report_datetime"],
    )
    injuries = predictor.aggregate_team_injuries(reports)

    class Model:
        feature_names_in_ = [
            "home_games_played", "away_games_played", "home_win_pct",
            "home_injury_report_available", "away_injury_report_available", "home_out_count", "away_out_count",
        ]

        def predict_proba(self, features):
            assert features.iloc[0].tolist() == [2, 1, 0.5, 0, 1, 0, 1]
            return np.array([[0.25, 0.75]])

    def fit(training):
        assert training["date"].tolist() == [prior_day, prior_day, day]
        assert training.iloc[-1][["home_team", "away_team", "home_win"]].tolist() == ["BOS", "DET", 0]
        return Model()

    monkeypatch.setattr(predictor, "_fit_logistic_regression", fit)
    matchups = [{"game_id": "target", "season": 2027, "home_team": "BOS", "away_team": "LAL"}]

    assert predictor.predict_slate(games, matchups, injuries, day, same_day_finals=finals) == {"target": 0.75}


def test_injury_loader_uses_latest_report_available_at_the_cutoff(tmp_path, monkeypatch):
    csv = tmp_path / "official_nba_injuries_by_team_live.csv"
    csv.write_text(
        "game_date,team,latest_report_datetime,out_count\n"
        "2026-11-08,LAL,2026-11-08 17:00:00,0\n"
        "2026-11-08,LAL,2026-11-08 17:45:00,1\n"
    )
    monkeypatch.setattr(predictor, "RAW_DIR", tmp_path)
    monkeypatch.setattr(predictor, "REFERENCE_DIR", tmp_path)
    games = pd.DataFrame({"season": [2027]})

    earlier = predictor.load_optional_team_injuries(games, as_of=pd.Timestamp("2026-11-08 17:30:00"))
    latest = predictor.load_optional_team_injuries(games, as_of=pd.Timestamp("2026-11-08 17:45:00"))

    assert earlier["out_count"].tolist() == [0]
    assert latest["out_count"].tolist() == [1]
