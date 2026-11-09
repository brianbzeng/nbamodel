from datetime import datetime

import pandas as pd

import prediction_replay as replay


def test_replay_scores_each_date_using_only_earlier_finals(monkeypatch):
    games = pd.DataFrame(
        [
            ("2026-06-13", 2026, "NYK", "SAS"),
            ("2026-10-20", 2027, "NYK", "SAS"),
            ("2026-10-20", 2027, "BOS", "LAL"),
            ("2026-10-21", 2027, "BOS", "NYK"),
        ],
        columns=["date", "season", "home_team", "away_team"],
    )
    games["date"] = pd.to_datetime(games["date"])
    reports = pd.DataFrame({"game_date": pd.to_datetime(["2026-10-20", "2026-10-21"])})
    training_sizes = []

    def score(prior, matchups, injuries, game_date, *, model):
        assert model == "random_forest"
        assert (prior["date"] < game_date).all()
        assert (injuries["game_date"] <= game_date).all()
        training_sizes.append(len(prior))
        return {matchup["game_id"]: 0.61 for matchup in matchups}

    monkeypatch.setattr(replay, "predict_slate", score)
    result = replay.build_prediction_replay(games, reports, games[games["season"] == 2027])

    assert training_sizes == [1, 3]
    assert result["home_win_probability"].tolist() == [0.61, 0.61, 0.61]


def test_refresh_adds_only_missing_picks_and_preserves_live_records(tmp_path, monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 8, tzinfo=tz)

    games = pd.DataFrame(
        [
            ("2026-06-13", 2026, "New York Knicks", "San Antonio Spurs", 100, 91),
            ("2026-10-20", 2027, "New York Knicks", "San Antonio Spurs", 105, 90),
            ("2026-10-21", 2027, "Boston Celtics", "Los Angeles Lakers", 95, 99),
            ("2026-10-22", 2027, "Boston Celtics", "New York Knicks", 98, 100),
        ],
        columns=["date", "season", "home_team", "away_team", "home_pts", "away_pts"],
    )
    games["date"] = pd.to_datetime(games["date"])
    baseline = tmp_path / "baseline.csv"
    games.iloc[:1].to_csv(baseline, index=False)
    path = tmp_path / "replay.csv"
    path.write_text("game_date,season,home_team,away_team,home_win_probability\n2026-10-20,2027,NYK,SAS,0.63\n")
    live = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-10-21"]), "season": [2027], "home_team": ["BOS"], "away_team": ["LAL"],
        "home_win_probability": [0.6], "predicted_at_utc": pd.to_datetime(["2026-10-21T23:00Z"]),
        "tipoff_utc": pd.to_datetime(["2026-10-21T23:30Z"]),
    })
    fits = []

    def score(prior, matchups, injuries, game_date, *, model):
        fits.append(game_date)
        return {"0": 0.67}

    monkeypatch.setattr(replay, "datetime", FixedDateTime)
    monkeypatch.setattr(replay, "BASELINE_GAMES_FILE", baseline)
    monkeypatch.setattr(replay, "read_game_updates", lambda: games.iloc[1:])
    monkeypatch.setattr(replay, "read_daily_predictions", lambda: live)
    monkeypatch.setattr(replay, "_load_replay_injuries", lambda *args: None)
    monkeypatch.setattr(replay, "predict_slate", score)

    result, changed = replay.refresh_prediction_replay(path)
    assert changed
    assert result["home_win_probability"].tolist() == [0.67, 0.63]
    assert result["away_team"].tolist() == ["NYK", "SAS"]
    saved = path.read_bytes()

    assert not replay.refresh_prediction_replay(path)[1]
    assert fits == [pd.Timestamp("2026-10-22")]
    assert path.read_bytes() == saved
