from datetime import date, datetime

import pandas as pd
import pytest

import daily_predictions
from game_updates import GAME_COLUMNS


def event(game_id, tipoff, name="STATUS_SCHEDULED", state="pre", *, home="Boston Celtics", away="Chicago Bulls"):
    return {
        "id": game_id,
        "season": {"type": 2, "year": 2027},
        "competitions": [{
            "date": tipoff,
            "status": {"type": {"name": name, "state": state}},
            "competitors": [
                {"homeAway": "home", "team": {"displayName": home}, "score": "110"},
                {"homeAway": "away", "team": {"displayName": away}, "score": "100"},
            ],
        }],
    }


@pytest.fixture(autouse=True)
def no_saved_finals(monkeypatch):
    monkeypatch.setattr(daily_predictions, "read_game_updates", lambda: pd.DataFrame(columns=GAME_COLUMNS))


def pick(probability, finals=0):
    return {
        **dict.fromkeys(daily_predictions.PICK_VALUES),
        "home_win_probability": probability,
        "logistic_home_win_probability": 0.48,
        "player_impact_home_win_probability": round(probability - 0.04, 2),
        "experiment_version": daily_predictions.PLAYER_IMPACT_EXPERIMENT,
        "elo_home_win_probability": round(probability + 0.03, 2),
        "elo_experiment_version": daily_predictions.ELO_EXPERIMENT,
        "no_rest_home_win_probability": round(probability + 0.02, 2),
        "no_rest_experiment_version": daily_predictions.NO_REST_EXPERIMENT,
        "model": daily_predictions.DEFAULT_PREDICTION_MODEL,
        "same_day_finals": finals,
    }


def test_unchanged_pregame_pick_does_not_republish_and_final_pick_is_frozen(tmp_path, monkeypatch):
    path = tmp_path / "slate.csv"
    now = datetime.fromisoformat("2026-11-09T00:10:00Z")
    tipoff = "2026-11-09T00:30:00Z"
    scoreboard = [event("1", tipoff), event("2", tipoff, "STATUS_IN_PROGRESS", "in", home="Miami Heat")]
    probability = 0.61
    finals_seen = 0
    fits = []

    def predict(slate, as_of, finals):
        fits.append([game["game_id"] for game in slate])
        return {game["game_id"]: pick(probability, finals_seen) for game in slate}

    monkeypatch.setattr(daily_predictions, "_utc_now", lambda: now)
    monkeypatch.setattr(
        daily_predictions, "fetch_scoreboard",
        lambda period: scoreboard if period == "20261108" else [],
    )
    monkeypatch.setattr(daily_predictions, "_predict_games", predict)

    first, changed = daily_predictions.refresh_daily_predictions(path)
    assert changed
    assert first["game_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-08"] * 2
    assert fits == [["1"]]
    assert first.iloc[0]["home_win_probability"] == 0.61
    assert first.iloc[0]["logistic_home_win_probability"] == 0.48
    assert first.iloc[0]["player_impact_home_win_probability"] == 0.57
    assert first.iloc[0]["experiment_version"] == "rf_player_impact_v1"
    assert first.iloc[0]["elo_home_win_probability"] == 0.64
    assert first.iloc[0]["elo_experiment_version"] == "rf_elo_v1"
    assert first.iloc[0]["no_rest_home_win_probability"] == 0.63
    assert first.iloc[0]["no_rest_experiment_version"] == "rf_no_rest_v1"
    assert first.iloc[0]["model"] == "random_forest"
    assert pd.isna(first.iloc[1]["home_win_probability"])

    now = datetime.fromisoformat("2026-11-09T00:20:00Z")
    _, changed = daily_predictions.refresh_daily_predictions(path)
    assert not changed

    probability = 0.72
    updated, changed = daily_predictions.refresh_daily_predictions(path)
    assert changed
    assert updated.iloc[0]["home_win_probability"] == 0.72
    assert updated.iloc[0]["predicted_at_utc"] == now

    now = datetime.fromisoformat("2026-11-09T00:25:00Z")
    finals_seen = 1
    updated, changed = daily_predictions.refresh_daily_predictions(path)
    assert changed
    assert updated.iloc[0]["same_day_finals"] == 1
    assert updated.iloc[0]["predicted_at_utc"] == now

    now = datetime.fromisoformat("2026-11-09T03:00:00Z")
    scoreboard[0] = event("1", tipoff, "STATUS_FINAL", "post")
    probability = 0.2
    final, changed = daily_predictions.refresh_daily_predictions(path)
    assert changed
    assert final.iloc[0]["status"] == "final"
    assert final.iloc[0]["home_win_probability"] == 0.72
    assert final.iloc[0]["logistic_home_win_probability"] == 0.48
    assert final.iloc[0]["player_impact_home_win_probability"] == 0.68
    assert final.iloc[0]["elo_home_win_probability"] == 0.75
    assert final.iloc[0]["no_rest_home_win_probability"] == 0.74
    assert final.iloc[0]["predicted_at_utc"] == updated.iloc[0]["predicted_at_utc"]
    assert fits == [["1"], ["1"], ["1"], ["1"]]

    saved_bytes = path.read_bytes()

    def unavailable(period):
        raise TimeoutError("ESPN timed out")

    monkeypatch.setattr(daily_predictions, "fetch_scoreboard", unavailable)
    with pytest.raises(TimeoutError):
        daily_predictions.refresh_daily_predictions(path)
    assert path.read_bytes() == saved_bytes


def test_predictions_crossing_tipoff_during_fit_are_not_saved(tmp_path, monkeypatch):
    path = tmp_path / "slate.csv"
    before_tipoff = datetime.fromisoformat("2026-11-09T00:29:59Z")
    at_tipoff = datetime.fromisoformat("2026-11-09T00:30:00Z")
    clock = iter([before_tipoff, before_tipoff, at_tipoff])
    monkeypatch.setattr(daily_predictions, "_utc_now", lambda: next(clock))
    monkeypatch.setattr(
        daily_predictions, "fetch_scoreboard",
        lambda period: [event("1", "2026-11-09T00:30:00Z")] if period == "20261108" else [],
    )
    monkeypatch.setattr(daily_predictions, "_predict_games", lambda slate, as_of, finals: {"1": pick(0.61)})

    snapshot, changed = daily_predictions.refresh_daily_predictions(path)

    assert changed
    assert len(snapshot) == 1
    assert snapshot["home_win_probability"].isna().all()
    assert snapshot["predicted_at_utc"].isna().all()


def test_legacy_early_picks_are_removed_for_scheduled_games_only(tmp_path, monkeypatch):
    path = tmp_path / "slate.csv"
    path.write_text(
        "game_id,game_date,tipoff_utc,season,home_team,away_team,status,home_win_probability,predicted_at_utc\n"
        "1,2026-11-08,2026-11-09T02:00:00Z,2027,BOS,CHI,scheduled,0.68,2026-11-08T06:00:00Z\n"
        "2,2026-11-08,2026-11-08T23:00:00Z,2027,MIA,CHI,in_progress,0.71,2026-11-08T18:00:00Z\n",
        encoding="utf-8",
    )
    now = datetime.fromisoformat("2026-11-09T00:59:59Z")
    events = [
        event("1", "2026-11-09T02:00:00Z"),
        event("2", "2026-11-08T23:00:00Z", "STATUS_IN_PROGRESS", "in", home="Miami Heat"),
        event("3", "2026-11-10T00:00:00Z"),
    ]
    monkeypatch.setattr(daily_predictions, "_utc_now", lambda: now)
    monkeypatch.setattr(daily_predictions, "fetch_scoreboard", lambda period: events)
    monkeypatch.setattr(daily_predictions, "_predict_games", lambda *args: {"1": pick(0.69)})

    assert not daily_predictions.pregame_window_open()
    snapshot, changed = daily_predictions.refresh_daily_predictions(path)
    assert changed
    rows = snapshot.set_index("game_id")
    assert rows.loc["1", "home_win_probability"] == 0.69
    assert rows.loc["1", "predicted_at_utc"] == now.replace(microsecond=0)
    assert rows.loc["2", "home_win_probability"] == 0.71
    assert rows.loc["2", "model"] == "logistic"
    assert pd.isna(rows.loc["2", "logistic_home_win_probability"])
    assert pd.isna(rows.loc["2", "same_day_finals"])
    assert pd.isna(rows.loc["3", "home_win_probability"])

    now = datetime.fromisoformat("2026-11-09T01:00:00Z")
    assert daily_predictions.pregame_window_open()


def test_entire_next_pacific_slate_is_ready_before_midnight(tmp_path, monkeypatch):
    path = tmp_path / "slate.csv"
    now = datetime.fromisoformat("2026-11-09T07:29:59Z")
    events = [
        event("early", "2026-11-09T20:00:00Z"),
        event("late", "2026-11-10T06:00:00Z", home="Miami Heat", away="Orlando Magic"),
        event("future", "2026-11-10T23:00:00Z"),
    ]
    calls = []

    def predict(matchups, as_of, finals):
        game_ids = [matchup["game_id"] for matchup in matchups]
        calls.append(game_ids)
        return {game_id: pick(0.6) for game_id in game_ids}

    monkeypatch.setattr(daily_predictions, "_utc_now", lambda: now)
    monkeypatch.setattr(daily_predictions, "fetch_scoreboard", lambda period: events)
    monkeypatch.setattr(daily_predictions, "_predict_games", predict)

    snapshot, _ = daily_predictions.refresh_daily_predictions(path)
    assert snapshot["home_win_probability"].isna().all()
    assert calls == []

    now = datetime.fromisoformat("2026-11-09T07:35:00Z")
    assert not daily_predictions.pregame_window_open()  # The fast refresh is still limited to the last hour.
    snapshot, changed = daily_predictions.refresh_daily_predictions(path)
    assert changed and calls == [["early", "late"]]
    assert snapshot.loc[snapshot["home_win_probability"].notna(), "game_id"].tolist() == ["early", "late"]

    now = datetime.fromisoformat("2026-11-09T08:00:00Z")
    snapshot, changed = daily_predictions.refresh_daily_predictions(path)
    assert not changed
    assert snapshot["home_win_probability"].count() == 2


def test_final_scores_include_fresh_corrections_but_exclude_future_games(monkeypatch):
    now = datetime.fromisoformat("2026-11-09T00:30:00Z")
    events = [
        event("1", "2026-11-08T22:00:00Z", "STATUS_FINAL", "post"),
        event("2", "2026-11-09T01:00:00Z", "STATUS_FINAL", "post", home="Miami Heat"),
    ]
    saved = pd.DataFrame(
        [
            (pd.Timestamp("2026-11-08"), 2027, "Boston Celtics", "Chicago Bulls", 105, 100),
            (pd.Timestamp("2026-11-08"), 2027, "Miami Heat", "Chicago Bulls", 105, 100),
            (pd.Timestamp("2026-11-08"), 2027, "San Antonio Spurs", "Dallas Mavericks", 118, 100),
            (pd.Timestamp("2026-11-09"), 2027, "Boston Celtics", "Chicago Bulls", 110, 100),
        ],
        columns=GAME_COLUMNS,
    )
    monkeypatch.setattr(daily_predictions, "read_game_updates", lambda: saved)
    slate = daily_predictions.scheduled_games(events, date(2026, 11, 8), date(2026, 11, 9))

    finals = daily_predictions._final_games(events, slate, now)

    assert finals[["home_team", "home_pts"]].to_records(index=False).tolist() == [("SAS", 118), ("BOS", 110)]


@pytest.mark.parametrize("experiment_available", [True, False])
def test_daily_pick_keeps_the_baseline_and_records_player_impact_separately(monkeypatch, experiment_available):
    as_of = datetime.fromisoformat("2026-11-09T00:15:00Z")
    slate = daily_predictions.scheduled_games(
        [event("1", "2026-11-09T00:30:00Z")], date(2026, 11, 8), date(2026, 11, 8)
    )
    day = slate[0]["game_date"]
    reports = pd.DataFrame([{
        "game_date": day, "team": "BOS", "latest_report_datetime": pd.Timestamp("2026-11-08 19:00"),
        "out_count": 1,
    }])
    monkeypatch.setattr("predictor.load_optional_team_injuries", lambda games, *, as_of: reports)
    def score(*args, models, optional_models, **kwargs):
        assert models == ("random_forest", "logistic", "random_forest_elo", "random_forest_no_rest")
        assert optional_models == ("random_forest_elo", "random_forest_no_rest")
        return {
            "random_forest": {"1": 0.55}, "logistic": {"1": 0.42},
            "random_forest_elo": {"1": 0.53}, "random_forest_no_rest": {"1": 0.57},
        }, {"1": (0, 2)}

    monkeypatch.setattr("predictor.predict_slate_models", score)

    def adjust(matchups, probabilities, team_reports):
        assert matchups == slate
        assert probabilities == {"1": 0.55}
        assert team_reports is reports
        if not experiment_available:
            raise ValueError("Invalid player box scores")
        return {"1": 0.51}

    monkeypatch.setattr("player_impact.player_adjusted_picks", adjust)

    pick = daily_predictions._predict_games(slate, as_of, pd.DataFrame(columns=["game_date"]))["1"]

    assert pick["home_win_probability"] == 0.55
    assert pick["logistic_home_win_probability"] == 0.42
    assert pick["model"] == "random_forest"
    assert pick["home_player_impact_pp"] is None
    assert pick["player_impact_home_win_probability"] == (0.51 if experiment_available else None)
    assert pick["experiment_version"] == ("rf_player_impact_v1" if experiment_available else None)
    assert pick["elo_home_win_probability"] == 0.53
    assert pick["elo_experiment_version"] == "rf_elo_v1"
    assert pick["no_rest_home_win_probability"] == 0.57
    assert pick["no_rest_experiment_version"] == "rf_no_rest_v1"
    assert pick["home_injury_report_utc"] == pd.Timestamp("2026-11-09T00:00:00Z")
    assert (pick["home_rest_days"], pick["away_rest_days"]) == (0, 2)
