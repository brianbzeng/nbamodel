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
        "1,2026-11-08,2026-11-09T02:00:00Z,2027,BOS,CHI,scheduled,0.68,2026-11-08T21:00:00Z\n"
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

    assert not daily_predictions.pregame_window_open()
    snapshot, changed = daily_predictions.refresh_daily_predictions(path)
    assert changed
    rows = snapshot.set_index("game_id")
    assert pd.isna(rows.loc["1", "home_win_probability"])
    assert rows.loc["2", "home_win_probability"] == 0.71
    assert rows.loc["2", "model"] == "logistic"
    assert pd.isna(rows.loc["2", "same_day_finals"])
    assert pd.isna(rows.loc["3", "home_win_probability"])

    now = datetime.fromisoformat("2026-11-09T01:00:00Z")
    assert daily_predictions.pregame_window_open()


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
