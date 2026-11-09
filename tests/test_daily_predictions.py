from datetime import datetime

import pandas as pd
import pytest

import daily_predictions


def event(game_id, tipoff, name="STATUS_SCHEDULED", state="pre"):
    return {
        "id": game_id,
        "season": {"type": 2, "year": 2027},
        "competitions": [{
            "date": tipoff,
            "status": {"type": {"name": name, "state": state}},
            "competitors": [
                {"homeAway": "home", "team": {"displayName": "Boston Celtics"}},
                {"homeAway": "away", "team": {"displayName": "Chicago Bulls"}},
            ],
        }],
    }


def test_unchanged_pregame_pick_does_not_republish_and_final_pick_is_frozen(tmp_path, monkeypatch):
    path = tmp_path / "slate.csv"
    now = datetime.fromisoformat("2026-11-09T00:10:00Z")
    tipoff = "2026-11-09T00:30:00Z"
    scoreboard = [event("1", tipoff), event("2", tipoff, "STATUS_IN_PROGRESS", "in")]
    probability = 0.61
    fits = []

    def predict(slate, as_of):
        fits.append([game["game_id"] for game in slate])
        return {game["game_id"]: probability for game in slate}

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
    assert pd.isna(first.iloc[1]["home_win_probability"])

    now = datetime.fromisoformat("2026-11-09T00:20:00Z")
    _, changed = daily_predictions.refresh_daily_predictions(path)
    assert not changed

    probability = 0.72
    updated, changed = daily_predictions.refresh_daily_predictions(path)
    assert changed
    assert updated.iloc[0]["home_win_probability"] == 0.72
    assert updated.iloc[0]["predicted_at_utc"] == now

    now = datetime.fromisoformat("2026-11-09T03:00:00Z")
    scoreboard[0] = event("1", tipoff, "STATUS_FINAL", "post")
    probability = 0.2
    final, changed = daily_predictions.refresh_daily_predictions(path)
    assert changed
    assert final.iloc[0]["status"] == "final"
    assert final.iloc[0]["home_win_probability"] == 0.72
    assert final.iloc[0]["predicted_at_utc"] == updated.iloc[0]["predicted_at_utc"]
    assert fits == [["1"], ["1"], ["1"]]

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
    monkeypatch.setattr(daily_predictions, "_predict_games", lambda slate, as_of: {"1": 0.61})

    snapshot, changed = daily_predictions.refresh_daily_predictions(path)

    assert changed
    assert len(snapshot) == 1
    assert snapshot["home_win_probability"].isna().all()
    assert snapshot["predicted_at_utc"].isna().all()
