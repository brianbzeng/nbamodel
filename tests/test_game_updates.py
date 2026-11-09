from datetime import date, datetime

import pytest
import requests

import game_updates


def event(day, home_score=121, status="STATUS_FINAL", season_type=2, kind="STD"):
    return {
        "id": str(day),
        "season": {"year": 2027, "type": season_type},
        "competitions": [{
            "date": f"2026-11-{day:02d}T02:00:00Z",
            "type": {"abbreviation": kind},
            "status": {"type": {"name": status}},
            "competitors": [
                {"homeAway": "away", "team": {"displayName": "Philadelphia 76ers"}, "score": "127"},
                {"homeAway": "home", "team": {"displayName": "Chicago Bulls"}, "score": str(home_score)},
            ],
        }],
    }


def test_only_final_nba_games_enter_the_snapshot():
    events = [
        event(2),
        event(3, status="STATUS_IN_PROGRESS"),
        event(4, season_type=1),
        event(5, kind="ALLSTAR"),
        event(6, season_type=5),
        event(7, season_type=3),
        event(12),
    ]

    games = game_updates.completed_games(events, date(2026, 11, 1), date(2026, 11, 8))

    assert games["date"].dt.day.tolist() == [1, 5, 6]
    assert games["home_team"].tolist() == ["Chicago Bulls"] * 3


def test_refresh_merges_corrections_and_retains_snapshot_on_failure(tmp_path, monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 8, 23, tzinfo=tz)

    path = tmp_path / "games.csv"
    monkeypatch.setattr(game_updates, "datetime", FixedDateTime)
    monkeypatch.setattr(game_updates, "FIRST_UPDATE_DATE", date(2026, 11, 1))
    monthly = [event(2), event(9)]
    daily = [event(9, home_score=122)]
    monkeypatch.setattr(game_updates, "fetch_scoreboard", lambda period: monthly if len(period) == 6 else daily)

    games, changed = game_updates.refresh_game_updates(path)
    assert changed
    assert games["home_pts"].tolist() == [121, 122]
    assert not game_updates.refresh_game_updates(path)[1]

    monthly.clear()
    daily[:] = [event(9, home_score=123)]
    games, changed = game_updates.refresh_game_updates(path)
    assert changed
    assert games["home_pts"].tolist() == [121, 123]
    saved_snapshot = path.read_bytes()

    def unavailable(period):
        raise requests.Timeout(period)

    monkeypatch.setattr(game_updates, "fetch_scoreboard", unavailable)
    with pytest.raises(requests.Timeout):
        game_updates.refresh_game_updates(path)
    assert path.read_bytes() == saved_snapshot
