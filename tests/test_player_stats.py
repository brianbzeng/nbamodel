from datetime import date

import pandas as pd
import pytest
import requests

import player_stats
from game_updates import completed_games


def _final_event():
    return {
        "id": "401909945",
        "season": {"year": 2027, "type": 2},
        "competitions": [{
            "date": "2026-11-09T01:30:00Z",
            "status": {"type": {"name": "STATUS_FINAL"}},
            "competitors": [
                {"homeAway": "away", "team": {"displayName": "Philadelphia 76ers"}, "score": "127"},
                {"homeAway": "home", "team": {"displayName": "Chicago Bulls"}, "score": "121"},
            ],
        }],
    }


def _summary(event):
    keys = [
        "minutes", "points", "fieldGoalsMade-fieldGoalsAttempted", "freeThrowsMade-freeThrowsAttempted",
        "offensiveRebounds", "defensiveRebounds", "steals", "assists", "blocks", "fouls", "turnovers",
    ]
    line = ["33:30", "20", "7-13", "5-6", "1", "3", "2", "3", "1", "2", "3"]
    teams = []
    for offset, team in enumerate(event["competitions"][0]["competitors"]):
        athletes = [
            {
                "athlete": {"id": str(100 + offset * 10 + i), "displayName": f"Test Player {i}"},
                "stats": line,
                "didNotPlay": i == 5,
            }
            for i in range(6)
        ]
        teams.append({"team": team["team"], "statistics": [{"keys": keys, "athletes": athletes}]})
    return {"header": event, "boxscore": {"players": teams}}


def test_final_box_score_uses_local_game_date_and_excludes_dnp():
    event = _final_event()
    results = completed_games([event], date(2026, 11, 8), date(2026, 11, 8))
    game = player_stats._canonical_games(results).iloc[0].to_dict()
    summary = _summary(event)
    players = player_stats._summary_rows(summary, game)

    assert len(players) == 10
    assert players["date"].unique().tolist() == [pd.Timestamp("2026-11-08")]
    assert set(players["team"]) == {"CHI", "PHI"}
    assert players.iloc[0]["minutes"] == 33.5
    assert players.iloc[0]["game_score"] == pytest.approx(15.9)

    event["competitions"][0]["status"]["type"]["name"] = "STATUS_IN_PROGRESS"
    with pytest.raises(ValueError, match="not final"):
        player_stats._summary_rows(summary, game)


def test_incremental_snapshot_retries_missing_box_score_then_becomes_noop(tmp_path, monkeypatch):
    event = _final_event()
    results = completed_games([event], date(2026, 11, 8), date(2026, 11, 8))
    monkeypatch.setattr(player_stats, "read_game_updates", lambda: results)
    scoreboard_calls = []

    def scoreboard(period):
        scoreboard_calls.append(period)
        return [event]

    responses = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return _summary(event)

    def get(url, params, timeout):
        assert url == player_stats.SUMMARY_URL
        assert params == {"event": "401909945"}
        responses.append(url)
        if len(responses) == 1:
            raise requests.Timeout()
        return Response()

    monkeypatch.setattr(player_stats, "fetch_scoreboard", scoreboard)
    monkeypatch.setattr(player_stats.requests, "get", get)
    snapshot = tmp_path / "player-stats.csv"

    rows, missing = player_stats.refresh_player_stats(snapshot)
    assert rows.empty and missing == 1
    assert not snapshot.exists()

    rows, missing = player_stats.refresh_player_stats(snapshot)
    assert len(rows) == 10 and missing == 0
    assert len(responses) == 2
    assert sorted(scoreboard_calls) == ["202611", "202611", "20261108", "20261108"]
    saved = snapshot.read_bytes()
    monkeypatch.setattr(player_stats, "write_snapshot", lambda *args, **kwargs: pytest.fail("Rewrote snapshot"))
    repeated, missing = player_stats.refresh_player_stats(snapshot)
    assert missing == 0
    pd.testing.assert_frame_equal(rows, repeated)
    assert len(responses) == 2
    assert snapshot.read_bytes() == saved
