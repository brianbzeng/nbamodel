import numpy as np
import pandas as pd

from player_impact import (
    adjust_win_probabilities,
    matchup_player_losses,
    player_loss_difference,
    rate_reported_players,
    team_player_losses,
)


def test_player_rating_uses_only_earlier_box_scores_and_follows_traded_players():
    games = pd.DataFrame(
        [
            ("2026-11-06", "DAL", 1, "Luka Doncic", 36, 28),
            ("2026-11-06", "LAL", 2, "Reserve Player Jr.", 18, 5),
            ("2026-11-08", "LAL", 1, "Luka Doncic", 36, 99),
            ("2026-11-09", "LAL", 1, "Luka Doncic", 36, 99),
        ],
        columns=["date", "team", "player_id", "player", "minutes", "game_score"],
    )
    reports = pd.DataFrame(
        [("Dončić, Luka", "LAL"), ("Player, Reserve", "LAL"), ("Unknown, Player", "DET")],
        columns=["player_name", "team"],
    ).assign(game_date="2026-11-08")

    values = rate_reported_players(games, reports)["player_value"]

    assert values.iloc[0] == 21 / 9
    assert values.iloc[1] == 1.5 / 9
    assert np.isnan(values.iloc[2])


def test_latest_report_replaces_previous_players_and_impact_respects_status():
    games = pd.DataFrame(
        [("2026-11-06", "DAL", 1, "Star Player", 36, 25)],
        columns=["date", "team", "player_id", "player", "minutes", "game_score"],
    )
    reports = pd.DataFrame(
        [
            ("DAL", "Player, Star", "Out", "2026-11-08 15:00"),
            ("DAL", "Player, Star", "Questionable", "2026-11-08 16:00"),
            ("BRK", "Player, Star", "Out", "2026-11-08 15:00"),
            ("BRK", "", "", "2026-11-08 16:00"),
        ],
        columns=["team", "player_name", "status", "report_datetime"],
    ).assign(game_date="2026-11-08")
    matchups = pd.DataFrame([
        {"date": "2026-11-08", "home_team": "DAL", "away_team": "BRK"},
        {"date": "2026-11-08", "home_team": "BRK", "away_team": "DAL"},
        {"date": "2026-11-09", "home_team": "BRK", "away_team": "DAL"},
    ])

    difference = player_loss_difference(matchups, team_player_losses(games, reports))
    probabilities = adjust_win_probabilities([0.5] * 3, difference, weight=0.04)

    np.testing.assert_allclose(difference, [-0.9, 0.9, 0.0])
    assert probabilities[0] < 0.5 < probabilities[1]
    assert probabilities[2] == 0.5


def test_matchup_uses_player_statuses_from_the_same_reports_as_team_inputs(monkeypatch):
    games = pd.DataFrame(
        [("2026-11-06", "DAL", 1, "Star Player", 36, 25)],
        columns=["date", "team", "player_id", "player", "minutes", "game_score"],
    )
    day = pd.Timestamp("2026-11-08")
    report_time = pd.Timestamp("2026-11-08 15:00")
    matchups = pd.DataFrame([{"game_date": day, "home_team": "DAL", "away_team": "BRK"}])
    report_times = pd.DataFrame([{"game_date": day, "team": "DAL", "latest_report_datetime": report_time}])
    details = pd.DataFrame([
        {"game_date": day, "team": "DAL", "player_name": "Player, Star", "status": "Out",
         "report_datetime": report_time},
        {"game_date": day, "team": "DAL", "player_name": "Player, Star", "status": "Available",
         "report_datetime": pd.Timestamp("2026-11-08 15:30")},
    ])
    monkeypatch.setattr("player_injuries.read_live_player_injury_reports", lambda: details)
    monkeypatch.setattr("player_stats.read_player_game_stats", lambda: games)

    assert matchup_player_losses(matchups, report_times).tolist() == [-2.0]

    report_times["latest_report_datetime"] = pd.Timestamp("2026-11-08 15:15")
    assert matchup_player_losses(matchups, report_times).tolist() == [0.0]
