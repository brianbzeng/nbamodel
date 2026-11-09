from datetime import datetime, timedelta

import pandas as pd
import pytest

import injury_updates
import player_injuries
import predictor


def test_uses_pregame_reports_and_distinguishes_unsubmitted_teams(monkeypatch):
    game_date = pd.Timestamp("2026-11-08")
    earlier = datetime(2026, 11, 8, 14, 45)
    nominal = datetime(2026, 11, 8, 15, 15)
    published = datetime(2026, 11, 8, 15, 35)

    def player(team, status, reason, report_time):
        return {
            "game_date": game_date,
            "team": team,
            "player_name": "Player" if status else "",
            "status": status,
            "reason": reason,
            "report_datetime": report_time,
        }

    reports = {
        earlier: pd.DataFrame([player("PHI", "Questionable", "Ankle sprain", earlier)]),
        nominal: pd.DataFrame([
            player("PHI", "Out", "Ankle sprain", published),
            player("CHI", "", "NOT YET SUBMITTED", published),
            player("DET", "", "NO INJURIES TO REPORT", published),
        ]),
    }
    monkeypatch.setattr(injury_updates, "prepare_report_parser", lambda: None)
    monkeypatch.setattr(injury_updates, "fetch_report", lambda stamp: reports.get(stamp, pd.DataFrame()))
    teams = {
        (game_date, "PHI"): (datetime(2026, 11, 8, 15, 30), 2027),
        (game_date, "CHI"): (datetime(2026, 11, 8, 15, 30), 2027),
        (game_date, "DET"): (datetime(2026, 11, 8, 18), 2027),
    }

    result, details = injury_updates.collect_injuries(
        teams,
        predictor.aggregate_team_injuries(pd.DataFrame()),
        pd.DataFrame(columns=player_injuries.PLAYER_INJURY_COLUMNS),
        datetime(2026, 11, 8, 17),
        {2027: [earlier, nominal]},
    )
    result = result.set_index("team")

    assert set(result.index) == {"PHI", "DET"}
    assert result.at["PHI", "latest_report_datetime"] == earlier
    assert result.at["PHI", "questionable_count"] == 1
    assert result.at["PHI", "out_count"] == 0
    assert result.at["DET", "reported_player_count"] == 0
    assert result.at["DET", "latest_report_datetime"] == published
    assert details.set_index("team")["status"].to_dict() == {"PHI": "QUESTIONABLE", "DET": ""}

    _, recovered = injury_updates.collect_injuries(
        {(game_date, "PHI"): teams[(game_date, "PHI")]},
        result.loc[["PHI"]].reset_index(),
        details.iloc[:0],
        datetime(2026, 11, 8, 17),
        {2027: [earlier, nominal]},
    )
    assert recovered["report_datetime"].tolist() == [earlier]


def test_refresh_retains_identity_changes_with_identical_team_counts(tmp_path, monkeypatch):
    game_date = pd.Timestamp("2026-11-08")
    reports = pd.DataFrame({
        "game_date": [game_date, game_date],
        "team": ["LAL", "LAL"],
        "player_name": ["Doncic, Luka", "Other, Player"],
        "status": ["OUT", "OUT"],
        "reason": ["Injury/Illness", "Injury/Illness"],
        "report_datetime": pd.to_datetime(["2026-11-08 16:45", "2026-11-08 17:15"]),
    })
    teams_path = tmp_path / "teams.csv"
    players_path = tmp_path / player_injuries.LIVE_PLAYER_INJURIES_FILE.name
    old = predictor.aggregate_team_injuries(reports.iloc[:1])
    old.to_csv(teams_path, index=False)
    reports.iloc[:1][player_injuries.PLAYER_INJURY_COLUMNS].to_csv(players_path, index=False)
    monkeypatch.setattr(injury_updates, "fetch_scoreboard", lambda _: [])
    new = predictor.aggregate_team_injuries(reports.iloc[1:])
    details = reports.iloc[1:][player_injuries.PLAYER_INJURY_COLUMNS]
    monkeypatch.setattr(injury_updates, "collect_injuries", lambda *args: (new, details))

    result = injury_updates.refresh_injury_updates(teams_path)

    assert result["status"] == "refreshed"
    assert pd.read_csv(teams_path)["latest_report_datetime"].tolist() == ["2026-11-08 17:15:00"]
    assert player_injuries.read_live_player_injury_reports(players_path)["player_name"].tolist() == [
        "Doncic, Luka", "Other, Player",
    ]
    assert injury_updates.refresh_injury_updates(teams_path)["status"] == "cached"


def test_refresh_validates_coverage_before_publishing(tmp_path, monkeypatch):
    today = pd.Timestamp(datetime.now(injury_updates.NBA_TIMEZONE).date())
    yesterday = today - timedelta(days=1)
    teams_path = tmp_path / "teams.csv"
    players_path = tmp_path / player_injuries.LIVE_PLAYER_INJURIES_FILE.name
    reports = pd.DataFrame([{
        "game_date": today, "team": "LAL", "player_name": "Doncic, Luka", "status": "OUT",
        "reason": "Injury/Illness", "report_datetime": today,
    }])
    monkeypatch.setattr(injury_updates, "fetch_scoreboard", lambda _: [])
    monkeypatch.setattr(injury_updates, "fetch_report_index", lambda _: [])
    monkeypatch.setattr(injury_updates, "scheduled_teams", lambda *_: {(yesterday, "DET"): (yesterday, 2027)})
    monkeypatch.setattr(injury_updates, "collect_injuries", lambda *_: (
        predictor.aggregate_team_injuries(reports), reports[player_injuries.PLAYER_INJURY_COLUMNS],
    ))

    with pytest.raises(RuntimeError, match="No official pregame injury reports"):
        injury_updates.refresh_injury_updates(teams_path)

    assert not teams_path.exists()
    assert not players_path.exists()


def test_predictor_loads_live_injuries_alongside_exact_historical_file(tmp_path, monkeypatch):
    monkeypatch.setattr(predictor, "RAW_DIR", tmp_path)
    live_path = tmp_path / "official_nba_injuries_by_team_live.csv"
    monkeypatch.setattr(predictor, "LIVE_INJURIES_FILE", live_path)
    header = "game_date,team,latest_report_datetime\n"
    (tmp_path / "official_nba_injuries_by_team_2022_2027.csv").write_text(
        header + "2026-11-08,Chicago Bulls,2026-11-08 14:30:00\n"
    )
    live_path.write_text(header + "2026-11-08,CHI,2026-11-08 15:00:00\n")

    injuries = predictor.load_optional_team_injuries(pd.DataFrame({"season": [2022, 2027]}))

    assert injuries["team"].tolist() == ["CHI"]
    assert injuries["latest_report_datetime"].tolist() == [pd.Timestamp("2026-11-08 15:00:00")]
