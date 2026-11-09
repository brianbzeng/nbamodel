from datetime import datetime

import pandas as pd

import injury_updates
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

    result = injury_updates.collect_injuries(
        teams,
        predictor.aggregate_team_injuries(pd.DataFrame()),
        datetime(2026, 11, 8, 17),
        {2027: [earlier, nominal]},
    ).set_index("team")

    assert set(result.index) == {"PHI", "DET"}
    assert result.at["PHI", "latest_report_datetime"] == earlier
    assert result.at["PHI", "questionable_count"] == 1
    assert result.at["PHI", "out_count"] == 0
    assert result.at["DET", "reported_player_count"] == 0
    assert result.at["DET", "latest_report_datetime"] == published


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
