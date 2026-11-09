from app import create_app
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def test_homepage_contains_project_sections():
    app = create_app()
    client = app.test_client()
    response = client.get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "NBA Odds Predictor" in body
    assert "Inner-Workings" in body
    assert "About" in body
    assert "Current-season behavior" not in body


def test_health_endpoint():
    app = create_app()
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_injury_aggregation_accepts_non_string_status_values():
    from predictor import aggregate_team_injuries

    injuries = pd.DataFrame(
        [
            {
                "game_date": "2025-04-12",
                "team": "BOS",
                "player_name": "Example Player",
                "status": 1.0,
                "reason": None,
                "report_datetime": "2025-04-12 17:00:00",
            }
        ]
    )

    aggregated = aggregate_team_injuries(injuries)

    assert len(aggregated) == 1
    assert aggregated.iloc[0]["reported_player_count"] == 1


def test_about_page_renders():
    app = create_app()
    client = app.test_client()
    response = client.get("/about")
    body = response.get_data(as_text = True)

    assert response.status_code == 200
    assert "Any questions?" in body
    assert "bzeng0000@gmail.com" in body
    assert "linkedin.com/in/brianbzeng" in body
    assert "github.com/brianbzeng" in body
    assert "mailto:bzeng0000@gmail.com" in body
    assert "Email question" in body


def test_leaderboard_uses_live_season_only(monkeypatch):
    import app as app_module

    games = app_module.pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "season": 2024,
                "home_team": "LAL",
                "away_team": "BOS",
                "home_pts": 100,
                "away_pts": 98,
                "home_win": 1,
                "margin": 2,
            },
            {
                "date": "2025-10-01",
                "season": 2025,
                "home_team": "DEN",
                "away_team": "MIA",
                "home_pts": 110,
                "away_pts": 105,
                "home_win": 1,
                "margin": 5,
            },
        ]
    )

    captured = {}

    def fake_build_leaderboard_frame(frame):
        captured["seasons"] = sorted(frame["season"].unique().tolist())
        return app_module.pd.DataFrame(
            [{"rank": 1, "team": "DEN", "rating": 1700.0}]
        )

    monkeypatch.setattr(app_module, "get_latest_live_games", lambda: games[games["season"] == 2025].copy())
    monkeypatch.setattr(
        app_module, "get_live_season_end_year", lambda reference_date=None: 2025
    )
    monkeypatch.setattr(app_module, "build_leaderboard_frame", fake_build_leaderboard_frame)
    monkeypatch.setattr(app_module, "PROCESSED_FILE", Path("README.md"))

    app = create_app()
    client = app.test_client()
    response = client.get("/leaderboard")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert captured["seasons"] == [2025]
    assert "DEN" in body
    assert "LAL" not in body


def test_leaderboard_route_renders(monkeypatch):
    import app as app_module

    games = app_module.pd.DataFrame(
        [
            {
                "date": "2025-10-01",
                "season": 2025,
                "home_team": "DEN",
                "away_team": "MIA",
                "home_pts": 110,
                "away_pts": 105,
                "home_win": 1,
                "margin": 5,
            },
        ]
    )

    monkeypatch.setattr(app_module, "get_latest_live_games", lambda: games)
    monkeypatch.setattr(
        app_module,
        "build_leaderboard_frame",
        lambda frame: app_module.pd.DataFrame([{"rank": 1, "team": "DEN", "rating": 1700.0}]),
    )

    app = create_app()
    client = app.test_client()
    response = client.get("/leaderboard")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Leaderboard" in body
    assert "Team" in body
    assert "Rating" in body


def set_prediction_data(monkeypatch):
    import app as app_module

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 9, 1, tzinfo=UTC).astimezone(tz)

    slate = pd.DataFrame(
        [
            ("2026-11-08", "2026-11-09T02:00:00Z", "LAC", "NOP", "scheduled", 0.46),
            ("2026-11-08", "2026-11-08T20:30:00Z", "CHI", "PHI", "final", None),
            ("2026-11-07", "2026-11-08T00:30:00Z", "ATL", "BRK", "final", 0.65),
            ("2026-11-09", "2026-11-10T00:00:00Z", "CHA", "CLE", "scheduled", 0.23),
            ("2026-11-08", "2026-11-08T23:00:00Z", "DET", "LAL", "in_progress", None),
        ],
        columns=["game_date", "tipoff_utc", "home_team", "away_team", "status", "home_win_probability"],
    )
    slate["game_date"] = pd.to_datetime(slate["game_date"])
    slate["tipoff_utc"] = pd.to_datetime(slate["tipoff_utc"], utc=True)
    slate["season"] = 2027
    slate["predicted_at_utc"] = pd.to_datetime(
        ["2026-11-09T01:00:00Z", None, "2026-11-08T00:00:00Z", "2026-11-09T23:45:00Z", None], utc=True
    )
    finals = pd.DataFrame(
        [
            (pd.Timestamp("2026-11-07"), 2027, "Atlanta Hawks", "Brooklyn Nets", 110, 101),
            (pd.Timestamp("2026-11-08"), 2027, "Chicago Bulls", "Philadelphia 76ers", 121, 127),
            (pd.Timestamp("2026-11-06"), 2027, "Detroit Pistons", "Utah Jazz", 116, 115),
            (pd.Timestamp("2026-06-13"), 2026, "New York Knicks", "San Antonio Spurs", 94, 90),
        ],
        columns=["date", "season", "home_team", "away_team", "home_pts", "away_pts"],
    )
    replay = pd.DataFrame(
        [
            (pd.Timestamp("2026-11-08"), "CHI", "PHI", 0.40),
            (pd.Timestamp("2026-11-08"), "DET", "LAL", 0.76),
            (pd.Timestamp("2026-11-07"), "ATL", "BRK", 0.30),
        ],
        columns=["game_date", "home_team", "away_team", "home_win_probability"],
    )

    monkeypatch.setattr(app_module, "datetime", FixedDateTime)
    monkeypatch.setattr(app_module, "read_daily_predictions", lambda: slate)
    monkeypatch.setattr(app_module, "read_game_updates", lambda: finals)
    monkeypatch.setattr(app_module, "read_prediction_replay", lambda: replay)
    return slate


def test_predictor_shows_the_pacific_slate_and_saved_picks(monkeypatch):
    set_prediction_data(monkeypatch)
    client = create_app().test_client()
    response = client.get("/predictor")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Sunday, November 8" in body
    assert "12:30 PM PT" in body and "3:00 PM PT" in body
    assert "6:00 PM PT" in body
    assert "New Orleans Pelicans" in body
    assert "54.0%" in body and "46.0%" in body
    assert "<strong>New Orleans Pelicans</strong>" in body
    assert body.count('prediction-pick-label">Random forest</span>') == 2
    assert "Replay" not in body and "Saved pregame" not in body
    assert "76.0%" in body
    assert body.index("Philadelphia 76ers") < body.index("Los Angeles Lakers") < body.index("New Orleans Pelicans")
    assert "In progress" not in body and "Time / status" not in body
    assert "prediction-score" not in body and "Philadelphia 76ers won" not in body
    assert "Atlanta Hawks" not in body and "Cleveland Cavaliers" not in body
    assert "<form" not in body
    assert client.post("/predictor").status_code == 405


def test_prediction_history_includes_every_current_season_final_newest_first(monkeypatch):
    set_prediction_data(monkeypatch)
    response = create_app().test_client().get("/predictor?view=history")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Prediction history" in body
    assert "2 / 3 correct" in body
    assert "3 completed games · 1 of 1 live picks correct" in body
    assert "Atlanta Hawks won" in body
    assert "Philadelphia 76ers won" in body
    assert "prediction-score" not in body and ">110<" not in body and ">101<" not in body
    assert "65.0%" in body
    assert "70.0%" not in body  # Live pick takes precedence over replay for Atlanta.
    assert "Replay · Random forest" in body
    assert 'prediction-pick-label">Logistic regression</span>' in body
    assert "Saved pregame" not in body
    assert "Detroit Pistons won" in body and "No pick saved before tipoff" in body
    assert body.index("Philadelphia 76ers") < body.index("Atlanta Hawks") < body.index("Detroit Pistons")
    assert "New York Knicks" not in body
    assert "New Orleans Pelicans" not in body
    assert "Los Angeles Lakers" not in body  # An unfinished replay does not count toward accuracy.
    assert "data-slate-refresh-in" not in body


def test_prediction_history_searches_full_season_by_team_and_keeps_query_across_pages(monkeypatch):
    import app as app_module

    set_prediction_data(monkeypatch)
    finals = app_module.read_game_updates()
    extra = finals.iloc[[2]].assign(date=pd.Timestamp("2026-11-05"), away_team="Boston Celtics")
    finals = pd.concat([finals, extra.astype(finals.dtypes)], ignore_index=True)
    monkeypatch.setattr(app_module, "read_game_updates", lambda: finals)
    monkeypatch.setattr(app_module, "PREDICTION_HISTORY_PAGE_SIZE", 1)
    client = create_app().test_client()

    first = client.get("/predictor", query_string={"view": "history", "q": "pIsToNs"}).get_data(as_text=True)
    assert "2 matching games" in first and "2 / 4 correct" in first
    assert "Utah Jazz" in first and "Boston Celtics" not in first and "Philadelphia 76ers" not in first
    assert 'view=history&amp;q=pIsToNs&amp;page=2' in first

    second = client.get("/predictor", query_string={"view": "history", "q": "pIsToNs", "page": 2}).get_data(as_text=True)
    assert "Boston Celtics" in second and "Utah Jazz" not in second
    assert 'value="pIsToNs"' in second and "Page 2 of 2" in second

    by_code = client.get("/predictor", query_string={"view": "history", "q": "BKN"}).get_data(as_text=True)
    assert "1 matching game" in by_code and "Brooklyn Nets" in by_code


def test_prediction_history_searches_exact_dates_and_shows_no_match_state(monkeypatch):
    set_prediction_data(monkeypatch)
    client = create_app().test_client()

    for date_query in ("Nov 8", "November 08, 2026", "2026-11-08", "11/08"):
        body = client.get("/predictor", query_string={"view": "history", "q": date_query}).get_data(as_text=True)
        assert "1 matching game" in body and "Philadelphia 76ers" in body and "Atlanta Hawks" not in body

    empty = client.get("/predictor", query_string={"view": "history", "q": "Nov 8, 2025"}).get_data(as_text=True)
    assert "0 matching games" in empty and "No games match your search." in empty
    assert "2 / 3 correct" in empty and "Clear search" in empty


def test_predictor_reports_the_injury_inputs_used_for_a_saved_pick(monkeypatch, tmp_path):
    import app as app_module

    slate = set_prediction_data(monkeypatch)
    report_file = tmp_path / "reports.csv"
    report_file.write_text(
        "game_date,team,latest_report_datetime,out_count\n2026-11-08,NOP,2026-11-08 20:45:00,3\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "LIVE_INJURIES_FILE", report_file)
    slate["predicted_at_utc"] = pd.to_datetime(["2026-11-09T01:20:00Z", None, None, None, None], utc=True)
    slate["same_day_finals"] = [1, None, None, None, None]
    slate["home_injury_report_utc"] = pd.to_datetime(["2026-11-09T01:15:00Z", None, None, None, None], utc=True)
    slate["away_injury_report_utc"] = pd.NaT
    slate["home_out_count"] = [2, None, None, None, None]
    slate["away_out_count"] = None
    slate["model"] = [app_module.PLAYER_IMPACT_MODEL, None, None, None, None]
    slate["home_player_impact_pp"] = [-2.14, None, None, None, None]
    slate["home_rest_days"] = [0, None, None, None, None]
    slate["away_rest_days"] = [7, None, None, None, None]

    response = create_app().test_client().get("/predictor")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<summary>Context</summary>" in body
    assert "with 1 earlier final" in body
    assert 'prediction-pick-label">Random forest + player impact</span>' in body
    assert "moved the random forest’s home win chance by -2.1 percentage points" in body
    assert "Injury-Report_2026-11-08_08_15PM.pdf" in body
    assert "LAC · 2 out" in body
    assert "NOP · No official report available" in body
    assert "NOP · 3 out" not in body
    assert "0 days · Back-to-back" in body
    assert "7+ days" in body


def test_predictor_shows_logistic_odds_in_context_without_changing_the_official_pick(monkeypatch):
    slate = set_prediction_data(monkeypatch)
    slate["model"] = ["random_forest", None, "random_forest", None, None]
    slate["logistic_home_win_probability"] = [0.58, None, 0.42, None, 0.8]
    slate["player_impact_home_win_probability"] = [0.77, None, 0.83, None, None]
    slate["experiment_version"] = ["rf_player_impact_v1", None, "rf_player_impact_v1", None, None]
    client = create_app().test_client()

    today = client.get("/predictor").get_data(as_text=True)
    assert "<strong>New Orleans Pelicans</strong>" in today
    assert "54.0%" in today and "46.0%" in today
    assert "77.0%" not in today and "rf_player_impact_v1" not in today
    assert today.count("<strong>Logistic regression</strong>") == 1
    assert "<span>NOP</span><span>42.0%</span>" in today
    assert "<span>LAC</span><span>58.0%</span>" in today
    assert "<strong>Models disagree.</strong> Logistic regression favors LA Clippers." in today

    history = client.get("/predictor?view=history").get_data(as_text=True)
    assert "<strong>Atlanta Hawks</strong>" in history and "65.0%" in history
    assert "2 / 3 correct" in history and "1 of 1 live picks correct" in history
    assert "83.0%" not in history and "rf_player_impact_v1" not in history
    assert "<span>BRK</span><span>58.0%</span>" in history
    assert "<span>ATL</span><span>42.0%</span>" in history
    assert "<strong>Models disagree.</strong> Logistic regression favors Brooklyn Nets." in history

    slate.loc[2, "logistic_home_win_probability"] = 0.5
    tied = client.get("/predictor?view=history").get_data(as_text=True)
    assert "<span>BRK</span><span>50.0%</span>" in tied
    assert "Models disagree." not in tied


def test_predictor_rolls_the_slate_over_at_pacific_midnight(monkeypatch):
    import app as app_module

    slate = set_prediction_data(monkeypatch)
    # Nov 9's pick was already saved the previous night.
    slate.loc[3, "predicted_at_utc"] = pd.Timestamp("2026-11-09T07:50:00Z")
    slate.loc[3, "same_day_finals"] = 0
    for side in ("home", "away"):
        slate[f"{side}_injury_report_utc"] = pd.NaT
        slate[f"{side}_out_count"] = None
    now = datetime(2026, 11, 9, 7, 59, 59, tzinfo=UTC)

    class RolloverDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz)

    monkeypatch.setattr(app_module, "datetime", RolloverDateTime)
    client = create_app().test_client()
    before = client.get("/predictor").get_data(as_text=True)
    assert "Sunday, November 8" in before
    assert 'data-slate-refresh-in="1000"' in before
    assert "Cleveland Cavaliers" not in before

    now = datetime(2026, 11, 9, 8, tzinfo=UTC)
    after = client.get("/predictor").get_data(as_text=True)
    assert "Monday, November 9" in after
    assert "<strong>Cleveland Cavaliers</strong>" in after and "77.0%" in after
    assert "Saved Nov 8 at 11:50 PM PT" in after
    assert "Philadelphia 76ers" not in after


def test_predictor_uses_only_earlier_finals_for_rest_on_legacy_rows(monkeypatch):
    import app as app_module

    set_prediction_data(monkeypatch)
    finals = app_module.read_game_updates()
    other_finals = finals.iloc[[2, 2]].assign(
        date=pd.to_datetime(["2026-11-08", "2026-11-09"]), away_team="Los Angeles Lakers",
    )
    combined = pd.concat([finals, other_finals.astype(finals.dtypes)], ignore_index=True)
    monkeypatch.setattr(app_module, "read_game_updates", lambda: combined)

    rows = app_module.load_prediction_rows(app_module.datetime.now(app_module.PREDICTION_DISPLAY_TZ), False)
    lakers_game = next(row for row in rows if row["team_codes"] == ("LAL", "DET"))
    assert [(team["team"], team["rest_days"]) for team in lakers_game["context"]] == [("LAL", None), ("DET", 1)]


def test_bayesian_page_renders():
    app = create_app()
    client = app.test_client()
    response = client.get("/bayesian-elo")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Inner-Workings" in body
    assert "Elo leaderboard" in body
    assert "Home-win predictor" in body
    assert "reports published before each game's tipoff" in body
    assert "1500" in body
    assert "<math" in body


def test_scrape_page_accepts_post(monkeypatch):
    import app as app_module

    def fake_save_scraped_games(start_season, end_season):
        return (
            app_module.pd.DataFrame(
                [
                    {
                        "date": "2025-10-01",
                        "season": 2025,
                        "home_team": "BOS",
                        "away_team": "DEN",
                        "home_pts": 102,
                        "away_pts": 98,
                    }
                ]
            ),
            "games_2016_2016_test.csv",
        )

    monkeypatch.setattr(app_module, "save_scraped_games", fake_save_scraped_games)

    app = create_app()
    client = app.test_client()
    response = client.post(
        "/scrape",
        data={"scrape_action": "games", "start_season": "2016", "end_season": "2016"},
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Scraped seasons 2016-2016" in body
    assert "games_2016_2016_test.csv" in body
    assert "Current-season behavior" in body
    assert "Preview rows" in body
    assert "Download CSV" in body


def test_scrape_page_accepts_injury_post(monkeypatch):
    import app as app_module

    class ImmediateThread:
        def __init__(self, target, args, daemon):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    def fake_save_latest_injury_report():
        detail_df = app_module.pd.DataFrame(
            [
                {
                    "game_date": "2025-04-12",
                    "team": "BOS",
                    "player_name": "Test Player",
                    "status": "Out",
                }
            ]
        )
        team_df = app_module.pd.DataFrame(
            [
                {
                    "game_date": "2025-04-12",
                    "team": "BOS",
                    "out_count": 1,
                    "doubtful_count": 0,
                    "questionable_count": 0,
                    "probable_count": 0,
                    "injury_impact_score": 3.0,
                }
            ]
        )
        meta = {
            "detail_name": "official_nba_injuries_detailed_latest_test.csv",
            "team_name": "official_nba_injuries_by_team_latest_test.csv",
            "report_date": "2025-04-12",
        }
        return detail_df, team_df, meta

    monkeypatch.setattr(app_module, "save_latest_injury_report", fake_save_latest_injury_report)
    monkeypatch.setattr(app_module, "Thread", ImmediateThread)
    app_module.INJURY_JOBS.clear()

    app = create_app()
    client = app.test_client()
    response = client.post(
        "/scrape",
        data={"scrape_action": "injuries", "injury_mode": "latest"},
        follow_redirects = True,
    )
    body = response.get_data(as_text = True)

    assert response.status_code == 200
    assert "Saved the latest available official injury report for 2025-04-12." in body
    assert "official_nba_injuries_detailed_latest_test.csv" in body
    assert "official_nba_injuries_by_team_latest_test.csv" in body
    assert "Injury CSV preview" in body


def test_refresh_and_reset_routes(monkeypatch):
    import app as app_module

    monkeypatch.setattr(
        app_module,
        "refresh_live_leaderboard",
        lambda: (
            app_module.pd.DataFrame(
                [{"rank": 1, "team": "DEN", "rating": 1700.0}]
            ),
            2025,
            True,
        ),
    )
    monkeypatch.setattr(
        app_module,
        "clear_generated_data",
        lambda: {"files": 3, "dirs": 1},
    )

    app = create_app()
    client = app.test_client()

    refresh_response = client.post("/leaderboard/refresh", follow_redirects=True)
    refresh_body = refresh_response.get_data(as_text=True)
    assert refresh_response.status_code == 200
    assert "already up to date" in refresh_body

    reset_response = client.post("/reset-data", follow_redirects=True)
    reset_body = reset_response.get_data(as_text=True)
    assert reset_response.status_code == 200
    assert "Removed 3 generated files and 1 generated folders" in reset_body
