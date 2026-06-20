from app import create_app


def test_homepage_contains_project_sections():
    app = create_app()
    client = app.test_client()
    response = client.get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "NBA Odds Predictor" in body
    assert "NBA Elo Model" in body
    assert "Current-season behavior" not in body


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

    monkeypatch.setattr(app_module, "load_processed_games", lambda: games)
    monkeypatch.setattr(
        app_module, "get_live_season_end_year", lambda reference_date=None: 2025
    )
    monkeypatch.setattr(app_module, "build_leaderboard_frame", fake_build_leaderboard_frame)

    app = create_app()
    client = app.test_client()
    response = client.get("/leaderboard")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert captured["seasons"] == [2025]
    assert "DEN" in body
    assert "LAL" not in body


def test_leaderboard_route_renders():
    app = create_app()
    client = app.test_client()
    response = client.get("/leaderboard")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Leaderboard" in body
    assert "Team" in body
    assert "Rating" in body


def test_predictor_page_renders(monkeypatch):
    import app as app_module

    evaluation_summary = {
        "split": "train_before_2025_test_2025",
        "train_rows": 10,
        "test_rows": 5,
        "test_season": 2025,
        "home_baseline_accuracy": 0.54,
        "elo_accuracy": 0.61,
        "logistic_accuracy": 0.64,
        "random_forest_accuracy": 0.66,
        "best_model": "Random forest",
        "random_forest_vs_elo_gain": 0.05,
        "export_name": "predictor_latest_season_predictions_2025.csv",
    }

    games = app_module.pd.DataFrame(
        [
            {
                "date": "2024-10-22",
                "season": 2025,
                "home_team": "BOS",
                "away_team": "NYK",
                "home_pts": 110,
                "away_pts": 101,
                "home_win": 1,
                "margin": 9,
            },
            {
                "date": "2024-10-23",
                "season": 2025,
                "home_team": "LAL",
                "away_team": "DEN",
                "home_pts": 99,
                "away_pts": 104,
                "home_win": 0,
                "margin": -5,
            },
        ]
    )

    monkeypatch.setattr(app_module, "load_predictor_games", lambda: games)
    monkeypatch.setattr(app_module, "load_optional_team_injuries", lambda frame: None)
    monkeypatch.setattr(app_module, "load_predictor_artifacts", lambda: evaluation_summary)
    monkeypatch.setattr(
        app_module,
        "get_latest_prediction_context",
        lambda frame: {
            "season": 2025,
            "latest_completed_date": "2024-10-23",
            "prediction_date": "2024-10-24",
            "teams": ["BOS", "DEN", "LAL", "NYK"],
        },
    )

    app = create_app()
    client = app.test_client()
    response = client.get("/predictor")
    body = response.get_data(as_text = True)

    assert response.status_code == 200
    assert "Home-win predictor" in body
    assert "Generate prediction" in body
    assert "BOS" in body
    assert "Refresh predictor data" in body
    assert "Training range 2020-2025" in body
    assert "No official injury CSV detected yet." in body
    assert "Latest completed games through 2024-10-23" in body
    assert "Engineered heuristics" in body
    assert "Model comparison" in body
    assert "Final project benchmark" in body
    assert "Download predicted vs true CSV" in body


def test_predictor_page_accepts_post(monkeypatch):
    import app as app_module

    evaluation_summary = {
        "split": "train_before_2025_test_2025",
        "train_rows": 10,
        "test_rows": 5,
        "test_season": 2025,
        "home_baseline_accuracy": 0.54,
        "elo_accuracy": 0.61,
        "logistic_accuracy": 0.64,
        "random_forest_accuracy": 0.66,
        "best_model": "Random forest",
        "random_forest_vs_elo_gain": 0.05,
        "export_name": "predictor_latest_season_predictions_2025.csv",
    }

    games = app_module.pd.DataFrame(
        [
            {
                "date": "2024-10-22",
                "season": 2025,
                "home_team": "BOS",
                "away_team": "NYK",
                "home_pts": 110,
                "away_pts": 101,
                "home_win": 1,
                "margin": 9,
            },
            {
                "date": "2024-10-23",
                "season": 2025,
                "home_team": "LAL",
                "away_team": "DEN",
                "home_pts": 99,
                "away_pts": 104,
                "home_win": 0,
                "margin": -5,
            },
        ]
    )

    # Keep the route test focused on wiring instead of model internals.
    def fake_predict_matchup_from_games(*args, **kwargs):
        return {
            "matchup": "NYK @ BOS",
            "season": 2025,
            "game_date": "2025-01-01",
            "elo": {
                "home_win_probability": 0.58,
                "predicted_winner": "BOS",
            },
            "logistic": {
                "home_win_probability": 0.64,
                "predicted_winner": "BOS",
            },
            "random_forest": {
                "home_win_probability": 0.61,
                "predicted_winner": "BOS",
            },
            "context": {
                "home_win_pct": 0.700,
                "away_win_pct": 0.500,
                "home_team_strength": 0.760,
                "away_team_strength": 0.540,
                "team_strength_diff": 0.220,
                "home_elo_rating": 1610.0,
                "away_elo_rating": 1535.0,
                "elo_diff": 75.0,
                "home_blended_strength": 0.801,
                "away_blended_strength": 0.623,
                "blended_strength_diff": 0.178,
                "home_rest_days": 1,
                "away_rest_days": 0,
                "home_out_count": 1.0,
                "away_out_count": 2.0,
                "home_injury_impact_score": 7.0,
                "away_injury_impact_score": 14.0,
            },
            "injury_data_used": True,
        }

    monkeypatch.setattr(app_module, "load_predictor_games", lambda: games)
    monkeypatch.setattr(app_module, "load_optional_team_injuries", lambda frame: app_module.pd.DataFrame([{"team": "BOS"}]))
    monkeypatch.setattr(app_module, "load_predictor_artifacts", lambda: evaluation_summary)
    monkeypatch.setattr(
        app_module,
        "get_latest_prediction_context",
        lambda frame: {
            "season": 2025,
            "latest_completed_date": "2024-10-23",
            "prediction_date": "2025-01-01",
            "teams": ["BOS", "DEN", "LAL", "NYK"],
        },
    )
    monkeypatch.setattr(app_module, "predict_matchup_from_games", fake_predict_matchup_from_games)

    app = create_app()
    client = app.test_client()
    response = client.post(
        "/predictor",
        data = {
            "home_team": "BOS",
            "away_team": "NYK",
        },
    )
    body = response.get_data(as_text = True)

    assert response.status_code == 200
    assert "NYK @ BOS" in body
    assert "58.0%" in body
    assert "64.0%" in body
    assert "61.0%" in body
    assert "Latest saved injury report was folded into the matchup features" in body
    assert "Random forest is running on the Elo-enhanced feature set" in body


def test_predictor_refresh_route(monkeypatch):
    import app as app_module

    evaluation_summary = {
        "test_season": 2025,
        "export_name": "predictor_latest_season_predictions_2025.csv",
    }

    monkeypatch.setattr(
        app_module,
        "refresh_predictor_dataset",
        lambda: (
            app_module.pd.DataFrame(
                [
                    {
                        "date": "2025-04-12",
                        "season": 2025,
                        "home_team": "BOS",
                        "away_team": "NYK",
                        "home_pts": 110,
                        "away_pts": 101,
                        "home_win": 1,
                        "margin": 9,
                    }
                ]
            ),
            False,
            {
                "status": "refreshed",
                "message": "Official injury data refreshed for seasons 2022-2025 across 10 game dates.",
                "rows": 40,
                "file": "data/raw/official_nba_injuries_by_team_2022_2025.csv",
            },
        ),
    )
    monkeypatch.setattr(app_module, "build_predictor_artifacts", lambda games: evaluation_summary)
    monkeypatch.setattr(
        app_module,
        "load_predictor_games",
        lambda: app_module.pd.DataFrame(
            [
                {
                    "date": "2025-04-12",
                    "season": 2025,
                    "home_team": "BOS",
                    "away_team": "NYK",
                    "home_pts": 110,
                    "away_pts": 101,
                    "home_win": 1,
                    "margin": 9,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        app_module,
        "get_latest_prediction_context",
        lambda frame: {
            "season": 2025,
            "latest_completed_date": "2025-04-12",
            "prediction_date": "2025-04-13",
            "teams": ["BOS", "NYK"],
        },
    )

    app = create_app()
    client = app.test_client()
    response = client.post("/predictor/refresh", follow_redirects = True)
    body = response.get_data(as_text = True)

    assert response.status_code == 200
    assert "Predictor training data refreshed for 2020-2025." in body
    assert "Official injury data refreshed for seasons 2022-2025 across 10 game dates." in body
    assert "Updated predictor comparison on test season 2025 and exported latest-season predictions." in body


def test_bayesian_page_renders():
    app = create_app()
    client = app.test_client()
    response = client.get("/bayesian-elo")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "NBA Elo Model - Inner Statistical Workings" in body
    assert "Beta" in body
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
        data={"start_season": "2016", "end_season": "2016"},
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Scraped seasons 2016-2016" in body
    assert "games_2016_2016_test.csv" in body
    assert "Current-season behavior" in body
    assert "Preview rows" in body
    assert "Download CSV" in body


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
