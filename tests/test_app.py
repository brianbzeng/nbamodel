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
