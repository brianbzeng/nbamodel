from app import create_app


def test_homepage_contains_project_sections():
    app = create_app()
    client = app.test_client()
    response = client.get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "ESPN Odds Predictor" in body
    assert "Bayesian Elo" in body
    assert "Leaderboard" in body


def test_leaderboard_route_renders():
    app = create_app()
    client = app.test_client()
    response = client.get("/leaderboard")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Leaderboard" in body
    assert "Team" in body
    assert "Rating" in body


def test_scrape_page_accepts_post(monkeypatch):
    import app as app_module

    def fake_save_scraped_games(start_season, end_season):
        return app_module.pd.DataFrame(
            [
                {
                    "date": "2025-10-01",
                    "season": 2025,
                    "home_team": "BOS",
                    "away_team": "DEN",
                    "home_pts": 102,
                    "away_pts": 98,
                    "home_win": 1,
                    "margin": 4,
                }
            ]
        )

    monkeypatch.setattr(app_module, "save_scraped_games", fake_save_scraped_games)
    monkeypatch.setattr(
        app_module,
        "build_leaderboard_frame",
        lambda games: app_module.pd.DataFrame(
            [{"rank": 1, "team": "BOS", "rating": 1650.0}]
        ),
    )

    app = create_app()
    client = app.test_client()
    response = client.post(
        "/scrape",
        data={"start_season": "2016", "end_season": "2016"},
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Scraped seasons 2016-2016" in body
