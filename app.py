"""Flask web app for the NBA odds predictor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from cleaner import clean_games, normalize_team_names
from model import run_elo
from scraper import scrape_multiple_seasons

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

RAW_FILE = RAW_DIR / "bref_games_2016_2025.csv"
PROCESSED_FILE = PROCESSED_DIR / "games_2016_2025_normalized.csv"

DEFAULT_START_SEASON = 2016
DEFAULT_END_SEASON = 2025

NAV_ITEMS = (
    {"endpoint": "home", "label": "Home"},
    {"endpoint": "leaderboard", "label": "Leaderboard"},
    {"endpoint": "scrape", "label": "Scrape"},
)


def ensure_data_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_games() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw games not found at {RAW_FILE}. Use the Scrape page first."
        )
    return pd.read_csv(RAW_FILE, parse_dates=["date"])


def load_processed_games() -> pd.DataFrame:
    if PROCESSED_FILE.exists():
        return pd.read_csv(PROCESSED_FILE, parse_dates=["date"])

    raw_games = load_raw_games()
    processed = normalize_team_names(clean_games(raw_games))
    processed.to_csv(PROCESSED_FILE, index=False)
    return processed


def save_scraped_games(start_season: int, end_season: int) -> pd.DataFrame:
    games = scrape_multiple_seasons(start_season, end_season, sleep=2)
    games.to_csv(RAW_FILE, index=False)

    processed = normalize_team_names(clean_games(games))
    processed.to_csv(PROCESSED_FILE, index=False)
    return processed


def build_leaderboard_frame(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["team", "rating"])

    _, final_ratings = run_elo(games)
    last_season = int(games["season"].max())
    teams = sorted(
        set(games.loc[games["season"] == last_season, "home_team"]).union(
            games.loc[games["season"] == last_season, "away_team"]
        )
    )

    standings = (
        pd.DataFrame(
            {"team": teams, "rating": [final_ratings[team] for team in teams]}
        )
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    standings["rank"] = standings.index + 1
    return standings[["rank", "team", "rating"]]


def build_home_stats(games: pd.DataFrame) -> dict[str, object]:
    if games.empty:
        return {
            "game_count": 0,
            "team_count": 0,
            "season_count": 0,
            "latest_season": None,
            "top_team": None,
        }

    leaderboard = build_leaderboard_frame(games)
    latest_season = int(games["season"].max())
    season_count = int(games["season"].nunique())
    team_count = int(
        len(set(games["home_team"]).union(set(games["away_team"]))))
    top_team = leaderboard.iloc[0].to_dict() if not leaderboard.empty else None

    return {
        "game_count": int(len(games)),
        "team_count": team_count,
        "season_count": season_count,
        "latest_season": latest_season,
        "top_team": top_team,
    }


def create_app() -> Flask:
    ensure_data_dirs()
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    @app.context_processor
    def inject_nav():
        return {"nav_items": NAV_ITEMS}

    @app.get("/")
    def home():
        try:
            games = load_processed_games()
            stats = build_home_stats(games)
        except FileNotFoundError:
            stats = build_home_stats(pd.DataFrame())
        return render_template("home.html", stats=stats)

    @app.get("/leaderboard")
    def leaderboard():
        try:
            games = load_processed_games()
            standings = build_leaderboard_frame(games)
            latest_season = int(games["season"].max()
                                ) if not games.empty else None
        except FileNotFoundError:
            standings = pd.DataFrame(columns=["rank", "team", "rating"])
            latest_season = None
        return render_template(
            "leaderboard.html",
            standings=standings.to_dict(orient="records"),
            latest_season=latest_season,
        )

    @app.route("/scrape", methods=["GET", "POST"])
    def scrape():
        message = None
        error = None
        preview = []
        start_season = DEFAULT_START_SEASON
        end_season = DEFAULT_END_SEASON

        if request.method == "POST":
            try:
                start_season = int(request.form.get(
                    "start_season", DEFAULT_START_SEASON))
                end_season = int(request.form.get(
                    "end_season", DEFAULT_END_SEASON))
                if start_season > end_season:
                    raise ValueError(
                        "Start season cannot be after end season.")

                processed = save_scraped_games(start_season, end_season)
                preview = build_leaderboard_frame(processed).head(10).to_dict(
                    orient="records"
                )
                message = (
                    f"Scraped seasons {start_season}-{end_season} and refreshed the "
                    "processed dataset."
                )
            except Exception as exc:  # noqa: BLE001 - user-facing page feedback
                error = str(exc)

        return render_template(
            "scrape.html",
            start_season=start_season,
            end_season=end_season,
            message=message,
            error=error,
            preview=preview,
        )

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
