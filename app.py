"""Flask web app for the NBA odds predictor."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for

from cleaner import clean_games, normalize_team_names
from model import run_elo
from predictor import (
    evaluate_prediction_models,
    get_latest_prediction_context,
    load_optional_team_injuries,
    predict_matchup_from_games,
    refresh_official_injury_dataset,
)
from scraper import scrape_bref_season_games, scrape_multiple_seasons

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
EXPORTS_DIR = DATA_DIR / "exports"
SCRAPE_EXPORTS_DIR = EXPORTS_DIR / "scrapes"
PREDICTOR_EXPORTS_DIR = EXPORTS_DIR / "predictor"

RAW_FILE = RAW_DIR / "bref_games_2016_2025.csv"
PROCESSED_FILE = PROCESSED_DIR / "games_2016_2025_normalized.csv"
PREDICTOR_START_SEASON = 2020
PREDICTOR_END_SEASON = 2025
PREDICTOR_RAW_FILE = RAW_DIR / f"bref_games_{PREDICTOR_START_SEASON}_{PREDICTOR_END_SEASON}.csv"
PREDICTOR_PROCESSED_FILE = (
    PROCESSED_DIR / f"games_{PREDICTOR_START_SEASON}_{PREDICTOR_END_SEASON}_normalized.csv"
)
PREDICTOR_EVAL_SUMMARY_FILE = (
    RESULTS_DIR / f"predictor_eval_{PREDICTOR_START_SEASON}_{PREDICTOR_END_SEASON}.json"
)
APP_TITLE = "NBA Odds Predictor"

DEFAULT_START_SEASON = 2016
DEFAULT_END_SEASON = 2025
PREVIEW_OPTIONS = (5, 10, 25, 50, 100)
ENGINEERED_HEURISTICS = (
    "Team strength blends season win percentage with average scoring margin.",
    "Blended strength combines the team-strength score with normalized Elo.",
    "Recent form tracks rolling five-game win rate and scoring margin.",
    "Rest measures days since each team last played before the matchup.",
    "Injury impact uses official status counts, weighted severity, and estimated absence days.",
)

NAV_ITEMS = (
    {"endpoint": "home", "label": "Home"},
    {"endpoint": "leaderboard", "label": "Leaderboard"},
    {"endpoint": "predictor", "label": "Predictor"},
    {"endpoint": "scrape", "label": "Scrape"},
    {"endpoint": "bayesian_elo", "label": "NBA Elo Model"},
)


def ensure_data_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRAPE_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTOR_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


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


def save_processed_games(games: pd.DataFrame) -> pd.DataFrame:
    processed = normalize_team_names(clean_games(games))
    processed.to_csv(PROCESSED_FILE, index=False)
    return processed


def load_predictor_games() -> pd.DataFrame:
    if PREDICTOR_PROCESSED_FILE.exists():
        return pd.read_csv(PREDICTOR_PROCESSED_FILE, parse_dates=["date"])

    if not PREDICTOR_RAW_FILE.exists():
        raise FileNotFoundError(
            "Predictor training data is missing. Refresh predictor data on the Predictor page first."
        )

    raw_games = pd.read_csv(PREDICTOR_RAW_FILE, parse_dates=["date"])
    processed = normalize_team_names(clean_games(raw_games))
    processed.to_csv(PREDICTOR_PROCESSED_FILE, index=False)
    return processed


def refresh_predictor_dataset() -> tuple[pd.DataFrame, bool, dict[str, object]]:
    existing_processed = None
    if PREDICTOR_PROCESSED_FILE.exists():
        existing_processed = pd.read_csv(PREDICTOR_PROCESSED_FILE, parse_dates=["date"])

    fresh_games = scrape_multiple_seasons(
        PREDICTOR_START_SEASON, PREDICTOR_END_SEASON, sleep=2
    )
    if fresh_games.empty:
        if PREDICTOR_PROCESSED_FILE.exists():
            existing_games = load_predictor_games()
            return existing_games, True, refresh_official_injury_dataset(existing_games)
        return pd.DataFrame(), True, {
            "status": "no_games",
            "message": "Predictor game refresh did not return any games yet.",
            "rows": 0,
            "file": None,
        }

    fresh_games.to_csv(PREDICTOR_RAW_FILE, index=False)
    fresh_processed = normalize_team_names(clean_games(fresh_games))

    if existing_processed is not None:
        current_sorted = existing_processed.sort_values(
            ["date", "home_team", "away_team"]
        ).reset_index(drop=True)
        fresh_sorted = fresh_processed.sort_values(
            ["date", "home_team", "away_team"]
        ).reset_index(drop=True)
        if current_sorted.equals(fresh_sorted):
            return existing_processed, True, refresh_official_injury_dataset(existing_processed)

    fresh_processed.to_csv(PREDICTOR_PROCESSED_FILE, index=False)
    return fresh_processed, False, refresh_official_injury_dataset(fresh_processed)


def build_predictor_artifacts(games: pd.DataFrame) -> dict[str, object]:
    injury_df = load_optional_team_injuries(games)
    evaluation_summary, export_df = evaluate_prediction_models(games, injury_df)
    export_name = (
        f"predictor_latest_season_predictions_{int(evaluation_summary['test_season'])}.csv"
    )
    export_path = PREDICTOR_EXPORTS_DIR / export_name
    export_df.to_csv(export_path, index = False)

    serialized_summary = dict(evaluation_summary)
    serialized_summary["export_name"] = export_name
    serialized_summary["export_path"] = str(export_path)
    PREDICTOR_EVAL_SUMMARY_FILE.write_text(json.dumps(serialized_summary, indent = 2))
    return serialized_summary


def load_predictor_artifacts() -> dict[str, object] | None:
    if not PREDICTOR_EVAL_SUMMARY_FILE.exists():
        return None
    return json.loads(PREDICTOR_EVAL_SUMMARY_FILE.read_text())


def save_scraped_games(start_season: int, end_season: int) -> tuple[pd.DataFrame, str]:
    games = scrape_multiple_seasons(start_season, end_season, sleep=2)
    export_name = f"games_{start_season}_{end_season}_{uuid4().hex[:8]}.csv"
    export_path = SCRAPE_EXPORTS_DIR / export_name
    games.to_csv(export_path, index=False)
    return games, export_name


def build_leaderboard_frame(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["rank", "team", "rating"])

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


def get_live_season_end_year(reference_date: datetime | None = None) -> int:
    current = reference_date or datetime.now()
    if current.month >= 10:
        return current.year + 1
    return current.year


def refresh_live_leaderboard() -> tuple[pd.DataFrame, int, bool]:
    season = get_live_season_end_year()
    fresh_games = scrape_bref_season_games(season, sleep=2)

    if fresh_games.empty:
        if PROCESSED_FILE.exists():
            return build_leaderboard_frame(get_latest_live_games()), season, True
        return pd.DataFrame(columns=["rank", "team", "rating"]), season, True

    fresh_processed = save_processed_games(fresh_games)

    if PROCESSED_FILE.exists():
        existing = load_processed_games()
        existing_current = existing[existing["season"] == season].copy()
        fresh_current = fresh_processed[fresh_processed["season"] == season].copy()

        if not existing_current.empty:
            existing_current = existing_current.sort_values(
                ["date", "home_team", "away_team"]
            ).reset_index(drop=True)
        if not fresh_current.empty:
            fresh_current = fresh_current.sort_values(
                ["date", "home_team", "away_team"]
            ).reset_index(drop=True)

        if not fresh_current.empty and existing_current.equals(fresh_current):
            return build_leaderboard_frame(existing), season, True

    return build_leaderboard_frame(fresh_processed), season, False


def get_latest_live_games() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        return pd.DataFrame(columns=["date", "season", "home_team", "away_team", "home_pts", "away_pts", "home_win", "margin"])
    games = load_processed_games()
    if games.empty:
        return pd.DataFrame(
            columns=["date", "season", "home_team", "away_team", "home_pts", "away_pts", "home_win", "margin"]
        )
    live_season = get_live_season_end_year()
    live_games = games[games["season"] == live_season].copy()
    if live_games.empty:
        live_season = int(games["season"].max())
        live_games = games[games["season"] == live_season].copy()
    return live_games


def load_home_stats() -> dict[str, object]:
    if not PROCESSED_FILE.exists():
        return {
            "game_count": 0,
            "team_count": 0,
            "season_count": 0,
            "latest_season": None,
            "latest_season_games": 0,
            "last_updated": None,
        }

    games = get_latest_live_games()
    latest_season = int(games["season"].max()) if not games.empty else None
    latest_season_games = int((games["season"] == latest_season).sum()) if latest_season else 0
    team_count = int(len(set(games["home_team"]).union(set(games["away_team"]))))
    last_updated = datetime.fromtimestamp(PROCESSED_FILE.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M"
    )

    return {
        "game_count": int(len(games)),
        "team_count": team_count,
        "season_count": int(games["season"].nunique()),
        "latest_season": latest_season,
        "latest_season_games": latest_season_games,
        "last_updated": last_updated,
    }


def clear_generated_data() -> dict[str, int]:
    removed_files = 0
    removed_dirs = 0

    for directory in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, EXPORTS_DIR):
        if not directory.exists():
            continue

        for path in list(directory.iterdir()):
            if path.is_file():
                path.unlink()
                removed_files += 1
            elif path.is_dir():
                rmtree(path)
                removed_dirs += 1

    return {"files": removed_files, "dirs": removed_dirs}


def create_app() -> Flask:
    ensure_data_dirs()
    app = Flask(__name__)
    app.secret_key = "nbamodel-local-dev"
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    @app.context_processor
    def inject_nav():
        return {"nav_items": NAV_ITEMS, "app_title": APP_TITLE}

    @app.get("/")
    def home():
        stats = load_home_stats()
        return render_template("home.html", stats=stats)

    @app.route("/leaderboard", methods=["GET"])
    def leaderboard():
        try:
            games = get_latest_live_games()
            standings = build_leaderboard_frame(games)
            latest_season = int(games["season"].max()) if not games.empty else None
            last_updated = datetime.fromtimestamp(PROCESSED_FILE.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
        except FileNotFoundError:
            standings = pd.DataFrame(columns=["rank", "team", "rating"])
            latest_season = None
            last_updated = None

        return render_template(
            "leaderboard.html",
            standings=standings.to_dict(orient="records"),
            latest_season=latest_season,
            last_updated=last_updated,
        )

    @app.post("/leaderboard/refresh")
    def refresh_leaderboard():
        standings, season, already_current = refresh_live_leaderboard()
        if already_current:
            flash(
                f"The current {season}-{str(season + 1)[-2:]} season is already up to date.",
                "info",
            )
        else:
            flash(
                f"Leaderboard refreshed from the live {season}-{str(season + 1)[-2:]} season.",
                "success",
            )
        return redirect(url_for("leaderboard"))

    @app.route("/scrape", methods=["GET", "POST"])
    def scrape():
        message = None
        error = None
        preview = []
        start_season = DEFAULT_START_SEASON
        end_season = DEFAULT_END_SEASON
        preview_limit = 10
        download_name = None
        current_season_note = (
            "If you scrape the current season, the import includes only games that have "
            "already been played. The leaderboard always reflects the live Elo state of "
            "the current season, or the most recent season if the current one has not started."
        )
        download_url = None

        def parse_preview_limit(raw_value: str | None) -> int:
            try:
                value = int(raw_value or 10)
            except ValueError:
                return 10
            return value if value in PREVIEW_OPTIONS else 10

        if request.method == "POST":
            try:
                start_season = int(request.form.get("start_season", DEFAULT_START_SEASON))
                end_season = int(request.form.get("end_season", DEFAULT_END_SEASON))
                preview_limit = parse_preview_limit(request.form.get("preview_limit"))
                if start_season > end_season:
                    raise ValueError("Start season cannot be after end season.")

                games, download_name = save_scraped_games(start_season, end_season)
                preview_frame = games.head(preview_limit).copy()
                if not preview_frame.empty and "date" in preview_frame.columns:
                    preview_frame["date"] = pd.to_datetime(preview_frame["date"]).dt.strftime(
                        "%Y-%m-%d"
                    )
                preview = preview_frame.to_dict(orient="records")
                download_url = url_for("download_scrape_export", filename=download_name)
                message = (
                    f"Scraped seasons {start_season}-{end_season}. "
                    "You can preview the CSV or download the full range export."
                )
            except Exception as exc:  # noqa: BLE001 - user-facing page feedback
                error = str(exc)

        return render_template(
            "scrape.html",
            start_season=start_season,
            end_season=end_season,
            preview_limit=preview_limit,
            message=message,
            error=error,
            preview=preview,
            download_name=download_name,
            download_url=download_url,
            current_season_note=current_season_note,
            preview_options=PREVIEW_OPTIONS,
        )

    @app.route("/predictor", methods=["GET", "POST"])
    def predictor():
        prediction = None
        error = None
        injury_file_present = False
        evaluation_summary = None

        try:
            games = load_predictor_games()
            context = get_latest_prediction_context(games)
            injury_df = load_optional_team_injuries(games)
            injury_file_present = injury_df is not None
            evaluation_summary = load_predictor_artifacts()
            if evaluation_summary is None:
                try:
                    evaluation_summary = build_predictor_artifacts(games)
                except Exception:  # noqa: BLE001 - keep the predictor page usable without cached evals
                    evaluation_summary = None
            selected_season = int(context["season"])
            prediction_date = context["prediction_date"]
            latest_completed_date = context["latest_completed_date"]
            teams = context["teams"]
            default_home_team = teams[0] if teams else ""
            default_away_team = teams[1] if len(teams) > 1 else default_home_team
            home_team = request.form.get("home_team", default_home_team)
            away_team = request.form.get("away_team", default_away_team)

            if request.method == "POST":
                prediction = predict_matchup_from_games(
                    games,
                    home_team = home_team,
                    away_team = away_team,
                    season = selected_season,
                    game_date = pd.Timestamp(prediction_date),
                )
        except Exception as exc:  # noqa: BLE001 - user-facing feedback
            error = str(exc)
            teams = []
            selected_season = None
            prediction_date = datetime.now().strftime("%Y-%m-%d")
            latest_completed_date = None
            home_team = ""
            away_team = ""
            injury_file_present = False
            evaluation_summary = None

        return render_template(
            "predictor.html",
            prediction = prediction,
            error = error,
            predictor_start_season = PREDICTOR_START_SEASON,
            predictor_end_season = PREDICTOR_END_SEASON,
            selected_season = selected_season,
            teams = teams,
            home_team = home_team,
            away_team = away_team,
            prediction_date = prediction_date,
            latest_completed_date = latest_completed_date,
            injury_file_present = injury_file_present,
            evaluation_summary = evaluation_summary,
            engineered_heuristics = ENGINEERED_HEURISTICS,
        )

    @app.post("/predictor/refresh")
    def refresh_predictor():
        games, already_current, injury_status = refresh_predictor_dataset()
        if games.empty and already_current:
            flash(
                "Predictor refresh did not return any games yet. Try again in a bit.",
                "error",
            )
        elif already_current:
            flash(
                f"Predictor training data for {PREDICTOR_START_SEASON}-{PREDICTOR_END_SEASON} is already up to date.",
                "info",
            )
        else:
            flash(
                f"Predictor training data refreshed for {PREDICTOR_START_SEASON}-{PREDICTOR_END_SEASON}.",
                "success",
            )

        if injury_status["status"] == "refreshed":
            flash(injury_status["message"], "success")
        elif injury_status["status"] in {"missing_dependency", "no_overlap"}:
            flash(injury_status["message"], "info")
        elif injury_status["status"] == "no_games":
            flash(injury_status["message"], "error")

        if not games.empty:
            try:
                evaluation_summary = build_predictor_artifacts(games)
                flash(
                    (
                        f"Updated predictor comparison on test season "
                        f"{evaluation_summary['test_season']} and exported latest-season predictions."
                    ),
                    "success",
                )
            except Exception as exc:  # noqa: BLE001 - user-facing feedback
                flash(f"Predictor evaluation refresh skipped: {exc}", "info")

        return redirect(url_for("predictor"))

    @app.post("/reset-data")
    def reset_data():
        counts = clear_generated_data()
        flash(
            f"Reset complete. Removed {counts['files']} generated files and "
            f"{counts['dirs']} generated folders.",
            "success",
        )
        return redirect(url_for("home"))

    @app.get("/bayesian-elo")
    def bayesian_elo():
        return render_template("bayesian_elo.html")

    @app.get("/downloads/scrapes/<path:filename>")
    def download_scrape_export(filename: str):
        return send_from_directory(SCRAPE_EXPORTS_DIR, filename, as_attachment=True)

    @app.get("/downloads/predictor/<path:filename>")
    def download_predictor_export(filename: str):
        return send_from_directory(PREDICTOR_EXPORTS_DIR, filename, as_attachment = True)

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
