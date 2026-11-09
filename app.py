"""Flask web app for the NBA odds predictor."""

from __future__ import annotations

import logging
import os
from threading import Lock, Thread
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from shutil import rmtree
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

from cleaner import TEAM_MAP_BREF, clean_games, normalize_team_names
from daily_predictions import DAILY_PREDICTIONS_FILE, read_daily_predictions
from game_updates import LIVE_GAMES_FILE, NBA_TIMEZONE, read_game_updates, refresh_game_updates
from injury_reports import report_pdf_url
from injury_updates import refresh_injury_updates
from model import run_elo
from predictor import (
    LIVE_INJURIES_FILE,
    aggregate_team_injuries,
    fetch_report_with_retries,
    get_latest_available_report_timestamp,
    scrape_injury_reports_for_dates,
)
from scraper import scrape_multiple_seasons

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
EXPORTS_DIR = DATA_DIR / "exports"
SCRAPE_EXPORTS_DIR = EXPORTS_DIR / "scrapes"
PREDICTOR_EXPORTS_DIR = EXPORTS_DIR / "predictor"
INJURY_JOBS: dict[str, dict[str, object]] = {}
INJURY_JOBS_LOCK = Lock()
MAX_INJURY_JOBS = 20

RAW_FILE = RAW_DIR / "bref_games_2016_2025.csv"
PROCESSED_FILE = PROCESSED_DIR / "games_2016_2025_normalized.csv"
PREDICTOR_START_SEASON = 2022
PREDICTOR_END_SEASON = 2025
PREDICTOR_RAW_FILE = RAW_DIR / f"bref_games_{PREDICTOR_START_SEASON}_{PREDICTOR_END_SEASON}.csv"
PREDICTOR_PROCESSED_FILE = (
    PROCESSED_DIR / f"games_{PREDICTOR_START_SEASON}_{PREDICTOR_END_SEASON}_normalized.csv"
)
PREDICTION_DISPLAY_TZ = ZoneInfo("America/Los_Angeles")
PREDICTION_HISTORY_PAGE_SIZE = 25
PREDICTION_TEAM_NAMES = {}
for team_name, team_code in TEAM_MAP_BREF.items():
    PREDICTION_TEAM_NAMES.setdefault(team_code, team_name)
APP_TITLE = "NBA Odds Predictor"
AUTHOR_NAME = "Brian Zeng"
AUTHOR_EMAIL = "bzeng0000@gmail.com"
AUTHOR_LINKEDIN = "https://www.linkedin.com/in/brianbzeng"
AUTHOR_GITHUB = "https://github.com/brianbzeng"

DEFAULT_START_SEASON = 2016
DEFAULT_END_SEASON = 2025
PREVIEW_OPTIONS = (5, 10, 25, 50, 100)
PREDICTOR_BENCHMARK_SUMMARY = {
    "label": "Final project benchmark",
    "split": "train 2022-2024, test 2025",
    "home_baseline_accuracy": 0.5459,
    "logistic_accuracy": 0.6479,
    "random_forest_accuracy": 0.6451,
    "best_model": "Logistic regression",
    "logistic_vs_baseline_gain": 0.1020,
    "random_forest_vs_baseline_gain": 0.0992,
    "notes": (
        "These are the benchmark numbers from the standalone terminal project."
    ),
}

NAV_ITEMS = (
    {"endpoint": "home", "label": "Home"},
    {"endpoint": "leaderboard", "label": "Leaderboard"},
    {"endpoint": "predictor", "label": "Predictor"},
    {"endpoint": "scrape", "label": "Scrape"},
    {"endpoint": "about", "label": "About"},
    {"endpoint": "bayesian_elo", "label": "Inner-Workings"},
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

    source_path = RAW_FILE if RAW_FILE.exists() else PREDICTOR_RAW_FILE
    if not source_path.exists():
        raise FileNotFoundError(
            "No bundled game data was found. Use the Scrape page first."
        )

    raw_games = pd.read_csv(source_path, parse_dates = ["date"])
    processed = normalize_team_names(clean_games(raw_games))
    processed.to_csv(PROCESSED_FILE, index=False)
    return processed


def _predictor_dataset_is_valid(games: pd.DataFrame) -> bool:
    # Reject obviously broken cached predictor datasets before the model uses them.
    if games.empty or "date" not in games.columns or "season" not in games.columns:
        return False

    dates = pd.to_datetime(games["date"], errors = "coerce").dropna()
    seasons = pd.to_numeric(games["season"], errors = "coerce").dropna()
    if dates.empty or seasons.empty:
        return False

    if len(games) < 4000:
        return False

    if dates.nunique() < 100:
        return False

    return (
        int(seasons.min()) <= PREDICTOR_START_SEASON
        and int(seasons.max()) >= PREDICTOR_END_SEASON
    )


def load_predictor_baseline() -> pd.DataFrame:
    if PREDICTOR_PROCESSED_FILE.exists():
        processed = pd.read_csv(PREDICTOR_PROCESSED_FILE, parse_dates=["date"])
        if _predictor_dataset_is_valid(processed):
            return processed

    if not PREDICTOR_RAW_FILE.exists():
        raise FileNotFoundError(
            "Predictor training data is missing. Refresh predictor data on the Predictor page first."
        )

    raw_games = pd.read_csv(PREDICTOR_RAW_FILE, parse_dates=["date"])
    if not _predictor_dataset_is_valid(raw_games):
        raise ValueError(
            "Predictor game data looks stale or corrupted. Use Refresh predictor data to rebuild the full seasons."
        )
    processed = normalize_team_names(clean_games(raw_games))
    processed.to_csv(PREDICTOR_PROCESSED_FILE, index=False)
    return processed


def load_predictor_games() -> pd.DataFrame:
    baseline = load_predictor_baseline()
    updates = read_game_updates()
    if updates.empty:
        return baseline
    return pd.concat([baseline, normalize_team_names(clean_games(updates))], ignore_index=True)


def prediction_injury_details(game, saved_reports: dict, used_reports: bool) -> list[dict]:
    details = []
    for side in ("away", "home"):
        team = getattr(game, f"{side}_team")
        if used_reports:
            report = getattr(game, f"{side}_injury_report_utc")
            out_count = getattr(game, f"{side}_out_count")
        else:
            report, out_count = saved_reports.get((game.game_date, team), (None, None))
        entry = {"team": team}
        if pd.notna(report):
            local = report.tz_convert(PREDICTION_DISPLAY_TZ)
            eastern = report.tz_convert(NBA_TIMEZONE).tz_localize(None)
            entry.update(
                out_count=int(out_count),
                published=local.strftime("%b %d, %I:%M %p PT").replace(" 0", " "),
                url=report_pdf_url(eastern),
            )
        details.append(entry)
    return details


def load_prediction_rows(today: date, history: bool) -> list[dict]:
    slate = read_daily_predictions()
    if slate.empty:
        return []

    finals = {
        (game.date, TEAM_MAP_BREF[game.home_team], TEAM_MAP_BREF[game.away_team]): game
        for game in read_game_updates().itertuples(index=False)
    }
    saved_reports = {}
    if not history and LIVE_INJURIES_FILE.exists():
        injuries = pd.read_csv(LIVE_INJURIES_FILE, parse_dates=["game_date", "latest_report_datetime"])
        saved_reports = {
            (row.game_date, row.team): (
                row.latest_report_datetime.tz_localize(NBA_TIMEZONE, ambiguous=False, nonexistent="shift_forward"),
                row.out_count,
            )
            for row in injuries.itertuples(index=False)
        }
    rows = []
    for game in slate.sort_values("tipoff_utc", ascending=not history).itertuples(index=False):
        local_tipoff = game.tipoff_utc.tz_convert(PREDICTION_DISPLAY_TZ)
        if not history and local_tipoff.date() != today:
            continue
        final = finals.get((game.game_date, game.home_team, game.away_team))
        has_pick = pd.notna(game.home_win_probability)
        if history and (final is None or not has_pick):
            continue

        away, home = game.away_team, game.home_team
        pick = (home if game.home_win_probability > 0.5 else away) if has_pick else None
        winner = (home if final.home_pts > final.away_pts else away) if final is not None else None
        same_day_finals = getattr(game, "same_day_finals", None)
        has_inputs = has_pick and pd.notna(same_day_finals)
        saved_at = game.predicted_at_utc.tz_convert(PREDICTION_DISPLAY_TZ) if has_inputs else None
        rows.append({
            "day": local_tipoff.strftime("%a, %b %d").replace(" 0", " "),
            "tipoff": local_tipoff.strftime("%I:%M %p").lstrip("0") + " PT",
            "tipoff_iso": game.tipoff_utc.isoformat(),
            "status": "final" if final is not None else game.status,
            "away": PREDICTION_TEAM_NAMES.get(away, away),
            "home": PREDICTION_TEAM_NAMES.get(home, home),
            "away_chance": f"{1 - game.home_win_probability:.1%}" if has_pick else None,
            "home_chance": f"{game.home_win_probability:.1%}" if has_pick else None,
            "away_score": final.away_pts if final is not None else None,
            "home_score": final.home_pts if final is not None else None,
            "pick": PREDICTION_TEAM_NAMES.get(pick, pick),
            "winner": PREDICTION_TEAM_NAMES.get(winner, winner),
            "correct": pick == winner if has_pick and winner is not None else None,
            "same_day_finals": int(same_day_finals) if has_inputs else None,
            "saved_at": saved_at.strftime("%I:%M %p PT").lstrip("0") if saved_at is not None else None,
            "injuries": prediction_injury_details(game, saved_reports, has_inputs) if has_inputs or not history else [],
        })
    # Keep the upcoming picks at the top of today's board.
    return rows if history else sorted(rows, key=lambda row: row["status"] != "scheduled")


def save_scraped_games(start_season: int, end_season: int) -> tuple[pd.DataFrame, str]:
    games = scrape_multiple_seasons(start_season, end_season)
    export_name = f"games_{start_season}_{end_season}_{uuid4().hex[:8]}.csv"
    export_path = SCRAPE_EXPORTS_DIR / export_name
    games.to_csv(export_path, index=False)
    return games, export_name


def _sanitize_injury_slug(value: str) -> str:
    return (
        value.replace(":", "-")
        .replace(" ", "_")
        .replace("/", "-")
    )


def _load_game_dates_for_injury_seasons(start_season: int, end_season: int) -> pd.Series:
    exact_path = RAW_DIR / f"bref_games_{start_season}_{end_season}.csv"
    games = pd.read_csv(exact_path, parse_dates=["date"]) if exact_path.exists() else load_predictor_games()
    season_games = games[
        (games["season"] >= start_season) & (games["season"] <= end_season)
    ].copy()
    if season_games.empty:
        raise ValueError(f"No games were found for seasons {start_season}-{end_season}.")

    return pd.to_datetime(season_games["date"]).dt.normalize()


def save_latest_injury_report() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    latest_timestamp = get_latest_available_report_timestamp()
    if latest_timestamp is None:
        raise ValueError("No recent official NBA injury reports were found.")

    detail_df = fetch_report_with_retries(latest_timestamp)
    if detail_df is None or detail_df.empty:
        raise ValueError(
            f"The latest available injury report for {latest_timestamp.date()} could not be loaded."
        )

    detail_df = detail_df.copy().rename(
        columns = {
            "Game Date": "game_date",
            "Game Time": "game_time",
            "Matchup": "matchup",
            "Team": "team",
            "Player Name": "player_name",
            "Current Status": "status",
            "Reason": "reason",
        }
    )
    detail_df["game_date"] = pd.to_datetime(detail_df["game_date"]).dt.normalize()
    detail_df["report_datetime"] = pd.Timestamp(latest_timestamp)
    team_df = aggregate_team_injuries(detail_df)

    slug = _sanitize_injury_slug(latest_timestamp.strftime("%Y-%m-%d_%H-%M"))
    detail_name = f"official_nba_injuries_detailed_latest_{slug}.csv"
    team_name = f"official_nba_injuries_by_team_latest_{slug}.csv"
    detail_path = SCRAPE_EXPORTS_DIR / detail_name
    team_path = SCRAPE_EXPORTS_DIR / team_name
    detail_df.to_csv(detail_path, index = False)
    team_df.to_csv(team_path, index = False)

    return detail_df, team_df, {
        "detail_name": detail_name,
        "team_name": team_name,
        "report_date": latest_timestamp.strftime("%Y-%m-%d"),
    }


def save_injury_reports_for_date_range(
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if start_ts > end_ts:
        raise ValueError("Injury start date cannot be after the end date.")

    game_dates = list(pd.date_range(start_ts, end_ts, freq = "D"))
    detail_df, missing_dates = scrape_injury_reports_for_dates(game_dates)
    if detail_df.empty:
        if start_ts == end_ts:
            raise ValueError(f"No official NBA injury report was found for {start_ts.date()}.")
        raise ValueError(
            f"No official NBA injury reports were found between {start_ts.date()} and {end_ts.date()}."
        )

    team_df = aggregate_team_injuries(detail_df)
    slug = f"{start_ts.strftime('%Y-%m-%d')}_{end_ts.strftime('%Y-%m-%d')}"
    detail_name = f"official_nba_injuries_detailed_{slug}.csv"
    team_name = f"official_nba_injuries_by_team_{slug}.csv"
    detail_path = SCRAPE_EXPORTS_DIR / detail_name
    team_path = SCRAPE_EXPORTS_DIR / team_name
    detail_df.to_csv(detail_path, index = False)
    team_df.to_csv(team_path, index = False)

    return detail_df, team_df, {
        "detail_name": detail_name,
        "team_name": team_name,
        "missing_dates": ", ".join(missing_dates[:10]) if missing_dates else "",
    }


def save_injury_reports_for_season_range(
    start_season: int,
    end_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if start_season > end_season:
        raise ValueError("Injury start season cannot be after the end season.")
    if start_season < 2022:
        raise ValueError("The nbainjuries-backed injury history in this app only starts in 2022.")

    game_dates = _load_game_dates_for_injury_seasons(start_season, end_season)
    detail_df, missing_dates = scrape_injury_reports_for_dates(list(game_dates))
    if detail_df.empty:
        raise ValueError(f"No official NBA injury reports were found for seasons {start_season}-{end_season}.")

    team_df = aggregate_team_injuries(detail_df)
    detail_name = f"official_nba_injuries_detailed_{start_season}_{end_season}.csv"
    team_name = f"official_nba_injuries_by_team_{start_season}_{end_season}.csv"
    detail_path = SCRAPE_EXPORTS_DIR / detail_name
    team_path = SCRAPE_EXPORTS_DIR / team_name
    detail_df.to_csv(detail_path, index = False)
    team_df.to_csv(team_path, index = False)

    return detail_df, team_df, {
        "detail_name": detail_name,
        "team_name": team_name,
        "missing_dates": ", ".join(missing_dates[:10]) if missing_dates else "",
    }


def _run_injury_scrape_job(
    job_id: str,
    mode: str,
    parameters: dict[str, object],
) -> None:
    try:
        if mode == "latest":
            _, team_df, meta = save_latest_injury_report()
            message = (
                f"Saved the latest available official injury report "
                f"for {meta['report_date']}."
            )
        elif mode == "date_range":
            start_date = str(parameters["start_date"])
            end_date = str(parameters["end_date"])
            _, team_df, meta = save_injury_reports_for_date_range(start_date, end_date)
            message = f"Saved official injury reports from {start_date} to {end_date}."
        elif mode == "season_range":
            start_season = int(parameters["start_season"])
            end_season = int(parameters["end_season"])
            _, team_df, meta = save_injury_reports_for_season_range(
                start_season,
                end_season,
            )
            message = (
                f"Saved official injury reports for seasons "
                f"{start_season}-{end_season}."
            )
        else:
            raise ValueError("Unknown injury scrape mode.")

        missing_dates = meta.get("missing_dates", "")
        if missing_dates:
            message += f" Missing dates skipped: {missing_dates}."

        preview_frame = team_df.head(10).copy()
        if not preview_frame.empty and "game_date" in preview_frame.columns:
            preview_frame["game_date"] = pd.to_datetime(
                preview_frame["game_date"]
            ).dt.strftime("%Y-%m-%d")

        result = {
            "status": "complete",
            "message": message,
            "preview": preview_frame.to_dict(orient = "records"),
            "detail_name": meta["detail_name"],
            "team_name": meta["team_name"],
        }
    except Exception as exc:  # noqa: BLE001 - surfaced through the job status
        result = {"status": "error", "error": str(exc)}

    with INJURY_JOBS_LOCK:
        INJURY_JOBS[job_id].update(result)


def start_injury_scrape_job(
    mode: str,
    parameters: dict[str, object],
) -> str:
    job_id = uuid4().hex
    with INJURY_JOBS_LOCK:
        while len(INJURY_JOBS) >= MAX_INJURY_JOBS:
            INJURY_JOBS.pop(next(iter(INJURY_JOBS)))
        INJURY_JOBS[job_id] = {
            "status": "running",
            "mode": mode,
            "parameters": parameters,
        }

    Thread(
        target = _run_injury_scrape_job,
        args = (job_id, mode, parameters),
        daemon = True,
    ).start()
    return job_id


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
    rating_span = float(standings["rating"].max() - standings["rating"].min())
    spread = rating_span if rating_span > 1e-9 else 1.0
    standings["rating_pct"] = (
        ((standings["rating"] - standings["rating"].min()) / spread * 100)
        .clip(0, 100)
        .round(1)
    )
    return standings[["rank", "team", "rating", "rating_pct"]]


def get_live_season_end_year(reference_date: datetime | None = None) -> int:
    current = reference_date or datetime.now()
    if current.month >= 10:
        return current.year + 1
    return current.year


def refresh_live_leaderboard() -> tuple[pd.DataFrame, int, bool]:
    _, changed = refresh_game_updates()
    try:
        refresh_injury_updates()
    except Exception:
        logging.getLogger(__name__).exception("Official NBA injury refresh failed")
    games = get_latest_live_games()
    season = int(games["season"].max()) if not games.empty else get_live_season_end_year()
    return build_leaderboard_frame(games), season, not changed


def get_latest_live_games() -> pd.DataFrame:
    updates = read_game_updates()
    if not updates.empty:
        games = normalize_team_names(clean_games(updates))
    else:
        try:
            games = load_processed_games()
        except FileNotFoundError:
            return pd.DataFrame(columns=["date", "season", "home_team", "away_team", "home_pts", "away_pts", "home_win", "margin"])
    return games[games["season"] == games["season"].max()].copy()


def load_home_stats() -> dict[str, object]:
    try:
        games = get_latest_live_games()
    except FileNotFoundError:
        return {
            "game_count": 0,
            "team_count": 0,
            "season_count": 0,
            "latest_season": None,
            "latest_season_games": 0,
            "last_updated": None,
        }

    latest_season = int(games["season"].max()) if not games.empty else None
    latest_season_games = int((games["season"] == latest_season).sum()) if latest_season else 0
    team_count = int(len(set(games["home_team"]).union(set(games["away_team"]))))
    last_updated = (
        pd.Timestamp(games["date"].max()).strftime("%Y-%m-%d") if not games.empty else None
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
    bundled_data = {
        LIVE_GAMES_FILE,
        LIVE_GAMES_FILE.with_suffix(".lock"),
        LIVE_INJURIES_FILE,
        LIVE_INJURIES_FILE.with_suffix(".lock"),
        DAILY_PREDICTIONS_FILE,
        DAILY_PREDICTIONS_FILE.with_suffix(".lock"),
        PREDICTOR_RAW_FILE,
        RAW_DIR / f"official_nba_injuries_by_team_{PREDICTOR_START_SEASON}_{PREDICTOR_END_SEASON}.csv",
        RAW_DIR / f"official_nba_injuries_detailed_{PREDICTOR_START_SEASON}_{PREDICTOR_END_SEASON}.csv",
    }

    for directory in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, EXPORTS_DIR):
        if not directory.exists():
            continue

        for path in list(directory.iterdir()):
            if path in bundled_data:
                continue
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
    static_versions = {
        name: sha256((Path(app.static_folder) / name).read_bytes()).hexdigest()[:12]
        for name in ("styles.css", "app.js")
    }
    app.secret_key = os.environ.get("SECRET_KEY", "nbamodel-local-dev")
    app.config["TEMPLATES_AUTO_RELOAD"] = os.environ.get("FLASK_DEBUG") == "1"
    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for = 1,
        x_proto = 1,
        x_host = 1,
        x_prefix = 1,
    )

    @app.context_processor
    def inject_nav():
        return {"nav_items": NAV_ITEMS, "app_title": APP_TITLE}

    @app.template_global()
    def static_asset_url(filename: str) -> str:
        return url_for("static", filename=filename, v=static_versions[filename])

    @app.get("/")
    def home():
        stats = load_home_stats()
        return render_template("home.html", stats=stats)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/about")
    def about():
        return render_template(
            "about.html",
            author_name = AUTHOR_NAME,
            author_email = AUTHOR_EMAIL,
            author_linkedin = AUTHOR_LINKEDIN,
            author_github = AUTHOR_GITHUB,
        )

    @app.route("/leaderboard", methods=["GET"])
    def leaderboard():
        try:
            games = get_latest_live_games()
            standings = build_leaderboard_frame(games)
            latest_season = int(games["season"].max()) if not games.empty else None
            last_updated = pd.Timestamp(games["date"].max()).strftime("%Y-%m-%d") if not games.empty else None
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
        try:
            _, season, already_current = refresh_live_leaderboard()
        except Exception:
            app.logger.exception("Leaderboard game refresh failed")
            flash("Could not refresh NBA scores. Showing the saved games.", "error")
            return redirect(url_for("leaderboard"))

        season_label = f"{season - 1}-{str(season)[-2:]}"
        if already_current:
            flash(
                f"The {season_label} leaderboard is already up to date.",
                "info",
            )
        else:
            flash(
                f"Leaderboard refreshed with completed games from {season_label}.",
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
        injury_message = None
        injury_error = None
        injury_preview = []
        injury_detail_name = None
        injury_team_name = None
        injury_detail_url = None
        injury_team_url = None
        injury_mode = "latest"
        injury_start_date = ""
        injury_end_date = ""
        injury_start_season = 2022
        injury_end_season = PREDICTOR_END_SEASON
        injury_job_id = request.args.get("injury_job", "")
        injury_job_running = False
        current_season_note = (
            "Scraping the active season only pulls games that have already tipped off and "
            "finished. Anything scheduled but not yet played is left out, so future results "
            "never leak into the model. The leaderboard picks up the live Elo state of that "
            "current season; if the season hasn't started yet, it falls back to the most "
            "recent completed season's ratings."
        )
        injury_note = (
            "Injury reports come from the official NBA injury feed, pulled through the "
            "nbainjuries library. Historical coverage that lines up cleanly with the scraped "
            "game calendar starts in the 2021-22 season; earlier reports aren't reliable "
            "enough to feed the predictor, so the season-range importer floors out at 2022."
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
                action = request.form.get("scrape_action", "games")
                preview_limit = parse_preview_limit(request.form.get("preview_limit"))

                if action == "games":
                    start_season = int(request.form.get("start_season", DEFAULT_START_SEASON))
                    end_season = int(request.form.get("end_season", DEFAULT_END_SEASON))
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
                else:
                    injury_mode = request.form.get("injury_mode", "latest")
                    if injury_mode == "latest":
                        parameters = {}
                    elif injury_mode == "date_range":
                        injury_start_date = request.form.get("injury_start_date", "")
                        injury_end_date = request.form.get("injury_end_date", "")
                        if not injury_start_date or not injury_end_date:
                            raise ValueError("Choose both a start date and an end date.")
                        parameters = {
                            "start_date": injury_start_date,
                            "end_date": injury_end_date,
                        }
                    elif injury_mode == "season_range":
                        injury_start_season = int(request.form.get("injury_start_season", 2022))
                        injury_end_season = int(
                            request.form.get("injury_end_season", PREDICTOR_END_SEASON)
                        )
                        parameters = {
                            "start_season": injury_start_season,
                            "end_season": injury_end_season,
                        }
                    else:
                        raise ValueError("Unknown injury scrape mode.")

                    injury_job_id = start_injury_scrape_job(injury_mode, parameters)
                    return redirect(url_for("scrape", injury_job = injury_job_id))
            except Exception as exc:  # noqa: BLE001 - user-facing page feedback
                if request.form.get("scrape_action", "games") == "games":
                    error = str(exc)
                else:
                    injury_error = str(exc)

        if injury_job_id:
            with INJURY_JOBS_LOCK:
                injury_job = dict(INJURY_JOBS.get(injury_job_id, {}))
            if injury_job:
                injury_mode = str(injury_job.get("mode", injury_mode))
                parameters = dict(injury_job.get("parameters", {}))
                injury_start_date = str(parameters.get("start_date", ""))
                injury_end_date = str(parameters.get("end_date", ""))
                injury_start_season = int(parameters.get("start_season", 2022))
                injury_end_season = int(
                    parameters.get("end_season", PREDICTOR_END_SEASON)
                )
            if injury_job.get("status") == "running":
                injury_job_running = True
            elif injury_job.get("status") == "complete":
                injury_message = str(injury_job["message"])
                injury_preview = list(injury_job["preview"])
                injury_detail_name = str(injury_job["detail_name"])
                injury_team_name = str(injury_job["team_name"])
                injury_detail_url = url_for(
                    "download_scrape_export",
                    filename = injury_detail_name,
                )
                injury_team_url = url_for(
                    "download_scrape_export",
                    filename = injury_team_name,
                )
            elif injury_job.get("status") == "error":
                injury_error = str(injury_job["error"])
            else:
                injury_error = "That injury scrape job is no longer available."

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
            injury_note=injury_note,
            injury_message=injury_message,
            injury_error=injury_error,
            injury_preview=injury_preview,
            injury_detail_name=injury_detail_name,
            injury_team_name=injury_team_name,
            injury_detail_url=injury_detail_url,
            injury_team_url=injury_team_url,
            injury_mode=injury_mode,
            injury_start_date=injury_start_date,
            injury_end_date=injury_end_date,
            injury_start_season=injury_start_season,
            injury_end_season=injury_end_season,
            injury_job_id=injury_job_id,
            injury_job_running=injury_job_running,
            preview_options=PREVIEW_OPTIONS,
        )

    @app.get("/scrape/injury-jobs/<job_id>")
    def injury_job_status(job_id: str):
        with INJURY_JOBS_LOCK:
            job = INJURY_JOBS.get(job_id)
            if job is None:
                return {"status": "missing"}, 404
            return {"status": job["status"]}

    @app.get("/predictor")
    def predictor():
        today = datetime.now(PREDICTION_DISPLAY_TZ).date()
        history = request.args.get("view") == "history"
        rows = load_prediction_rows(today, history)
        correct = sum(row["correct"] is True for row in rows)
        total = len(rows)
        page_count = max(1, (total + PREDICTION_HISTORY_PAGE_SIZE - 1) // PREDICTION_HISTORY_PAGE_SIZE)
        page = min(max(1, request.args.get("page", 1, type=int)), page_count) if history else 1
        if history:
            start = (page - 1) * PREDICTION_HISTORY_PAGE_SIZE
            rows = rows[start:start + PREDICTION_HISTORY_PAGE_SIZE]

        return render_template(
            "predictor.html",
            history=history,
            today=today.strftime("%A, %B %d").replace(" 0", " "),
            rows=rows,
            total=total,
            correct=correct,
            page=page,
            page_count=page_count,
        )

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
        return render_template("bayesian_elo.html", benchmark=PREDICTOR_BENCHMARK_SUMMARY)

    @app.get("/downloads/scrapes/<path:filename>")
    def download_scrape_export(filename: str):
        return send_from_directory(SCRAPE_EXPORTS_DIR, filename, as_attachment=True)

    @app.get("/downloads/predictor/<path:filename>")
    def download_predictor_export(filename: str):
        return send_from_directory(PREDICTOR_EXPORTS_DIR, filename, as_attachment = True)

    return app


if __name__ == "__main__":
    create_app().run(
        host = os.environ.get("HOST", "0.0.0.0"),
        port = int(os.environ.get("PORT", "5000")),
        debug = os.environ.get("FLASK_DEBUG") == "1",
    )
