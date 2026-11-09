"""Load official player-level injury snapshots published before tipoff."""

from pathlib import Path

import pandas as pd


RAW_DIR = Path(__file__).resolve().parent / "data/raw"
LIVE_PLAYER_INJURIES_FILE = RAW_DIR / "official_nba_injuries_players_live.csv"
PLAYER_INJURY_KEY = ["game_date", "team", "report_datetime"]
PLAYER_INJURY_COLUMNS = ["game_date", "team", "player_name", "status", "report_datetime"]
PLAYER_STATUSES = {"OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE", "AVAILABLE"}
HEALTHY_REASONS = {"NO INJURIES TO REPORT", "NO INJURIES REPORTED"}


def submitted_injury_rows(report: pd.DataFrame) -> pd.DataFrame:
    """Keep listed players and explicit healthy reports, dropping pending reports."""
    if report.empty:
        return report

    statuses = report["status"].fillna("").str.strip().str.upper()
    reasons = report["reason"].fillna("").str.strip().str.upper()
    names = report["player_name"].fillna("").str.strip()
    players = statuses.isin(PLAYER_STATUSES) & names.ne("")
    healthy = reasons.isin(HEALTHY_REASONS)
    submitted = report.loc[players | healthy].copy()
    submitted["player_name"] = names
    submitted["status"] = statuses
    submitted.loc[healthy, ["player_name", "status"]] = ""
    return submitted


def read_live_player_injury_reports(path: Path = LIVE_PLAYER_INJURIES_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PLAYER_INJURY_COLUMNS)

    return pd.read_csv(path, keep_default_na=False, parse_dates=["game_date", "report_datetime"])
