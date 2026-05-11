# cleaner.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"


def clean_games(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date  # Python date objects
    df["home_win"] = (df["home_pts"] > df["away_pts"]).astype(int)
    df["margin"] = (df["home_pts"] - df["away_pts"]).abs()
    df = df.sort_values("date").reset_index(drop=True)
    return df


TEAM_MAP_BREF = {
    # Modern teams (you'll mostly see these 2015+)
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",

    # Older names (for safety if you ever go pre-2010+)
    "Charlotte Bobcats": "CHA",
    "New Jersey Nets": "BRK",
    "New Orleans Hornets": "NOP",
    "Seattle SuperSonics": "OKC",
    "Washington Bullets": "WAS",
}

def normalize_team_names(df: pd.DataFrame, mapping=TEAM_MAP_BREF) -> pd.DataFrame:
    df = df.copy()
    for col in ["home_team", "away_team"]:
        abbr_col = col + "_abbr"
        df[abbr_col] = df[col].map(mapping)
        if df[abbr_col].isnull().any():
            unknown = df.loc[df[abbr_col].isnull(), col].unique()
            raise ValueError(f"Unknown team names found in '{col}': {unknown}")
    df["home_team"] = df["home_team_abbr"]
    df["away_team"] = df["away_team_abbr"]
    return df.drop(columns=["home_team_abbr", "away_team_abbr"])
