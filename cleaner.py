"""Data cleaning helpers for the NBA odds predictor."""

import pandas as pd


TEAM_MAP_BREF = {
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


def clean_games(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize game results into the shape expected by the Elo model."""

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="raise").astype(int)
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="raise").astype(int)
    df["home_win"] = (df["home_pts"] > df["away_pts"]).astype(int)
    df["margin"] = (df["home_pts"] - df["away_pts"]).abs().astype(int)
    return df.sort_values("date").reset_index(drop=True)