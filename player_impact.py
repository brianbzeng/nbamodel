"""Estimate the production lost to reported player absences before tipoff."""

from datetime import timedelta
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.special import expit, logit


# A conservative baseline of seven Game Score points per 36 minutes.
REPLACEMENT_PER_MINUTE = 7 / 36
RATING_LOOKBACK_DAYS = 365
RATING_GAMES = 30
PRIOR_GAMES = 8
STATUS_ABSENCE_WEIGHTS = {"OUT": 1.0, "DOUBTFUL": 0.75, "QUESTIONABLE": 0.45, "PROBABLE": 0.15}
PLAYER_IMPACT_MODEL = "random_forest_player_impact"
# Fitted against out-of-sample random forest picks from the 2025–26 season.
PLAYER_IMPACT_WEIGHT = 0.02441


def _name_key(name: str) -> str:
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first} {last}"
    name = unicodedata.normalize("NFKD", name.casefold().translate(str.maketrans("øłđ", "old")))
    name = "".join(char for char in name if not unicodedata.combining(char))
    name = re.sub(r"[.,]", "", name.strip())
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name)
    return "".join(char for char in name if char.isalnum())


def rate_reported_players(player_games: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Rate each reported player from earlier appearances, across seasons and trades."""
    result = reports.copy()
    result["player_value"] = np.nan
    if player_games.empty or result.empty:
        return result

    games = player_games.copy()
    games["name_key"] = games["player"].map(_name_key)
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()
    result["name_key"] = result["player_name"].fillna("").map(_name_key)
    games = games[games["name_key"].isin(result["name_key"])].sort_values("date")
    games["above_replacement"] = games["game_score"] - REPLACEMENT_PER_MINUTE * games["minutes"]
    dates = pd.to_datetime(result["game_date"]).dt.normalize().to_numpy(dtype="datetime64[D]")
    values = np.full(len(result), np.nan)

    # Key by normalized name for the injury join and by stable ID for the rating.
    histories = {
        name: [(rows["date"].to_numpy(dtype="datetime64[D]"), rows["team"].to_numpy(),
                np.r_[0.0, rows["above_replacement"].cumsum().to_numpy()])
               for _, rows in matches.groupby("player_id", sort=False)]
        for name, matches in games.groupby("name_key", sort=False)
    }
    for index, (name, team, day) in enumerate(zip(result["name_key"], result["team"], dates)):
        eligible = []
        for game_days, teams, cumulative in histories.get(name, ()):
            end = game_days.searchsorted(day, side="left")
            start = max(end - RATING_GAMES, game_days.searchsorted(day - np.timedelta64(RATING_LOOKBACK_DAYS, "D")))
            if end > start:
                value = (cumulative[end] - cumulative[start]) / (end - start + PRIOR_GAMES)
                eligible.append((teams[end - 1], max(value, 0.0)))
        if len(eligible) == 1:
            values[index] = eligible[0][1]
        elif eligible:
            same_team = [value for last_team, value in eligible if last_team == team]
            if len(same_team) == 1:
                values[index] = same_team[0]

    return result.drop(columns="name_key").assign(player_value=values)


def team_player_losses(player_games: pd.DataFrame, reports: pd.DataFrame) -> pd.Series:
    """Sum status-weighted player values from each team's latest eligible report."""
    reports = reports.copy()
    reports["game_date"] = pd.to_datetime(reports["game_date"]).dt.normalize()
    keys = ["game_date", "team"]
    latest = reports.groupby(keys)["report_datetime"].transform("max")
    reports = reports.loc[reports["report_datetime"].eq(latest)].drop_duplicates([*keys, "player_name"])
    reports = reports.reset_index(drop=True)
    weights = reports["status"].str.strip().str.upper().map(STATUS_ABSENCE_WEIGHTS)
    rated = rate_reported_players(player_games, reports.loc[weights.notna()])
    return (rated["player_value"] * weights).groupby([reports[key] for key in keys]).sum(min_count=1)


def player_loss_difference(matchups: pd.DataFrame, losses: pd.Series) -> np.ndarray:
    """Away losses minus home losses, using exact matchup dates and teams."""
    date_column = "game_date" if "game_date" in matchups else "date"
    dates = pd.to_datetime(matchups[date_column]).dt.normalize()
    home, away = (
        losses.reindex(pd.MultiIndex.from_arrays([dates, matchups[f"{side}_team"]])).fillna(0).to_numpy()
        for side in ("home", "away")
    )
    return away - home


def matchup_player_losses(matchups: pd.DataFrame, team_reports: pd.DataFrame | None) -> np.ndarray:
    """Use only player details from the same official reports as the team inputs."""
    from player_injuries import read_live_player_injury_reports
    from player_stats import read_player_game_stats

    if team_reports is None or team_reports.empty:
        return np.zeros(len(matchups))

    report_keys = ["game_date", "team", "report_datetime"]
    dates = pd.to_datetime(matchups["game_date"]).dt.normalize()
    teams = set(matchups["home_team"]) | set(matchups["away_team"])
    sources = team_reports.loc[team_reports["game_date"].isin(dates) & team_reports["team"].isin(teams)]
    sources = sources.rename(columns={"latest_report_datetime": "report_datetime"})[report_keys]
    if sources.empty:
        return np.zeros(len(matchups))
    reports = read_live_player_injury_reports()
    if reports.empty:
        return np.zeros(len(matchups))
    reports = reports.merge(sources, on=report_keys)
    if reports.empty:
        return np.zeros(len(matchups))

    player_games = read_player_game_stats()
    game_days = pd.to_datetime(player_games["date"])
    earliest = dates.min() - timedelta(days=RATING_LOOKBACK_DAYS)
    eligible = game_days.ge(earliest) & game_days.lt(dates.max())
    losses = team_player_losses(player_games.loc[eligible], reports)
    return player_loss_difference(matchups, losses)


def adjust_win_probabilities(probabilities, loss_difference, weight: float) -> np.ndarray:
    """Apply a calibrated positive injury effect on the model's log odds."""
    probabilities = np.asarray(probabilities, dtype=float)
    adjustment = weight * np.asarray(loss_difference, dtype=float)
    return np.where(adjustment == 0, probabilities, expit(logit(probabilities) + adjustment))


def player_adjusted_picks(
    matchups: list[dict], probabilities: dict[str, float], team_reports: pd.DataFrame | None,
) -> dict[str, float]:
    game_ids = [str(matchup["game_id"]) for matchup in matchups]
    losses = matchup_player_losses(pd.DataFrame(matchups), team_reports)
    adjusted = adjust_win_probabilities([probabilities[game_id] for game_id in game_ids], losses, PLAYER_IMPACT_WEIGHT)
    return dict(zip(game_ids, adjusted))
