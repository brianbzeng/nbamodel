"""Compare the player adjustment with the same random forest on later games."""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, log_loss

from cleaner import clean_games, normalize_team_names
from daily_predictions import BASELINE_GAMES_FILE
from game_updates import GAME_KEY, read_game_updates
from player_impact import PLAYER_IMPACT_WEIGHT, adjust_win_probabilities, matchup_player_losses
from predictor import _fit_random_forest, build_pregame_features, load_optional_team_injuries, merge_injury_features


def score_periods(features: pd.DataFrame, targets: pd.DataFrame, periods: pd.Series) -> pd.DataFrame:
    predictions = []
    for _, games in targets.groupby(periods):
        earlier = features[features["date"] < games["date"].min()]
        forest = _fit_random_forest(earlier)
        probability = forest.predict_proba(games[list(forest.feature_names_in_)])[:, 1]
        predictions.append(games.assign(probability=probability))
    return pd.concat(predictions, ignore_index=True)


def adjusted(picks: pd.DataFrame, weight: float) -> np.ndarray:
    return adjust_win_probabilities(picks["probability"], picks["player_loss_difference"], weight)


def evaluate(through: pd.Timestamp | None) -> None:
    raw = pd.concat([pd.read_csv(BASELINE_GAMES_FILE), read_game_updates()], ignore_index=True)
    games = normalize_team_names(clean_games(raw)).drop_duplicates(GAME_KEY, keep="last")
    if through is not None:
        games = games[games["date"] <= through]
    games = games.sort_values(GAME_KEY, ignore_index=True)
    injuries = load_optional_team_injuries(games)
    features = merge_injury_features(build_pregame_features(games), injuries)
    features = features.sort_values(GAME_KEY, ignore_index=True)
    season = int(features["season"].max())
    targets = features[features["season"].isin([season - 1, season])].copy()
    matchups = targets.rename(columns={"date": "game_date"})
    targets["player_loss_difference"] = matchup_player_losses(matchups, injuries)
    calibration = targets[targets["season"] == season - 1]
    holdout = targets[targets["season"] == season]

    # Every calibration month is scored by a forest trained on earlier dates.
    calibration = score_periods(features, calibration, calibration["date"].dt.to_period("M"))

    def objective(weight):
        return log_loss(calibration["home_win"], adjusted(calibration, weight), labels=[0, 1])

    optimum = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    fitted_weight = optimum.x if optimum.fun < objective(0) else 0.0
    print(f"Fitted weight on season ending {season - 1} = {fitted_weight:.5f}")
    print(f"Evaluating the experimental weight of {PLAYER_IMPACT_WEIGHT:.5f} on season ending {season}.")

    for label, periods in (("Fixed", holdout["season"]), ("Daily refits", holdout["date"])):
        scored = score_periods(features, holdout, periods)
        actual = scored["home_win"].to_numpy()
        print(f"{label}, {len(scored)} games")
        variants = (("Baseline", scored["probability"]), ("Player impact", adjusted(scored, PLAYER_IMPACT_WEIGHT)))
        for variant, probabilities in variants:
            correct = ((np.round(probabilities, 6) > 0.5) == actual).sum()
            brier = brier_score_loss(actual, probabilities)
            loss = log_loss(actual, probabilities, labels=[0, 1])
            print(f"  {variant:<13} {correct}/{len(scored)} correct, Brier {brier:.5f}, log loss {loss:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=pd.Timestamp, help="Latest game date to include (YYYY-MM-DD).")
    evaluate(parser.parse_args().through)
