"""Compare random forest feature variants on the two latest saved seasons."""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from cleaner import clean_games, normalize_team_names
from daily_predictions import BASELINE_GAMES_FILE
from elo_features import ELO_DIFF_COLUMN, build_elo_features
from game_updates import FIRST_UPDATE_DATE, GAME_KEY, read_game_updates
from prediction_replay import _load_replay_injuries
from predictor import (
    _fit_elo_random_forest,
    _fit_no_rest_random_forest,
    _fit_random_forest,
    build_pregame_features,
    merge_injury_features,
)


CHALLENGERS = {
    "elo": ("Elo", _fit_elo_random_forest),
    "no-rest": ("No rest", _fit_no_rest_random_forest),
}


def score_holdout(features: pd.DataFrame, season: int, challenger: str = "elo") -> pd.DataFrame:
    train = features.loc[features["season"] < season]
    test = features.loc[features["season"] == season]
    baseline = _fit_random_forest(train)
    _, fit_challenger = CHALLENGERS[challenger]
    variant = fit_challenger(train)
    results = test[["date", "season", "home_win"]].copy()
    for column, model in (("baseline", baseline), (challenger, variant)):
        results[column] = np.round(model.predict_proba(test[list(model.feature_names_in_)])[:, 1], 6)
    return results


def accuracy_interval(results: pd.DataFrame, correct_delta: np.ndarray) -> tuple[float, float]:
    """Resample paired predictions by date, independently within each season."""
    samples = 10_000
    rng = np.random.default_rng(100)
    gains = np.zeros(samples)
    totals = np.zeros(samples)
    for _, season in results.assign(correct_delta=correct_delta).groupby("season", sort=True):
        daily = season.groupby("date")["correct_delta"].agg(["sum", "size"])
        sampled = rng.integers(len(daily), size=(samples, len(daily)))
        gains += daily["sum"].to_numpy()[sampled].sum(axis=1)
        totals += daily["size"].to_numpy()[sampled].sum(axis=1)
    low, high = np.quantile(gains / totals * 100, [0.025, 0.975])
    return float(low), float(high)


def report(label: str, results: pd.DataFrame, challenger: str = "elo") -> None:
    actual = results["home_win"].to_numpy()
    baseline_correct = (results["baseline"].to_numpy() > 0.5) == actual
    variant_correct = (results[challenger].to_numpy() > 0.5) == actual
    correct_delta = variant_correct.astype(int) - baseline_correct.astype(int)
    low, high = accuracy_interval(results, correct_delta)
    challenger_name, _ = CHALLENGERS[challenger]
    print(f"{label} | {len(results):,} games")
    for name, column, correct in (
        ("Baseline", "baseline", baseline_correct), (challenger_name, challenger, variant_correct),
    ):
        odds = results[column]
        print(f"  {name:<8} {correct.sum():,}/{len(results):,} correct ({correct.mean():.2%})"
              f"   Brier {brier_score_loss(actual, odds):.5f}   Log loss {log_loss(actual, odds, labels=[0, 1]):.5f}")
    print(f"  {challenger_name} flips {(correct_delta > 0).sum():,} to correct and {(correct_delta < 0).sum():,} to incorrect")
    print(f"  Paired accuracy difference {correct_delta.mean() * 100:+.2f} pp"
          f"   95% date-bootstrap interval [{low:+.2f}, {high:+.2f}] pp")


def evaluate(through: pd.Timestamp | None, challenger: str = "elo") -> None:
    raw = pd.concat([pd.read_csv(BASELINE_GAMES_FILE), read_game_updates()], ignore_index=True)
    games = normalize_team_names(clean_games(raw)).drop_duplicates(GAME_KEY, keep="last")
    if through is not None:
        games = games.loc[games["date"] <= through]
    games = games.sort_values(GAME_KEY, ignore_index=True)
    seasons = sorted(games["season"].unique())
    if len(seasons) < 3:
        raise ValueError("Need at least three saved seasons for two forward holdouts.")
    holdouts = seasons[-2:]
    first_holdout = games.loc[games["season"] == holdouts[0], "date"].min()
    if first_holdout.date() < FIRST_UPDATE_DATE:
        raise ValueError("Need two holdout seasons covered by the verified pregame injury archive.")

    # Holdouts use the archived team reports verified to precede each tipoff.
    injuries = _load_replay_injuries(games, first_holdout)
    features = merge_injury_features(build_pregame_features(games), injuries)
    if challenger == "elo":
        elo, _, _ = build_elo_features(games)
        features = features.merge(elo[[*GAME_KEY, ELO_DIFF_COLUMN]], on=GAME_KEY, validate="one_to_one")
        if len(features) != len(games):
            raise ValueError("The Elo feature does not cover every saved game.")
    features = features.sort_values(GAME_KEY, ignore_index=True)

    results = []
    for season in holdouts:
        fold = score_holdout(features, season, challenger)
        results.append(fold)
        train = games.loc[games["season"] < season]
        report(f"{season - 1}-{str(season)[-2:]} through {fold['date'].max():%Y-%m-%d}"
               f" | trained on {len(train):,} games through {train['date'].max():%Y-%m-%d}", fold, challenger)
    report("Combined forward holdouts", pd.concat(results, ignore_index=True), challenger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=pd.Timestamp, help="Latest game date to include (YYYY-MM-DD).")
    parser.add_argument("--challenger", choices=CHALLENGERS, default="elo")
    args = parser.parse_args()
    evaluate(args.through, args.challenger)
