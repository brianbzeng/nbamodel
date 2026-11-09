"""Compare pregame challengers on the two latest saved seasons."""

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
    _fit_logistic_regression,
    _fit_margin_random_forest,
    _fit_no_rest_random_forest,
    _fit_random_forest,
    _fit_schedule_density_random_forest,
    build_pregame_features,
    merge_injury_features,
)
from recent_form import build_recent_form_features
from schedule_density import build_schedule_density_features


CHALLENGERS = {
    "elo": ("Elo", _fit_elo_random_forest),
    "no-rest": ("No rest", _fit_no_rest_random_forest),
    "margin": ("Point margin", _fit_margin_random_forest),
    "blend": ("RF/logistic blend", _fit_logistic_regression),
    "recent-form": ("Weighted form", _fit_random_forest),
    "schedule-density": ("Schedule density", _fit_schedule_density_random_forest),
}
NEW_CHALLENGERS = ("margin", "blend", "recent-form", "schedule-density")


def predict_holdout(features: pd.DataFrame, season: int, fit_model) -> np.ndarray:
    model = fit_model(features.loc[features["season"] < season])
    test = features.loc[features["season"] == season, list(model.feature_names_in_)]
    return model.predict_proba(test)[:, 1]


def score_holdout(
    features: pd.DataFrame,
    season: int,
    challenger: str = "elo",
    *,
    variant_features: pd.DataFrame | None = None,
    baseline_probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    if baseline_probabilities is None:
        baseline_probabilities = predict_holdout(features, season, _fit_random_forest)
    if variant_features is None:
        variant_features = features
    _, fit_challenger = CHALLENGERS[challenger]
    variant_probabilities = predict_holdout(variant_features, season, fit_challenger)
    if challenger == "blend":
        variant_probabilities = (baseline_probabilities + variant_probabilities) / 2
    results = features.loc[features["season"] == season, ["date", "season", "home_win"]].copy()
    results["baseline"] = np.round(baseline_probabilities, 6)
    results[challenger] = np.round(variant_probabilities, 6)
    return results


def paired_date_interval(results: pd.DataFrame, delta: np.ndarray) -> tuple[float, float]:
    """Resample paired predictions by date, independently within each season."""
    samples = 10_000
    rng = np.random.default_rng(100)
    gains = np.zeros(samples)
    totals = np.zeros(samples)
    for _, season in results.assign(delta=delta).groupby("season", sort=True):
        daily = season.groupby("date")["delta"].agg(["sum", "size"])
        sampled = rng.integers(len(daily), size=(samples, len(daily)))
        gains += daily["sum"].to_numpy()[sampled].sum(axis=1)
        totals += daily["size"].to_numpy()[sampled].sum(axis=1)
    low, high = np.quantile(gains / totals, [0.025, 0.975])
    return float(low), float(high)


def accuracy_interval(results: pd.DataFrame, correct_delta: np.ndarray) -> tuple[float, float]:
    return paired_date_interval(results, correct_delta * 100)


def report(label: str, results: pd.DataFrame, challenger: str = "elo") -> None:
    actual = results["home_win"].to_numpy()
    baseline_correct = (results["baseline"].to_numpy() > 0.5) == actual
    variant_correct = (results[challenger].to_numpy() > 0.5) == actual
    correct_delta = variant_correct.astype(int) - baseline_correct.astype(int)
    low, high = accuracy_interval(results, correct_delta)
    eps = np.finfo(float).eps
    baseline_actual = np.where(actual, results["baseline"], 1 - results["baseline"])
    variant_actual = np.where(actual, results[challenger], 1 - results[challenger])
    loss_delta = np.log(baseline_actual.clip(eps, 1)) - np.log(variant_actual.clip(eps, 1))
    loss_low, loss_high = paired_date_interval(results, loss_delta)
    challenger_name, _ = CHALLENGERS[challenger]
    print(f"{label} | {len(results):,} games")
    for name, column, correct in (
        ("Baseline", "baseline", baseline_correct), (challenger_name, challenger, variant_correct),
    ):
        odds = results[column]
        print(f"  {name:<18} {correct.sum():,}/{len(results):,} correct ({correct.mean():.2%})"
              f"   Brier {brier_score_loss(actual, odds):.5f}   Log loss {log_loss(actual, odds, labels=[0, 1]):.5f}")
    print(f"  {challenger_name} flips {(correct_delta > 0).sum():,} to correct and {(correct_delta < 0).sum():,} to incorrect")
    print(f"  Paired accuracy difference {correct_delta.mean() * 100:+.2f} pp"
          f"   95% date-bootstrap interval [{low:+.2f}, {high:+.2f}] pp")
    print(f"  Paired log loss difference {loss_delta.mean():+.5f}"
          f"   95% date-bootstrap interval [{loss_low:+.5f}, {loss_high:+.5f}]")


def challenger_features(features: pd.DataFrame, games: pd.DataFrame, challenger: str) -> pd.DataFrame:
    if challenger == "elo":
        elo, _, _ = build_elo_features(games)
        additions = elo[[*GAME_KEY, ELO_DIFF_COLUMN]]
    elif challenger == "recent-form":
        additions, _ = build_recent_form_features(games)
    elif challenger == "schedule-density":
        additions, _ = build_schedule_density_features(games)
    else:
        return features

    replaced_columns = additions.columns.difference(GAME_KEY)
    merged = features.drop(columns=replaced_columns, errors="ignore").merge(
        additions, on=GAME_KEY, validate="one_to_one",
    )
    if len(merged) != len(features):
        raise ValueError(f"The {challenger} features do not cover every saved game.")
    return merged.sort_values(GAME_KEY, ignore_index=True)


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
    features = features.sort_values(GAME_KEY, ignore_index=True)
    challengers = NEW_CHALLENGERS if challenger == "all-new" else (challenger,)
    baseline = {season: predict_holdout(features, season, _fit_random_forest) for season in holdouts}

    for name in challengers:
        variant_features = challenger_features(features, games, name)
        results = []
        for season in holdouts:
            fold = score_holdout(
                features, season, name,
                variant_features=variant_features, baseline_probabilities=baseline[season],
            )
            results.append(fold)
            train = games.loc[games["season"] < season]
            report(f"{season - 1}-{str(season)[-2:]} through {fold['date'].max():%Y-%m-%d}"
                   f" | trained on {len(train):,} games through {train['date'].max():%Y-%m-%d}", fold, name)
        report("Combined forward holdouts", pd.concat(results, ignore_index=True), name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=pd.Timestamp, help="Latest game date to include (YYYY-MM-DD).")
    parser.add_argument("--challenger", choices=[*CHALLENGERS, "all-new"], default="elo")
    args = parser.parse_args()
    evaluate(args.through, args.challenger)
