"""Score paired model forecasts saved before tipoff against completed games."""

import argparse

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from cleaner import clean_games, normalize_team_names
from daily_predictions import (
    DEFAULT_PREDICTION_MODEL,
    EXPERIMENT_VERSION_COLUMN,
    LOGISTIC_PROBABILITY_COLUMN,
    PLAYER_IMPACT_PROBABILITY_COLUMN,
    read_daily_predictions,
)
from game_updates import GAME_KEY, read_game_updates


BASELINE_PROBABILITY_COLUMN = "home_win_probability"
FORECAST_KEY = ["game_date", *GAME_KEY[1:]]
MODEL_COLUMNS = (
    ("Random forest baseline", BASELINE_PROBABILITY_COLUMN),
    ("Random forest + player impact", PLAYER_IMPACT_PROBABILITY_COLUMN),
    ("Logistic regression", LOGISTIC_PROBABILITY_COLUMN),
)


def paired_results(predictions: pd.DataFrame, finals: pd.DataFrame) -> pd.DataFrame:
    """Keep one saved comparison per completed game, using the last valid pregame row."""
    valid = (
        predictions["predicted_at_utc"].lt(predictions["tipoff_utc"])
        & predictions["model"].eq(DEFAULT_PREDICTION_MODEL)
        & predictions[EXPERIMENT_VERSION_COLUMN].notna()
        & predictions[BASELINE_PROBABILITY_COLUMN].between(0, 1)
        & predictions[PLAYER_IMPACT_PROBABILITY_COLUMN].between(0, 1)
    )
    picks = predictions.loc[valid].sort_values("predicted_at_utc", kind="stable")
    picks = picks.drop_duplicates(FORECAST_KEY, keep="last")
    if picks.empty or finals.empty:
        return picks.iloc[:0].assign(home_win=pd.Series(dtype=int))

    completed = normalize_team_names(finals).drop_duplicates(GAME_KEY, keep="last")
    completed = clean_games(completed)
    valid_scores = completed["home_pts"].gt(0) & completed["away_pts"].gt(0) & completed["margin"].gt(0)
    completed = completed.loc[valid_scores, [*GAME_KEY, "home_win"]].rename(columns={"date": "game_date"})
    return picks.merge(completed, on=FORECAST_KEY, validate="one_to_one")


def print_metrics(results: pd.DataFrame, models: tuple[tuple[str, str], ...]) -> None:
    actual = results["home_win"]
    for name, column in models:
        odds = results[column]
        correct = int((odds.gt(0.5) == actual).sum())
        brier = brier_score_loss(actual, odds)
        loss = log_loss(actual, odds, labels=[0, 1])
        print(f"  {name:<30} {correct:>4}/{len(results)} correct ({correct / len(results):.1%})"
              f"   Brier {brier:.5f}   Log loss {loss:.5f}")


def report(results: pd.DataFrame) -> None:
    if results.empty:
        print("No completed games with paired pregame predictions yet.")
        return

    for version, games in results.groupby(EXPERIMENT_VERSION_COLUMN):
        baseline = games[BASELINE_PROBABILITY_COLUMN]
        variant = games[PLAYER_IMPACT_PROBABILITY_COLUMN]
        odds_changed = int(variant.ne(baseline).sum())
        pick_changed = int(variant.gt(0.5).ne(baseline.gt(0.5)).sum())
        print(f"{version} | {len(games)} completed games | {odds_changed} changed odds | {pick_changed} changed picks")
        logistic = games.loc[games[LOGISTIC_PROBABILITY_COLUMN].between(0, 1)]
        if len(logistic) == len(games):
            print_metrics(games, MODEL_COLUMNS)
        else:
            print_metrics(games, MODEL_COLUMNS[:2])
            if not logistic.empty:
                print(f"  Matched logistic comparison on {len(logistic)} games:")
                print_metrics(logistic, MODEL_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, help="Season end year, such as 2027 for 2026–27.")
    parser.add_argument("--version", help="Experiment version to score.")
    args = parser.parse_args()

    results = paired_results(read_daily_predictions(), read_game_updates())
    if args.season is not None:
        results = results.loc[results["season"].eq(args.season)]
    if args.version is not None:
        results = results.loc[results[EXPERIMENT_VERSION_COLUMN].eq(args.version)]
    report(results)


if __name__ == "__main__":
    main()
