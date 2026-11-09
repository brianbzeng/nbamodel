"""Score paired model forecasts saved before tipoff against completed games."""

import argparse

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from cleaner import clean_games, normalize_team_names
from daily_predictions import (
    DEFAULT_PREDICTION_MODEL,
    ELO_EXPERIMENT_VERSION_COLUMN,
    ELO_PROBABILITY_COLUMN,
    EXPERIMENT_VERSION_COLUMN,
    LOGISTIC_PROBABILITY_COLUMN,
    PLAYER_IMPACT_PROBABILITY_COLUMN,
    read_daily_predictions,
)
from game_updates import GAME_KEY, read_game_updates


BASELINE_PROBABILITY_COLUMN = "home_win_probability"
FORECAST_KEY = ["game_date", *GAME_KEY[1:]]
BASELINE_MODEL = ("Random forest baseline", BASELINE_PROBABILITY_COLUMN)
PLAYER_IMPACT_MODEL = ("Random forest + player impact", PLAYER_IMPACT_PROBABILITY_COLUMN)
ELO_MODEL = ("Random forest + Elo", ELO_PROBABILITY_COLUMN)
LOGISTIC_MODEL = ("Logistic regression", LOGISTIC_PROBABILITY_COLUMN)
EXPERIMENTS = (
    (EXPERIMENT_VERSION_COLUMN, PLAYER_IMPACT_MODEL),
    (ELO_EXPERIMENT_VERSION_COLUMN, ELO_MODEL),
)


def paired_results(
    predictions: pd.DataFrame,
    finals: pd.DataFrame,
    *,
    version_column: str = EXPERIMENT_VERSION_COLUMN,
    probability_column: str = PLAYER_IMPACT_PROBABILITY_COLUMN,
) -> pd.DataFrame:
    """Keep one saved comparison per completed game, using the last valid pregame row."""
    if predictions.empty or finals.empty or not {version_column, probability_column}.issubset(predictions.columns):
        return predictions.iloc[:0].assign(home_win=pd.Series(dtype=int))

    valid = (
        predictions["predicted_at_utc"].lt(predictions["tipoff_utc"])
        & predictions["model"].eq(DEFAULT_PREDICTION_MODEL)
        & predictions[version_column].notna()
        & predictions[BASELINE_PROBABILITY_COLUMN].between(0, 1)
        & predictions[probability_column].between(0, 1)
    )
    picks = predictions.loc[valid].sort_values("predicted_at_utc", kind="stable")
    picks = picks.drop_duplicates(FORECAST_KEY, keep="last")
    if picks.empty:
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


def report(
    results: pd.DataFrame,
    *,
    version_column: str = EXPERIMENT_VERSION_COLUMN,
    challenger: tuple[str, str] = PLAYER_IMPACT_MODEL,
) -> None:
    if results.empty:
        print("No completed games with paired pregame predictions yet.")
        return

    models = (BASELINE_MODEL, challenger, LOGISTIC_MODEL)
    _, probability_column = challenger
    for version, games in results.groupby(version_column):
        baseline = games[BASELINE_PROBABILITY_COLUMN]
        variant = games[probability_column]
        odds_changed = int(variant.ne(baseline).sum())
        pick_changed = int(variant.gt(0.5).ne(baseline.gt(0.5)).sum())
        print(f"{version} | {len(games)} completed games | {odds_changed} changed odds | {pick_changed} changed picks")
        logistic = games.iloc[:0]
        if LOGISTIC_PROBABILITY_COLUMN in games:
            logistic = games.loc[games[LOGISTIC_PROBABILITY_COLUMN].between(0, 1)]
        if len(logistic) == len(games):
            print_metrics(games, models)
        else:
            print_metrics(games, models[:2])
            if not logistic.empty:
                print(f"  Matched logistic comparison on {len(logistic)} games:")
                print_metrics(logistic, models)


def report_experiments(
    predictions: pd.DataFrame,
    finals: pd.DataFrame,
    *,
    season: int | None = None,
    version: str | None = None,
) -> None:
    """Compare each challenger with the baseline on its own matched cohort."""
    if season is not None and not predictions.empty:
        predictions = predictions.loc[predictions["season"].eq(season)]

    found_results = False
    for version_column, challenger in EXPERIMENTS:
        _, probability_column = challenger
        results = paired_results(
            predictions, finals, version_column=version_column, probability_column=probability_column,
        )
        if results.empty:
            continue
        if version is not None:
            results = results.loc[results[version_column].eq(version)]
        if not results.empty:
            report(results, version_column=version_column, challenger=challenger)
            found_results = True

    if not found_results:
        print("No completed games with paired pregame predictions yet.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, help="Season end year, such as 2027 for 2026–27.")
    parser.add_argument("--version", help="Experiment version to score.")
    args = parser.parse_args()

    report_experiments(read_daily_predictions(), read_game_updates(), season=args.season, version=args.version)


if __name__ == "__main__":
    main()
