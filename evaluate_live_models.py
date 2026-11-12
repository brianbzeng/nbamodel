"""Score paired model forecasts saved before tipoff against completed games."""

import argparse

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from cleaner import clean_games, normalize_team_names
from daily_predictions import (
    BASELINE_PROBABILITY_COLUMN,
    BLEND_KEY,
    DEFAULT_PREDICTION_MODEL,
    ELO_EXPERIMENT_VERSION_COLUMN,
    ELO_PROBABILITY_COLUMN,
    EXPERIMENT_VERSION_COLUMN,
    IMPACT_BLEND_HIGH_KEY,
    IMPACT_BLEND_KEY,
    IMPACT_BLEND_LOW_KEY,
    LOGISTIC_PROBABILITY_COLUMN,
    NO_REST_EXPERIMENT_VERSION_COLUMN,
    NO_REST_PROBABILITY_COLUMN,
    PLAYER_IMPACT_PROBABILITY_COLUMN,
    SHADOW_EXPERIMENTS,
    read_daily_predictions,
)
from game_updates import GAME_KEY, read_game_updates
from player_impact import PLAYER_IMPACT_MODEL as OFFICIAL_PLAYER_IMPACT_MODEL


FORECAST_KEY = ["game_date", *GAME_KEY[1:]]
BASELINE_MODEL = ("Random forest baseline", BASELINE_PROBABILITY_COLUMN)
PLAYER_IMPACT_MODEL = ("Random forest + player impact", PLAYER_IMPACT_PROBABILITY_COLUMN)
LOGISTIC_MODEL = ("Logistic regression", LOGISTIC_PROBABILITY_COLUMN)
FOREST_LOGISTIC_BLEND = next(
    (experiment.label, experiment.probability_column)
    for experiment in SHADOW_EXPERIMENTS
    if experiment.key == BLEND_KEY
)
HYBRID_EXPERIMENTS = (IMPACT_BLEND_KEY, IMPACT_BLEND_LOW_KEY, IMPACT_BLEND_HIGH_KEY)
HYBRID_COMPARATORS = (PLAYER_IMPACT_MODEL, LOGISTIC_MODEL, FOREST_LOGISTIC_BLEND)


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

    legacy_baseline = predictions["home_win_probability"].where(predictions["model"].eq(DEFAULT_PREDICTION_MODEL))
    baseline = predictions.get(BASELINE_PROBABILITY_COLUMN, legacy_baseline).fillna(legacy_baseline)
    valid = (
        predictions["predicted_at_utc"].lt(predictions["tipoff_utc"])
        & predictions["model"].isin((DEFAULT_PREDICTION_MODEL, OFFICIAL_PLAYER_IMPACT_MODEL))
        & predictions[version_column].notna()
        & baseline.between(0, 1)
        & predictions[probability_column].between(0, 1)
    )
    picks = predictions.loc[valid].assign(**{BASELINE_PROBABILITY_COLUMN: baseline.loc[valid]})
    picks = picks.sort_values("predicted_at_utc", kind="stable")
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
    comparators: tuple[tuple[str, str], ...] = (LOGISTIC_MODEL,),
) -> None:
    if results.empty:
        print("No completed games with paired pregame predictions yet.")
        return

    baseline_model = BASELINE_MODEL
    if BASELINE_PROBABILITY_COLUMN not in results:
        baseline_model = (BASELINE_MODEL[0], "home_win_probability")
    _, probability_column = challenger
    for version, games in results.groupby(version_column):
        baseline = games[baseline_model[1]]
        variant = games[probability_column]
        odds_changed = int(variant.ne(baseline).sum())
        pick_changed = int(variant.gt(0.5).ne(baseline.gt(0.5)).sum())
        print(f"{version} | {len(games)} completed games | {odds_changed} changed odds | {pick_changed} changed picks")
        models = (baseline_model, challenger)
        partial = []
        for comparator in comparators:
            _, column = comparator
            if column not in games:
                continue
            matched = games.loc[games[column].between(0, 1)]
            if len(matched) == len(games):
                models += (comparator,)
            elif not matched.empty:
                partial.append((comparator, matched))

        print_metrics(games, models)
        for comparator, matched in partial:
            name = "logistic" if comparator == LOGISTIC_MODEL else comparator[0]
            games_label = "game" if len(matched) == 1 else "games"
            print(f"  Matched {name} comparison on {len(matched)} {games_label}:")
            print_metrics(matched, (*models, comparator))


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
    for experiment in SHADOW_EXPERIMENTS:
        results = paired_results(
            predictions, finals, version_column=experiment.version_column,
            probability_column=experiment.probability_column,
        )
        if results.empty:
            continue
        if version is not None:
            results = results.loc[results[experiment.version_column].eq(version)]
        if not results.empty:
            report(
                results, version_column=experiment.version_column,
                challenger=(experiment.label, experiment.probability_column),
                comparators=HYBRID_COMPARATORS if experiment.key in HYBRID_EXPERIMENTS else (LOGISTIC_MODEL,),
            )
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
