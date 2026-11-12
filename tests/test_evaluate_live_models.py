import pandas as pd

from daily_predictions import SHADOW_EXPERIMENTS
from evaluate_live_models import (
    ELO_EXPERIMENT_VERSION_COLUMN,
    ELO_PROBABILITY_COLUMN,
    EXPERIMENT_VERSION_COLUMN,
    LOGISTIC_PROBABILITY_COLUMN,
    NO_REST_EXPERIMENT_VERSION_COLUMN,
    NO_REST_PROBABILITY_COLUMN,
    PLAYER_IMPACT_PROBABILITY_COLUMN,
    paired_results,
    report,
    report_experiments,
)
from game_updates import GAME_COLUMNS


def forecast(game_id, game_date="2026-11-09", **changes):
    return {
        "game_id": game_id,
        "game_date": pd.Timestamp(game_date),
        "season": 2027,
        "home_team": "BOS",
        "away_team": "NYK",
        "tipoff_utc": pd.Timestamp(f"{game_date}T23:00Z"),
        "predicted_at_utc": pd.Timestamp(f"{game_date}T22:00Z"),
        "model": "random_forest",
        "home_win_probability": 0.45,
        EXPERIMENT_VERSION_COLUMN: "rf_player_impact_v1",
        PLAYER_IMPACT_PROBABILITY_COLUMN: 0.55,
        LOGISTIC_PROBABILITY_COLUMN: 0.61,
        **changes,
    }


def test_scorecard_matches_latest_valid_pregame_forecast_to_final_once():
    predictions = pd.DataFrame([
        forecast("old"),
        forecast("valid", predicted_at_utc=pd.Timestamp("2026-11-09T22:45Z")),
        forecast("at-tip", predicted_at_utc=pd.Timestamp("2026-11-09T23:00Z")),
        forecast("later", predicted_at_utc=pd.Timestamp("2026-11-09T23:01Z")),
        forecast("invalid-odds", predicted_at_utc=pd.Timestamp("2026-11-09T22:55Z"),
                 **{PLAYER_IMPACT_PROBABILITY_COLUMN: 1.01}),
        forecast("old-model", model="random_forest_player_impact"),
        forecast("no-version", **{EXPERIMENT_VERSION_COLUMN: None}),
        forecast("unplayed", "2026-11-10"),
        forecast("wrong-season", season=2026),
        forecast("wrong-side", home_team="NYK", away_team="BOS"),
        forecast("tied-score", "2026-11-11"),
    ])
    finals = pd.DataFrame([
        (pd.Timestamp("2026-11-09"), 2027, "Boston Celtics", "New York Knicks", 106, 109),
        (pd.Timestamp("2026-11-09"), 2027, "Boston Celtics", "New York Knicks", 110, 109),
        (pd.Timestamp("2026-11-11"), 2027, "Boston Celtics", "New York Knicks", 107, 107),
    ], columns=GAME_COLUMNS)

    results = paired_results(predictions, finals)

    assert results[["game_id", "home_win"]].values.tolist() == [["valid", 1]]


def test_scorecard_reports_matched_cohorts_by_version(capsys):
    results = pd.DataFrame({
        EXPERIMENT_VERSION_COLUMN: ["v1", "v1", "v1", "v2"],
        "home_win": [1, 0, 1, 0],
        "home_win_probability": [0.4, 0.3, 0.7, 0.5],
        PLAYER_IMPACT_PROBABILITY_COLUMN: [0.6, 0.7, 0.7, 0.5],
        LOGISTIC_PROBABILITY_COLUMN: [0.8, float("nan"), 0.9, 0.7],
    })

    report(results)

    output = capsys.readouterr().out
    assert "v1 | 3 completed games | 2 changed odds | 2 changed picks" in output
    assert "2/3 correct (66.7%)   Brier 0.18000   Log loss" in output
    assert "2/3 correct (66.7%)   Brier 0.24667   Log loss" in output
    assert "Matched logistic comparison on 2 games" in output
    assert "2/2 correct (100.0%)   Brier 0.02500   Log loss" in output
    assert "v2 | 1 completed games | 0 changed odds | 0 changed picks" in output
    assert "1/1 correct (100.0%)" in output


def test_scorecard_explains_empty_results(capsys):
    report(pd.DataFrame())

    assert capsys.readouterr().out == "No completed games with paired pregame predictions yet.\n"


def test_scorecard_scores_each_challenger_on_its_own_pregame_cohort(capsys):
    elo = {ELO_EXPERIMENT_VERSION_COLUMN: "rf_elo_v1", ELO_PROBABILITY_COLUMN: 0.4}
    no_rest = {NO_REST_EXPERIMENT_VERSION_COLUMN: "rf_no_rest_v1", NO_REST_PROBABILITY_COLUMN: 0.6}
    predictions = pd.DataFrame([
        forecast("player-impact"),
        forecast("elo", "2026-11-10", **elo, **{EXPERIMENT_VERSION_COLUMN: None}),
        forecast("all", "2026-11-11", **elo, **no_rest),
        forecast("late", "2026-11-12", **elo, **no_rest, predicted_at_utc=pd.Timestamp("2026-11-12T23:01Z")),
        forecast("no-rest", "2026-11-13", **no_rest, **{EXPERIMENT_VERSION_COLUMN: None}),
    ])
    finals = pd.DataFrame([
        (pd.Timestamp("2026-11-09"), 2027, "Boston Celtics", "New York Knicks", 110, 109),
        (pd.Timestamp("2026-11-10"), 2027, "Boston Celtics", "New York Knicks", 108, 109),
        (pd.Timestamp("2026-11-11"), 2027, "Boston Celtics", "New York Knicks", 110, 109),
        (pd.Timestamp("2026-11-12"), 2027, "Boston Celtics", "New York Knicks", 110, 109),
        (pd.Timestamp("2026-11-13"), 2027, "Boston Celtics", "New York Knicks", 110, 109),
    ], columns=GAME_COLUMNS)

    report_experiments(predictions, finals)
    output = capsys.readouterr().out
    impact, remaining = output.split("rf_elo_v1 | 2 completed games")
    elo_report, no_rest_report = remaining.split("rf_no_rest_v1 | 2 completed games")
    impact, elo_report, no_rest_report = (" ".join(section.split()) for section in (impact, elo_report, no_rest_report))
    assert "rf_player_impact_v1 | 2 completed games" in impact
    assert "Random forest baseline 0/2 correct" in impact
    assert "Random forest + player impact 2/2 correct" in impact
    assert "Random forest baseline 1/2 correct" in elo_report
    assert "Random forest + Elo 1/2 correct" in elo_report
    assert "Random forest baseline 0/2 correct" in no_rest_report
    assert "Random forest without rest 2/2 correct" in no_rest_report

    report_experiments(predictions, finals, season=2027, version="rf_elo_v1")
    output = capsys.readouterr().out
    assert "rf_elo_v1 | 2 completed games" in output
    assert "rf_player_impact_v1" not in output
    assert "rf_no_rest_v1" not in output

    report_experiments(predictions, finals, season=2027, version="rf_no_rest_v1")
    output = capsys.readouterr().out
    assert "rf_no_rest_v1 | 2 completed games" in output
    assert "rf_elo_v1" not in output

    report_experiments(predictions, finals, season=2026, version="rf_elo_v1")
    assert capsys.readouterr().out == "No completed games with paired pregame predictions yet.\n"


def test_scorecard_scores_new_challengers_on_separate_cohorts(capsys):
    variants = (
        ("margin", "rf_margin_v1"),
        ("blend", "rf_logistic_blend_v1"),
        ("weighted_form", "rf_weighted_form_v1"),
        ("schedule_density", "rf_schedule_density_v1"),
    )
    forecasts = []
    completed = []
    for day, (prefix, version) in enumerate(variants, start=14):
        game_date = f"2026-11-{day}"
        columns = {f"{prefix}_experiment_version": version, EXPERIMENT_VERSION_COLUMN: None}
        probability_column = f"{prefix}_home_win_probability"
        forecasts.extend([
            forecast(version, game_date, **columns, **{probability_column: 0.65}),
            forecast(f"late-{version}", game_date, **columns, **{probability_column: 0.25},
                     predicted_at_utc=pd.Timestamp(f"{game_date}T23:00Z")),
        ])
        completed.append((pd.Timestamp(game_date), 2027, "Boston Celtics", "New York Knicks", 110, 109))

    predictions = pd.DataFrame(forecasts)
    finals = pd.DataFrame(completed, columns=GAME_COLUMNS)

    report_experiments(predictions, finals)
    output = capsys.readouterr().out
    for _, version in variants:
        assert f"{version} | 1 completed games | 1 changed odds | 1 changed picks" in output
    assert "rf_player_impact_v1" not in output

    for _, version in variants:
        report_experiments(predictions, finals, version=version)
        output = " ".join(capsys.readouterr().out.split())
        label = next(experiment.label for experiment in SHADOW_EXPERIMENTS if experiment.version == version)
        assert f"{version} | 1 completed games" in output
        assert "Random forest baseline 0/1 correct" in output
        assert f"{label} 1/1 correct" in output
        assert all(other not in output for _, other in variants if other != version)


def test_scorecard_compares_hybrid_parents_on_the_same_pregame_games(capsys):
    hybrid = {
        "impact_blend_experiment_version": "rf_player_impact_logistic_blend_v1",
        "impact_blend_home_win_probability": 0.58,
    }
    blend = {
        "blend_experiment_version": "rf_logistic_blend_v1",
        "blend_home_win_probability": 0.53,
    }
    predictions = pd.DataFrame([
        forecast("before-hybrid", "2026-11-09", **blend),
        forecast("matched", "2026-11-10", **hybrid, **blend),
        forecast("missing-blend", "2026-11-11", **hybrid),
        forecast("late", "2026-11-12", **hybrid, **blend, predicted_at_utc=pd.Timestamp("2026-11-12T23:00Z")),
    ])
    finals = pd.DataFrame([
        (pd.Timestamp(f"2026-11-{day:02}"), 2027, "Boston Celtics", "New York Knicks", 110, 109)
        for day in range(9, 13)
    ], columns=GAME_COLUMNS)

    report_experiments(predictions, finals, version=hybrid["impact_blend_experiment_version"])

    output = " ".join(capsys.readouterr().out.split())
    full, partial = output.split("Matched Random forest + logistic blend comparison on 1 game:")
    assert "rf_player_impact_logistic_blend_v1 | 2 completed games" in full
    assert "Random forest baseline 0/2 correct" in full
    assert "Player impact + logistic blend 2/2 correct" in full
    assert "Random forest + player impact 2/2 correct" in full
    assert "Logistic regression 2/2 correct" in full
    assert "Random forest + logistic blend" not in full
    assert "Random forest + logistic blend 1/1 correct" in partial
    assert "Player impact + logistic blend 1/1 correct" in partial
