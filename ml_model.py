from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_DIR = Path("data/raw")


def prompt_for_game_csv() -> Path:
    print("\n=== NBA Model Trainer ===")
    print("Available game CSVs in data/raw:")

    game_files = sorted(RAW_DIR.glob("bref_games*.csv"))
    for path in game_files:
        print(f"- {path.name}")

    default_name = game_files[0].name if game_files else "bref_games_2016_2025.csv"
    chosen = input(f"Game CSV filename [{default_name}]: ").strip() or default_name
    return RAW_DIR / chosen


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop = True)
    return df


def load_team_injuries(filepath: str) -> pd.DataFrame:
    injury_df = pd.read_csv(filepath)
    injury_df["game_date"] = pd.to_datetime(injury_df["game_date"]).dt.normalize()
    injury_df["latest_report_datetime"] = pd.to_datetime(injury_df["latest_report_datetime"])
    return injury_df


def get_default_injury_path(df: pd.DataFrame) -> Path:
    start_season = int(df["season"].min())
    end_season = int(df["season"].max())
    return RAW_DIR / f"official_nba_injuries_by_team_{start_season}_{end_season}.csv"


def prompt_for_season_split(df: pd.DataFrame) -> tuple[int, int, int]:
    available_seasons = sorted(df["season"].unique())
    min_season = int(min(available_seasons))
    max_season = int(max(available_seasons))

    if len(available_seasons) < 2:
        raise ValueError("Need at least two seasons of data for a season-based train/test split.")

    default_test_season = max_season
    default_train_start = min_season
    default_train_end = max_season - 1

    print("\n=== Season Split ===")
    print(f"Available seasons: {', '.join(str(season) for season in available_seasons)}")

    train_start = int(
        input(f"Choose training starting point [{default_train_start}]: ").strip()
        or default_train_start
    )
    train_end = int(
        input(f"Choose training ending point [{default_train_end}]: ").strip()
        or default_train_end
    )
    test_season = int(
        input(f"Choose testing season [{default_test_season}]: ").strip() or default_test_season
    )

    if train_start > train_end:
        raise ValueError("Training start season must be less than or equal to training end season.")
    if test_season <= train_end:
        raise ValueError("Test season must be after the training end season.")

    requested = set(range(train_start, train_end + 1)) | {test_season}
    missing = sorted(requested.difference(set(int(season) for season in available_seasons)))
    if missing:
        raise ValueError(f"Selected seasons are not all present in the game data: {missing}")

    return train_start, train_end, test_season


def _team_state():
    return {
        "games": 0,
        "wins": 0,
        "home_games": 0,
        "home_wins": 0,
        "away_games": 0,
        "away_wins": 0,
        "pts_for": 0,
        "pts_against": 0,
        "margin_sum": 0,
        "recent_wins": deque(maxlen = 5),
        "recent_margins": deque(maxlen = 5),
        "last_date": None,
    }


def _safe_pct(wins: int, games: int, default: float = 0.5) -> float:
    return wins / games if games else default


def _safe_avg(total: float, count: int, default: float = 0.0) -> float:
    return total / count if count else default


def _recent_avg(values: deque, default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _team_strength_score(win_pct: float, avg_margin: float) -> float:
    normalized_margin = max(min(avg_margin / 15.0, 1.0), -1.0)
    margin_component = (normalized_margin + 1.0) / 2.0
    return win_pct * 0.7 + margin_component * 0.3


def build_pregame_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop = True).copy()
    season_states = defaultdict(lambda: defaultdict(_team_state))
    feature_rows = []

    for row in df.itertuples(index = False):
        season = row.season
        home_team = row.home_team
        away_team = row.away_team
        game_date = row.date

        home_state = season_states[season][home_team]
        away_state = season_states[season][away_team]

        home_rest_days = (
            (game_date - home_state["last_date"]).days - 1
            if home_state["last_date"] is not None
            else 7
        )
        away_rest_days = (
            (game_date - away_state["last_date"]).days - 1
            if away_state["last_date"] is not None
            else 7
        )

        home_win_pct = _safe_pct(home_state["wins"], home_state["games"])
        away_win_pct = _safe_pct(away_state["wins"], away_state["games"])
        home_home_win_pct = _safe_pct(home_state["home_wins"], home_state["home_games"])
        away_away_win_pct = _safe_pct(away_state["away_wins"], away_state["away_games"])

        home_avg_pts_for = _safe_avg(home_state["pts_for"], home_state["games"], default = 110.0)
        away_avg_pts_for = _safe_avg(away_state["pts_for"], away_state["games"], default = 110.0)
        home_avg_pts_against = _safe_avg(
            home_state["pts_against"], home_state["games"], default = 110.0
        )
        away_avg_pts_against = _safe_avg(
            away_state["pts_against"], away_state["games"], default = 110.0
        )
        home_avg_margin = _safe_avg(home_state["margin_sum"], home_state["games"])
        away_avg_margin = _safe_avg(away_state["margin_sum"], away_state["games"])
        home_recent_win_pct = _recent_avg(home_state["recent_wins"], default = 0.5)
        away_recent_win_pct = _recent_avg(away_state["recent_wins"], default = 0.5)
        home_recent_margin = _recent_avg(home_state["recent_margins"], default = 0.0)
        away_recent_margin = _recent_avg(away_state["recent_margins"], default = 0.0)
        home_team_strength = _team_strength_score(home_win_pct, home_avg_margin)
        away_team_strength = _team_strength_score(away_win_pct, away_avg_margin)

        feature_rows.append(
            {
                "date": game_date,
                "season": season,
                "home_team": home_team,
                "away_team": away_team,
                "home_games_played": home_state["games"],
                "away_games_played": away_state["games"],
                "home_win_pct": home_win_pct,
                "away_win_pct": away_win_pct,
                "win_pct_diff": home_win_pct - away_win_pct,
                "home_home_win_pct": home_home_win_pct,
                "away_away_win_pct": away_away_win_pct,
                "venue_win_pct_diff": home_home_win_pct - away_away_win_pct,
                "home_avg_pts_for": home_avg_pts_for,
                "away_avg_pts_for": away_avg_pts_for,
                "offense_diff": home_avg_pts_for - away_avg_pts_for,
                "home_avg_pts_against": home_avg_pts_against,
                "away_avg_pts_against": away_avg_pts_against,
                "defense_diff": away_avg_pts_against - home_avg_pts_against,
                "home_avg_margin": home_avg_margin,
                "away_avg_margin": away_avg_margin,
                "margin_diff": home_avg_margin - away_avg_margin,
                "home_team_strength": home_team_strength,
                "away_team_strength": away_team_strength,
                "team_strength_diff": home_team_strength - away_team_strength,
                "home_recent_win_pct": home_recent_win_pct,
                "away_recent_win_pct": away_recent_win_pct,
                "recent_win_pct_diff": home_recent_win_pct - away_recent_win_pct,
                "home_recent_margin": home_recent_margin,
                "away_recent_margin": away_recent_margin,
                "recent_margin_diff": home_recent_margin - away_recent_margin,
                "home_rest_days": max(home_rest_days, 0),
                "away_rest_days": max(away_rest_days, 0),
                "rest_diff": max(home_rest_days, 0) - max(away_rest_days, 0),
                "home_win": row.home_win,
            }
        )

        home_margin = row.home_pts - row.away_pts
        away_margin = -home_margin
        home_win = int(row.home_win)
        away_win = 1 - home_win

        home_state["games"] += 1
        home_state["wins"] += home_win
        home_state["home_games"] += 1
        home_state["home_wins"] += home_win
        home_state["pts_for"] += row.home_pts
        home_state["pts_against"] += row.away_pts
        home_state["margin_sum"] += home_margin
        home_state["recent_wins"].append(home_win)
        home_state["recent_margins"].append(home_margin)
        home_state["last_date"] = game_date

        away_state["games"] += 1
        away_state["wins"] += away_win
        away_state["away_games"] += 1
        away_state["away_wins"] += away_win
        away_state["pts_for"] += row.away_pts
        away_state["pts_against"] += row.home_pts
        away_state["margin_sum"] += away_margin
        away_state["recent_wins"].append(away_win)
        away_state["recent_margins"].append(away_margin)
        away_state["last_date"] = game_date

    return pd.DataFrame(feature_rows)


def merge_injury_features(feature_df: pd.DataFrame, injury_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if injury_df is None or injury_df.empty:
        result = feature_df.copy()
        default_columns = {
            "home_out_count": 0.0,
            "away_out_count": 0.0,
            "home_doubtful_count": 0.0,
            "away_doubtful_count": 0.0,
            "home_questionable_count": 0.0,
            "away_questionable_count": 0.0,
            "home_probable_count": 0.0,
            "away_probable_count": 0.0,
            "home_weighted_injury_score": 0.0,
            "away_weighted_injury_score": 0.0,
            "home_reported_player_count": 0.0,
            "away_reported_player_count": 0.0,
            "home_estimated_absence_days_total": 0.0,
            "away_estimated_absence_days_total": 0.0,
            "home_estimated_absence_days_max": 0.0,
            "away_estimated_absence_days_max": 0.0,
            "home_injury_impact_score": 0.0,
            "away_injury_impact_score": 0.0,
            "home_long_term_absence_count": 0.0,
            "away_long_term_absence_count": 0.0,
            "home_injury_report_available": 0,
            "away_injury_report_available": 0,
            "out_count_diff": 0.0,
            "doubtful_count_diff": 0.0,
            "questionable_count_diff": 0.0,
            "probable_count_diff": 0.0,
            "weighted_injury_diff": 0.0,
            "estimated_absence_days_diff": 0.0,
            "max_absence_days_diff": 0.0,
            "injury_impact_diff": 0.0,
            "long_term_absence_diff": 0.0,
        }
        for column, value in default_columns.items():
            result[column] = value
        return result

    home_injuries = injury_df.rename(
        columns = {
            "game_date": "date",
            "team": "home_team",
            "latest_report_datetime": "home_latest_report_datetime",
            "out_count": "home_out_count",
            "doubtful_count": "home_doubtful_count",
            "questionable_count": "home_questionable_count",
            "probable_count": "home_probable_count",
            "available_count": "home_available_count",
            "weighted_injury_score": "home_weighted_injury_score",
            "reported_player_count": "home_reported_player_count",
            "estimated_absence_days_total": "home_estimated_absence_days_total",
            "estimated_absence_days_max": "home_estimated_absence_days_max",
            "injury_impact_score": "home_injury_impact_score",
            "long_term_absence_count": "home_long_term_absence_count",
        }
    )
    away_injuries = injury_df.rename(
        columns = {
            "game_date": "date",
            "team": "away_team",
            "latest_report_datetime": "away_latest_report_datetime",
            "out_count": "away_out_count",
            "doubtful_count": "away_doubtful_count",
            "questionable_count": "away_questionable_count",
            "probable_count": "away_probable_count",
            "available_count": "away_available_count",
            "weighted_injury_score": "away_weighted_injury_score",
            "reported_player_count": "away_reported_player_count",
            "estimated_absence_days_total": "away_estimated_absence_days_total",
            "estimated_absence_days_max": "away_estimated_absence_days_max",
            "injury_impact_score": "away_injury_impact_score",
            "long_term_absence_count": "away_long_term_absence_count",
        }
    )

    result = feature_df.merge(home_injuries, on = ["date", "home_team"], how = "left")
    result = result.merge(away_injuries, on = [ "date", "away_team"], how = "left")

    fill_zero_columns = [
        "home_out_count",
        "away_out_count",
        "home_doubtful_count",
        "away_doubtful_count",
        "home_questionable_count",
        "away_questionable_count",
        "home_probable_count",
        "away_probable_count",
        "home_available_count",
        "away_available_count",
        "home_weighted_injury_score",
        "away_weighted_injury_score",
        "home_reported_player_count",
        "away_reported_player_count",
        "home_estimated_absence_days_total",
        "away_estimated_absence_days_total",
        "home_estimated_absence_days_max",
        "away_estimated_absence_days_max",
        "home_injury_impact_score",
        "away_injury_impact_score",
        "home_long_term_absence_count",
        "away_long_term_absence_count",
    ]
    for column in fill_zero_columns:
        result[column] = result[column].fillna(0.0)

    result["home_injury_report_available"] = result["home_latest_report_datetime"].notna().astype(int)
    result["away_injury_report_available"] = result["away_latest_report_datetime"].notna().astype(int)
    result["out_count_diff"] = result["away_out_count"] - result["home_out_count"]
    result["doubtful_count_diff"] = result["away_doubtful_count"] - result["home_doubtful_count"]
    result["questionable_count_diff"] = (
        result["away_questionable_count"] - result["home_questionable_count"]
    )
    result["probable_count_diff"] = result["away_probable_count"] - result["home_probable_count"]
    result["weighted_injury_diff"] = (
        result["away_weighted_injury_score"] - result["home_weighted_injury_score"]
    )
    result["estimated_absence_days_diff"] = (
        result["away_estimated_absence_days_total"] - result["home_estimated_absence_days_total"]
    )
    result["max_absence_days_diff"] = (
        result["away_estimated_absence_days_max"] - result["home_estimated_absence_days_max"]
    )
    result["injury_impact_diff"] = (
        result["away_injury_impact_score"] - result["home_injury_impact_score"]
    )
    result["long_term_absence_diff"] = (
        result["away_long_term_absence_count"] - result["home_long_term_absence_count"]
    )

    return result


def summarize_injury_coverage(feature_df: pd.DataFrame) -> None:
    total_games = len(feature_df)
    home_matches = int(feature_df["home_injury_report_available"].sum())
    away_matches = int(feature_df["away_injury_report_available"].sum())
    both_matches = int(
        (
            (feature_df["home_injury_report_available"] == 1)
            & (feature_df["away_injury_report_available"] == 1)
        ).sum()
    )

    print("\n=== Injury Feature Coverage ===")
    print(f"Games in dataset: {total_games:,}")
    print(f"Home-team injury matches: {home_matches:,}")
    print(f"Away-team injury matches: {away_matches:,}")
    print(f"Games with both teams matched: {both_matches:,}")
    if both_matches == 0:
        print("No overlapping injury data was matched to this game dataset.")


def build_model(
    df: pd.DataFrame,
    injury_df: Optional[pd.DataFrame] = None,
    train_start_season: Optional[int] = None,
    train_end_season: Optional[int] = None,
    test_season: Optional[int] = None,
):
    feature_df = build_pregame_features(df)
    feature_df = merge_injury_features(feature_df, injury_df)
    summarize_injury_coverage(feature_df)

    target_var = "home_win"
    categorical_features = ["home_team", "away_team"]
    numeric_features = [
        "season",
        "home_games_played",
        "away_games_played",
        "home_win_pct",
        "away_win_pct",
        "win_pct_diff",
        "home_home_win_pct",
        "away_away_win_pct",
        "venue_win_pct_diff",
        "home_avg_pts_for",
        "away_avg_pts_for",
        "offense_diff",
        "home_avg_pts_against",
        "away_avg_pts_against",
        "defense_diff",
        "home_avg_margin",
        "away_avg_margin",
        "margin_diff",
        "home_team_strength",
        "away_team_strength",
        "team_strength_diff",
        "home_recent_win_pct",
        "away_recent_win_pct",
        "recent_win_pct_diff",
        "home_recent_margin",
        "away_recent_margin",
        "recent_margin_diff",
        "home_rest_days",
        "away_rest_days",
        "rest_diff",
        "home_out_count",
        "away_out_count",
        "home_doubtful_count",
        "away_doubtful_count",
        "home_questionable_count",
        "away_questionable_count",
        "home_probable_count",
        "away_probable_count",
        "home_weighted_injury_score",
        "away_weighted_injury_score",
        "home_reported_player_count",
        "away_reported_player_count",
        "home_estimated_absence_days_total",
        "away_estimated_absence_days_total",
        "home_estimated_absence_days_max",
        "away_estimated_absence_days_max",
        "home_injury_impact_score",
        "away_injury_impact_score",
        "home_long_term_absence_count",
        "away_long_term_absence_count",
        "home_injury_report_available",
        "away_injury_report_available",
        "out_count_diff",
        "doubtful_count_diff",
        "questionable_count_diff",
        "probable_count_diff",
        "weighted_injury_diff",
        "estimated_absence_days_diff",
        "max_absence_days_diff",
        "injury_impact_diff",
        "long_term_absence_diff",
    ]

    if train_start_season is None or train_end_season is None or test_season is None:
        raise ValueError("Season split parameters are required for training the model.")

    train_mask = (
        (feature_df["season"] >= train_start_season) & (feature_df["season"] <= train_end_season)
    )
    test_mask = feature_df["season"] == test_season

    train_df = feature_df.loc[train_mask].copy()
    test_df = feature_df.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Selected season split produced an empty training or test set.")

    X_train = train_df[categorical_features + numeric_features]
    y_train = train_df[target_var]
    X_test = test_df[categorical_features + numeric_features]
    y_test = test_df[target_var]

    categorical_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown = "ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    logistic_regression = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter = 2000, random_state = 100)),
        ]
    )

    random_forest = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators = 300,
                    min_samples_leaf = 3,
                    random_state = 100,
                    n_jobs = -1,
                ),
            ),
        ]
    )

    baseline_prediction = pd.Series(1, index = y_test.index)
    baseline_accuracy = accuracy_score(y_test, baseline_prediction)

    print("\n=== Train/Test Split ===")
    print(f"Training seasons: {train_start_season}-{train_end_season}")
    print(f"Test season: {test_season}")
    print(f"Training games: {len(train_df):,}")
    print(f"Test games: {len(test_df):,}")
    logistic_regression.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    lr_y_pred = logistic_regression.predict(X_test)
    rf_y_pred = random_forest.predict(X_test)

    lr_accuracy = accuracy_score(y_test, lr_y_pred)
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    lr_classification_rep = classification_report(y_test, lr_y_pred)
    rf_classification_rep = classification_report(y_test, rf_y_pred)

    print("\n=== Model Evaluation ===\n")
    print(f"Home-team baseline accuracy: {baseline_accuracy:.4f}")
    print(f"Logistic Regression accuracy: {lr_accuracy:.4f}")
    print(f"Random Forest accuracy: {rf_accuracy:.4f}\n")
    print("Logistic Regression classification report:")
    print(lr_classification_rep)
    print("Random Forest classification report:")
    print(rf_classification_rep)

    return logistic_regression, random_forest, feature_df


if __name__ == "__main__":
    try:
        game_csv_path = prompt_for_game_csv()
        nba_df = load_data(str(game_csv_path))
        print(f"Loaded game data from {game_csv_path}")
        train_start_season, train_end_season, test_season = prompt_for_season_split(nba_df)

        selected_df = nba_df[
            (nba_df["season"] >= train_start_season) & (nba_df["season"] <= test_season)
        ].copy()

        injury_range_path = RAW_DIR / (
            f"official_nba_injuries_by_team_{train_start_season}_{test_season}.csv"
        )
        injury_path = injury_range_path if injury_range_path.exists() else get_default_injury_path(selected_df)
        injury_df = load_team_injuries(injury_path) if injury_path.exists() else None
        if injury_df is None:
            print("No matching team-level injury CSV found for this game file.")
            print(
                "Expected injury CSV: "
                f"data/raw/official_nba_injuries_by_team_{train_start_season}_{test_season}.csv"
            )
            print("Running without injury features.")
        else:
            print(f"Loaded {len(injury_df):,} team-level injury rows from {injury_path}")
            print(
                "Injury report game-date range: "
                f"{injury_df['game_date'].min().date()} to {injury_df['game_date'].max().date()}"
            )

        logistic_regression, random_forest, feature_df = build_model(
            selected_df,
            injury_df = injury_df,
            train_start_season = train_start_season,
            train_end_season = train_end_season,
            test_season = test_season,
        )

        future_game = pd.DataFrame(
            {
                "home_team": ["Washington Wizards"],
                "away_team": ["Boston Celtics"],
                "season": [test_season],
                "home_games_played": [30],
                "away_games_played": [30],
                "home_win_pct": [0.300],
                "away_win_pct": [0.800],
                "win_pct_diff": [-0.500],
                "home_home_win_pct": [0.400],
                "away_away_win_pct": [0.700],
                "venue_win_pct_diff": [-0.300],
                "home_avg_pts_for": [108.0],
                "away_avg_pts_for": [118.0],
                "offense_diff": [-10.0],
                "home_avg_pts_against": [117.0],
                "away_avg_pts_against": [108.0],
                "defense_diff": [-9.0],
                "home_avg_margin": [-9.0],
                "away_avg_margin": [10.0],
                "margin_diff": [-19.0],
                "home_team_strength": [0.255],
                "away_team_strength": [0.817],
                "team_strength_diff": [-0.562],
                "home_recent_win_pct": [0.200],
                "away_recent_win_pct": [0.800],
                "recent_win_pct_diff": [-0.600],
                "home_recent_margin": [-11.0],
                "away_recent_margin": [8.0],
                "recent_margin_diff": [-19.0],
                "home_rest_days": [1],
                "away_rest_days": [1],
                "rest_diff": [0],
                "home_out_count": [2],
                "away_out_count": [0],
                "home_doubtful_count": [1],
                "away_doubtful_count": [0],
                "home_questionable_count": [1],
                "away_questionable_count": [0],
                "home_probable_count": [0],
                "away_probable_count": [1],
                "home_weighted_injury_score": [3.25],
                "away_weighted_injury_score": [0.25],
                "home_reported_player_count": [4],
                "away_reported_player_count": [1],
                "home_estimated_absence_days_total": [39.0],
                "away_estimated_absence_days_total": [3.0],
                "home_estimated_absence_days_max": [14.0],
                "away_estimated_absence_days_max": [3.0],
                "home_injury_impact_score": [29.25],
                "away_injury_impact_score": [0.45],
                "home_long_term_absence_count": [2],
                "away_long_term_absence_count": [0],
                "home_injury_report_available": [1],
                "away_injury_report_available": [1],
                "out_count_diff": [-2],
                "doubtful_count_diff": [-1],
                "questionable_count_diff": [-1],
                "probable_count_diff": [1],
                "weighted_injury_diff": [-3.0],
                "estimated_absence_days_diff": [-36.0],
                "max_absence_days_diff": [-11.0],
                "injury_impact_diff": [-28.8],
                "long_term_absence_diff": [-2.0],
            }
        )

        lr_prediction = logistic_regression.predict(future_game)
        lr_win_prob = logistic_regression.predict_proba(future_game)[0][1]

        rf_prediction = random_forest.predict(future_game)
        rf_win_prob = random_forest.predict_proba(future_game)[0][1]

        print("\n=== Logistic Regression Prediction ===")
        print(f"Matchup: {future_game['away_team'][0]} @ {future_game['home_team'][0]}")
        print(f"Home Win Predicted: {bool(lr_prediction[0])}")
        print(f"Home Win Probability: {lr_win_prob:.1%}")

        print("\n=== Random Forest Prediction ===")
        print(f"Matchup: {future_game['away_team'][0]} @ {future_game['home_team'][0]}")
        print(f"Home Win Predicted: {bool(rf_prediction[0])}")
        print(f"Home Win Probability: {rf_win_prob:.1%}")

    except FileNotFoundError:
        print("Please run the web scraper first to generate a CSV file.")
    except ValueError as exc:
        print(f"Configuration error: {exc}")
