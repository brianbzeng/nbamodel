"""Forecast signed score margin and map it to pregame win probability."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


SIGNED_MARGIN_COLUMN = "home_score_margin"
# Fit on 2024–25 forecasts from the 2021–24 forest, filtering calibration
# injury reports against archived ESPN tipoffs before building features.
MARGIN_CALIBRATION_INTERCEPT = -0.121020038601
MARGIN_CALIBRATION_SLOPE = 0.132340423644


@dataclass(frozen=True)
class MarginForest:
    regression: Pipeline

    @property
    def feature_names_in_(self) -> np.ndarray:
        return self.regression.feature_names_in_

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        predicted_margin = self.regression.predict(features)
        home_probability = expit(MARGIN_CALIBRATION_INTERCEPT + MARGIN_CALIBRATION_SLOPE * predicted_margin)
        return np.column_stack((1.0 - home_probability, home_probability))


def fit_margin_forest(
    feature_df: pd.DataFrame,
    *,
    preprocessor: ColumnTransformer,
    feature_columns: list[str],
    n_jobs: int,
) -> MarginForest:
    """Fit the forest on the home score minus the away score."""
    regression = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt", random_state=100, n_jobs=n_jobs,
        )),
    ])
    regression.fit(feature_df[feature_columns], feature_df[SIGNED_MARGIN_COLUMN])
    return MarginForest(regression)
