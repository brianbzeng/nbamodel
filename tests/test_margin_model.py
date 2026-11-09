import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer

from margin_model import (
    MARGIN_CALIBRATION_INTERCEPT,
    MARGIN_CALIBRATION_SLOPE,
    SIGNED_MARGIN_COLUMN,
    fit_margin_forest,
)


def test_margin_forest_fits_signed_scores_and_returns_calibrated_win_probabilities():
    training = pd.DataFrame({
        "signal": [-1] * 16 + [1] * 16,
        SIGNED_MARGIN_COLUMN: [-20] * 16 + [10] * 16,
    })
    preprocessor = ColumnTransformer([("signal", "passthrough", ["signal"])])
    forest = fit_margin_forest(training, preprocessor=preprocessor, feature_columns=["signal"], n_jobs=1)
    features = pd.DataFrame({"signal": [-1, 1]})

    assert forest.feature_names_in_.tolist() == ["signal"]
    margins = forest.regression.predict(features)
    np.testing.assert_allclose(margins, [-20, 10])

    probabilities = forest.predict_proba(features)
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(
        probabilities[:, 1], expit(MARGIN_CALIBRATION_INTERCEPT + MARGIN_CALIBRATION_SLOPE * margins),
    )
    assert probabilities[0, 1] < 0.5 < probabilities[1, 1]
