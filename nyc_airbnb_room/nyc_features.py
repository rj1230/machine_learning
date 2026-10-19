"""
Reusable, picklable components for the NYC Airbnb room-type model.

These classes live in their own module (not inside train_model.py) so the
saved model can be loaded from any script, e.g. a Streamlit app:

    import sys; sys.path.append("NYC_Airbnb_Room")
    import joblib
    model = joblib.load("NYC_Airbnb_Room/models/Best_Model_Pipeline.pkl")
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.metrics import f1_score


BASE_NUMERICAL_COLUMNS = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

ENGINEERED_NUMERICAL_COLUMNS = [
    "log_price",
    "price_to_neighbourhood_median",
    "price_to_group_median",
    "has_reviews",
    "long_stay_minimum",
    "zero_availability",
    "log_host_listings",
    "occupancy_proxy",
    "neighbourhood_listing_count",
]

CATEGORICAL_COLUMNS = ["neighbourhood_group", "neighbourhood"]

NUMERICAL_COLUMNS = BASE_NUMERICAL_COLUMNS + ENGINEERED_NUMERICAL_COLUMNS


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leakage-free outlier capping + feature engineering.

    Everything that depends on data (cap values, neighbourhood median prices,
    neighbourhood counts) is learned in `.fit()` on training folds only.
    """

    def __init__(
        self,
        cap_quantile: float = 0.99,
        cols_to_cap: tuple[str, ...] = ("price", "minimum_nights"),
    ):
        self.cap_quantile = cap_quantile
        self.cols_to_cap = cols_to_cap

    def _cap(self, X: pd.DataFrame) -> pd.DataFrame:
        for column, upper_bound in self.upper_bounds_.items():
            if column in X.columns:
                X[column] = X[column].clip(upper=upper_bound)
        return X

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()

        self.upper_bounds_ = {
            column: float(X[column].quantile(self.cap_quantile))
            for column in self.cols_to_cap
            if column in X.columns
        }
        X = self._cap(X)

        self.global_median_price_ = float(X["price"].median()) or 1.0
        self.neighbourhood_median_price_ = (
            X.groupby("neighbourhood")["price"].median().to_dict()
        )
        self.group_median_price_ = (
            X.groupby("neighbourhood_group")["price"].median().to_dict()
        )
        self.neighbourhood_counts_ = X["neighbourhood"].value_counts().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._cap(pd.DataFrame(X).copy())

        X["reviews_per_month"] = X["reviews_per_month"].fillna(0)

        group_median = (
            X["neighbourhood_group"]
            .map(self.group_median_price_)
            .astype(float)
            .fillna(self.global_median_price_)
            .replace(0, self.global_median_price_)
        )
        neighbourhood_median = (
            X["neighbourhood"]
            .map(self.neighbourhood_median_price_)
            .astype(float)
            .fillna(group_median)
            .replace(0, self.global_median_price_)
        )

        X["log_price"] = np.log1p(X["price"].clip(lower=0))
        X["price_to_neighbourhood_median"] = X["price"] / neighbourhood_median
        X["price_to_group_median"] = X["price"] / group_median
        X["has_reviews"] = (X["number_of_reviews"] > 0).astype(int)
        X["long_stay_minimum"] = (X["minimum_nights"] >= 30).astype(int)
        X["zero_availability"] = (X["availability_365"] == 0).astype(int)
        X["log_host_listings"] = np.log1p(X["calculated_host_listings_count"])
        X["occupancy_proxy"] = X["reviews_per_month"] * X["minimum_nights"]
        X["neighbourhood_listing_count"] = (
            X["neighbourhood"].map(self.neighbourhood_counts_).astype(float).fillna(0)
        )
        return X


class WeightedDecisionClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps a (calibrated) probabilistic classifier.

    - predict_proba(): returns the wrapped model's (calibrated) probabilities.
    - predict(): argmax(probability * class_weight), which lets a rare class
      like "Shared room" be predicted more often without distorting the
      probabilities themselves.
    """

    def __init__(self, estimator, class_weights: dict | None = None):
        self.estimator = estimator
        self.class_weights = class_weights

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        weights = self.class_weights or {}
        self.weight_vector_ = np.array(
            [float(weights.get(str(c), 1.0)) for c in self.classes_]
        )
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        probabilities = self.predict_proba(X) * self.weight_vector_
        return self.classes_[np.argmax(probabilities, axis=1)]


def tune_class_weights(
    y_true,
    probabilities: np.ndarray,
    classes: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[dict[str, float], float]:
    """
    Grid-search per-class decision weights that maximise macro-F1.

    Must be run on out-of-fold (training) probabilities, never on the test set.
    The most frequent class is fixed at weight 1.0.
    """
    if grid is None:
        grid = np.round(np.arange(0.5, 8.01, 0.25), 2)

    y_true = np.asarray(y_true)
    counts = {c: int((y_true == c).sum()) for c in classes}
    reference_class = max(counts, key=counts.get)
    free_classes = [c for c in classes if c != reference_class]

    best_weights = {str(c): 1.0 for c in classes}
    best_score = f1_score(
        y_true, classes[np.argmax(probabilities, axis=1)], average="macro"
    )

    for combo in itertools.product(grid, repeat=len(free_classes)):
        weights = {str(reference_class): 1.0}
        weights.update({str(c): float(w) for c, w in zip(free_classes, combo)})
        vector = np.array([weights[str(c)] for c in classes])
        predictions = classes[np.argmax(probabilities * vector, axis=1)]
        score = f1_score(y_true, predictions, average="macro")
        if score > best_score:
            best_score, best_weights = score, weights

    return best_weights, float(best_score)


## to run this project uv run python .\NYC_Airbnb_Room\train_model.py
