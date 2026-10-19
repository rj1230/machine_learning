from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
# --------------------------------------------------------------------------

import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna.exceptions import TrialPruned
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from nyc_features import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    FeatureEngineer,
    WeightedDecisionClassifier,
    tune_class_weights,
)


# ============================================================
# CONFIGURATION
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.33
TUNING_SPLITS = int(os.getenv("TUNING_SPLITS", 3))
CALIBRATION_SPLITS = 3
N_TRIALS = int(os.getenv("N_TRIALS", 40))

# Run several Optuna trials concurrently (threads). Optuna's study.optimize
# uses a thread pool for n_jobs > 1, and scikit-learn's fit calls release the
# GIL for the heavy numeric work, so this gives real speedup even in Python.
# Default is 1 (fully sequential, identical behavior to before). Set e.g.
# OPTUNA_N_JOBS=4 to parallelize.
OPTUNA_N_JOBS = int(os.getenv("OPTUNA_N_JOBS", 1))

# Guard against oversubscription: RandomForest's own n_jobs=-1 would spawn a
# full set of worker processes PER concurrent trial. If we're running trials
# in parallel, force each individual model fit to be single-threaded instead
# and let trial-level parallelism do the work. Sequential runs (the default)
# keep the original n_jobs=-1 behavior.
CLASSIFIER_N_JOBS = 1 if OPTUNA_N_JOBS > 1 else -1

DATA_PATH = BASE_DIR / "nyc.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = MODELS_DIR / "Best_Model_Pipeline.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
TRIALS_PATH = REPORTS_DIR / "optuna_trials.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASELINE_PARAMS = {
    "model_type": "random_forest",
    "rf_n_estimators": 200,
    "rf_max_depth": None,
    "rf_min_samples_split": 2,
    "rf_min_samples_leaf": 1,
    "rf_max_features": "sqrt",
    "rf_class_weight": "balanced",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def print_section(title: str) -> None:
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def save_plot(filename: str) -> None:
    """Save the current Matplotlib figure in the reports directory."""
    output_path = REPORTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close("all")
    print(f"Saved plot: {output_path.name}")


def create_preprocessor() -> ColumnTransformer:
    """Tree models don't need scaling/power transforms: impute + one-hot only."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, NUMERICAL_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def classifier_from_params(params: dict):
    if params["model_type"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_split=params["rf_min_samples_split"],
            min_samples_leaf=params["rf_min_samples_leaf"],
            max_features=params["rf_max_features"],
            class_weight=params["rf_class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=CLASSIFIER_N_JOBS,
        )

    return HistGradientBoostingClassifier(
        learning_rate=params["hgb_learning_rate"],
        max_iter=params["hgb_max_iter"],
        max_leaf_nodes=params["hgb_max_leaf_nodes"],
        min_samples_leaf=params["hgb_min_samples_leaf"],
        l2_regularization=params["hgb_l2_regularization"],
        class_weight=params["hgb_class_weight"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )


def create_model_pipeline(params: dict) -> Pipeline:
    """Full raw-input -> prediction pipeline."""
    return Pipeline(
        steps=[
            ("features", FeatureEngineer(cap_quantile=0.99)),
            ("preprocessor", create_preprocessor()),
            ("classifier", classifier_from_params(params)),
        ]
    )


def suggest_params(trial: optuna.Trial) -> dict:
    model_type = trial.suggest_categorical(
        "model_type",
        ["random_forest", "hist_gradient_boosting"],
    )

    params = {"model_type": model_type}

    if model_type == "random_forest":
        params.update(
            {
                "rf_n_estimators": trial.suggest_int(
                    "rf_n_estimators",
                    200,
                    600,
                    step=100,
                ),
                "rf_max_depth": trial.suggest_categorical(
                    "rf_max_depth",
                    [None, 16, 24, 32, 40],
                ),
                "rf_min_samples_split": trial.suggest_int(
                    "rf_min_samples_split",
                    2,
                    10,
                ),
                "rf_min_samples_leaf": trial.suggest_int(
                    "rf_min_samples_leaf",
                    1,
                    5,
                ),
                "rf_max_features": trial.suggest_categorical(
                    "rf_max_features",
                    ["sqrt", "log2", 0.3, 0.5],
                ),
                "rf_class_weight": trial.suggest_categorical(
                    "rf_class_weight",
                    [None, "balanced", "balanced_subsample"],
                ),
            }
        )
    else:
        params.update(
            {
                "hgb_learning_rate": trial.suggest_float(
                    "hgb_learning_rate",
                    0.02,
                    0.2,
                    log=True,
                ),
                "hgb_max_iter": trial.suggest_int(
                    "hgb_max_iter",
                    200,
                    800,
                    step=100,
                ),
                "hgb_max_leaf_nodes": trial.suggest_int(
                    "hgb_max_leaf_nodes",
                    15,
                    127,
                ),
                "hgb_min_samples_leaf": trial.suggest_int(
                    "hgb_min_samples_leaf",
                    10,
                    100,
                ),
                "hgb_l2_regularization": trial.suggest_float(
                    "hgb_l2_regularization",
                    1e-4,
                    10.0,
                    log=True,
                ),
                "hgb_class_weight": trial.suggest_categorical(
                    "hgb_class_weight",
                    [None, "balanced"],
                ),
            }
        )

    return params


def precompute_transformed_folds(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> list[tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]]:
    """
    Fit FeatureEngineer + ColumnTransformer ONCE per CV fold and cache the
    resulting arrays.

    Preprocessing does not depend on any classifier hyperparameter, so it is
    wasteful to redo it inside every Optuna trial.
    """
    folds = []

    for train_idx, valid_idx in cv.split(X, y):
        transformer = Pipeline(
            steps=[
                ("features", FeatureEngineer(cap_quantile=0.99)),
                ("preprocessor", create_preprocessor()),
            ]
        )

        X_train_fold = X.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_train_fold = y.iloc[train_idx]
        y_valid_fold = y.iloc[valid_idx]

        X_train_t = transformer.fit_transform(
            X_train_fold,
            y_train_fold,
        )
        X_valid_t = transformer.transform(X_valid_fold)

        folds.append(
            (
                X_train_t,
                X_valid_t,
                y_train_fold,
                y_valid_fold,
            )
        )

    return folds


def cross_validate_params(
    params: dict,
    folds: list[tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]],
    trial: optuna.Trial | None = None,
) -> dict[str, float]:
    """
    Manual CV loop over pre-transformed folds so Optuna can prune after every
    fold. Only the classifier is fit here because preprocessing was already
    done in precompute_transformed_folds().
    """
    f1_scores = []
    accuracy_scores = []

    for step, (
        X_train_t,
        X_valid_t,
        y_train_fold,
        y_valid_fold,
    ) in enumerate(folds):
        model = classifier_from_params(params)

        model.fit(
            X_train_t,
            y_train_fold,
        )

        predictions = model.predict(X_valid_t)

        f1_scores.append(
            f1_score(
                y_valid_fold,
                predictions,
                average="macro",
            )
        )

        accuracy_scores.append(
            accuracy_score(
                y_valid_fold,
                predictions,
            )
        )

        if trial is not None:
            trial.report(
                float(np.mean(f1_scores)),
                step,
            )

            if trial.should_prune():
                raise TrialPruned()

    return {
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "accuracy_mean": float(np.mean(accuracy_scores)),
        "accuracy_std": float(np.std(accuracy_scores)),
    }


def run_eda(df: pd.DataFrame) -> None:
    """Generate and save lightweight EDA visuals."""

    print_section("EXPLORATORY DATA ANALYSIS")

    print(f"Dataset shape: {df.shape}")

    print("\nRoom-type distribution:")
    print(df["room_type"].value_counts())

    plt.figure(figsize=(9, 5))

    ax = sns.countplot(
        data=df,
        x="room_type",
        order=df["room_type"].value_counts().index,
    )

    plt.title(
        "Distribution of NYC Airbnb Room Types",
        fontweight="bold",
    )
    plt.xlabel("Room Type")
    plt.ylabel("Number of Listings")
    plt.xticks(rotation=10)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%d",
        )

    save_plot("room_type_distribution.png")

    numeric_columns = [
        "price",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    available_numeric_columns = [c for c in numeric_columns if c in df.columns]

    df[available_numeric_columns].hist(
        bins=30,
        figsize=(14, 9),
        edgecolor="black",
    )

    plt.suptitle(
        "Distribution of Numerical Features",
        fontweight="bold",
        fontsize=16,
    )

    save_plot("numerical_feature_distributions.png")

    if "neighbourhood_group" in df.columns:
        plt.figure(figsize=(10, 5))

        ax = sns.countplot(
            data=df,
            x="neighbourhood_group",
            order=df["neighbourhood_group"].value_counts().index,
        )

        plt.title(
            "Listings by Neighbourhood Group",
            fontweight="bold",
        )
        plt.xlabel("Neighbourhood Group")
        plt.ylabel("Number of Listings")

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%d",
            )

        save_plot("neighbourhood_group_distribution.png")

    if {"price", "room_type"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))

        sns.boxplot(
            data=df,
            x="room_type",
            y="price",
        )

        plt.title(
            "Price Distribution by Room Type",
            fontweight="bold",
        )
        plt.xlabel("Room Type")
        plt.ylabel("Price")
        plt.ylim(
            0,
            df["price"].quantile(0.99) * 1.05,
        )

        save_plot("price_by_room_type.png")

    correlation_columns = [
        "latitude",
        "longitude",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    available_correlation_columns = [c for c in correlation_columns if c in df.columns]

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        df[available_correlation_columns].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )

    plt.title(
        "Numerical Feature Correlation Matrix",
        fontweight="bold",
    )

    save_plot("correlation_matrix.png")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print_section("NYC AIRBNB ROOM TYPE CLASSIFICATION")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place nyc.csv inside the NYC_Airbnb_Room folder."
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded: {DATA_PATH.name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    required_columns = {
        "room_type",
        *NUMERICAL_COLUMNS[:8],
        *CATEGORICAL_COLUMNS,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"The dataset is missing required columns: {sorted(missing_columns)}"
        )

    print_section("MISSING VALUES")

    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

    print(
        "No missing values found."
        if missing_values.empty
        else missing_values.to_string()
    )

    run_eda(df)

    # --------------------------------------------------------
    print_section("DATA CLEANING")

    columns_to_remove = [
        "id",
        "name",
        "host_name",
        "last_review",
        "host_id",
    ]

    df_clean = df.drop(
        columns=columns_to_remove,
        errors="ignore",
    ).copy()

    print(f"Removed columns if present: {columns_to_remove}")

    print(f"Cleaned dataset shape: {df_clean.shape}")

    X = df_clean.drop(columns="room_type")

    y = df_clean["room_type"]

    # --------------------------------------------------------
    print_section("TRAIN / TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    print(f"Training samples: {len(X_train):,}")

    print(f"Testing samples:  {len(X_test):,}")

    print("\nTraining class distribution:")

    print(y_train.value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

    tuning_cv = StratifiedKFold(
        n_splits=TUNING_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # KEY EFFICIENCY CHANGE: fit FeatureEngineer + preprocessor once per
    # tuning fold and reuse the cached arrays for every baseline
    # score and every Optuna trial.

    print_section("PRECOMPUTING CV FOLDS (feature engineering + encoding)")

    fold_start = time.time()

    cv_folds = precompute_transformed_folds(
        X_train,
        y_train,
        tuning_cv,
    )

    print(
        f"Cached {len(cv_folds)} preprocessed folds in "
        f"{time.time() - fold_start:.1f}s "
        "(reused across baseline + all Optuna trials)"
    )

    # --------------------------------------------------------
    print_section("BASELINE MODEL CROSS-VALIDATION")

    baseline_scores = cross_validate_params(
        BASELINE_PARAMS,
        cv_folds,
    )

    print(
        f"Random Forest baseline (+ engineered features, "
        f"{TUNING_SPLITS}-fold): "
        f"Accuracy {baseline_scores['accuracy_mean']:.4f} "
        f"± {baseline_scores['accuracy_std']:.4f} | "
        f"Macro-F1 {baseline_scores['f1_mean']:.4f} "
        f"± {baseline_scores['f1_std']:.4f}"
    )

    # --------------------------------------------------------
    print_section("OPTUNA HYPERPARAMETER TUNING")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)

        scores = cross_validate_params(
            params,
            cv_folds,
            trial,
        )

        trial.set_user_attr(
            "accuracy_mean",
            scores["accuracy_mean"],
        )

        return scores["f1_mean"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
        ),
    )

    study.enqueue_trial(BASELINE_PARAMS)

    print(
        f"Running with OPTUNA_N_JOBS={OPTUNA_N_JOBS} "
        f"(classifier n_jobs={CLASSIFIER_N_JOBS})"
    )

    start = time.time()

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=OPTUNA_N_JOBS,
        show_progress_bar=True,
    )

    elapsed = time.time() - start

    study.trials_dataframe().to_csv(
        TRIALS_PATH,
        index=False,
    )

    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)

    print(
        f"Finished {len(study.trials)} trials in "
        f"{elapsed / 60:.1f} min ({n_pruned} pruned)"
    )

    print(f"Baseline CV Macro-F1: {baseline_scores['f1_mean']:.4f}")

    print(f"Best CV Macro-F1:     {study.best_value:.4f}")

    print("Best parameters:")

    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    if study.best_value >= baseline_scores["f1_mean"]:
        best_params = study.best_params
        best_cv_f1 = study.best_value
    else:
        print("Tuned model did not beat baseline -> keeping baseline parameters.")

        best_params = BASELINE_PARAMS
        best_cv_f1 = baseline_scores["f1_mean"]

    # Free the cached folds - not needed past this point,
    # and can be sizeable.
    del cv_folds

    # --------------------------------------------------------
    # Calibration needs the FULL pipeline.
    # It is only run once, not per Optuna trial.

    print_section("CALIBRATION + DECISION-WEIGHT TUNING")

    calibrated_template = CalibratedClassifierCV(
        estimator=create_model_pipeline(best_params),
        method="isotonic",
        cv=StratifiedKFold(
            n_splits=CALIBRATION_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
    )

    oof_proba = cross_val_predict(
        calibrated_template,
        X_train,
        y_train,
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE + 1,
        ),
        method="predict_proba",
        n_jobs=1,
    )

    classes = np.unique(y_train)

    oof_argmax_f1 = f1_score(
        y_train,
        classes[np.argmax(oof_proba, axis=1)],
        average="macro",
    )

    class_weights, oof_weighted_f1 = tune_class_weights(
        y_train,
        oof_proba,
        classes,
    )

    print(f"OOF Macro-F1 (plain argmax):      {oof_argmax_f1:.4f}")

    print(f"OOF Macro-F1 (tuned weights):     {oof_weighted_f1:.4f}")

    print("Decision weights:")

    for room_class, weight in class_weights.items():
        print(f"  {room_class}: {weight}")

    # --------------------------------------------------------
    print_section("TRAINING FINAL MODEL")

    final_model = WeightedDecisionClassifier(
        estimator=calibrated_template,
        class_weights=class_weights,
    )

    final_model.fit(
        X_train,
        y_train,
    )

    print(f"Model type: {best_params['model_type']}")

    # --------------------------------------------------------
    print_section("FINAL TEST-SET EVALUATION")

    y_proba = final_model.predict_proba(X_test)

    y_pred = final_model.predict(X_test)

    y_pred_argmax = final_model.classes_[np.argmax(y_proba, axis=1)]

    test_accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    test_macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    test_macro_f1_argmax = f1_score(
        y_test,
        y_pred_argmax,
        average="macro",
    )

    print(f"Test Accuracy:                 {test_accuracy:.4f}")

    print(f"Test Macro-F1 (tuned weights): {test_macro_f1:.4f}")

    print(f"Test Macro-F1 (plain argmax):  {test_macro_f1_argmax:.4f} (reference only)")

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
        )
    )

    # --------------------------------------------------------
    print_section("CONFUSION MATRIX")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=final_model.classes_,
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=final_model.classes_,
        yticklabels=final_model.classes_,
        linewidths=0.5,
    )

    plt.title(
        "Confusion Matrix: Final Model",
        fontweight="bold",
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    save_plot("confusion_matrix.png")

    # --------------------------------------------------------
    print_section("CALIBRATION ANALYSIS")

    n_classes = len(final_model.classes_)

    fig, axes = plt.subplots(
        1,
        n_classes,
        figsize=(6 * n_classes, 5),
    )

    axes = np.atleast_1d(axes)

    brier_scores: dict[str, float] = {}

    for index, room_class in enumerate(final_model.classes_):
        y_binary = (y_test == room_class).astype(int)

        class_probability = y_proba[:, index]

        fraction_positive, mean_predicted_value = calibration_curve(
            y_binary,
            class_probability,
            n_bins=10,
            strategy="quantile",
        )

        brier = brier_score_loss(
            y_binary,
            class_probability,
        )

        brier_scores[str(room_class)] = float(brier)

        axis = axes[index]

        axis.plot(
            mean_predicted_value,
            fraction_positive,
            "s-",
            label="Calibrated model",
        )

        axis.plot(
            [0, 1],
            [0, 1],
            "k--",
            label="Perfect calibration",
        )

        axis.set_title(f"{room_class}\nBrier: {brier:.4f}")

        axis.set_xlabel("Mean predicted probability")

        axis.set_ylabel("Observed frequency")

        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

        axis.legend(loc="lower right")

    fig.suptitle(
        "Reliability Diagrams",
        fontweight="bold",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR / "reliability_diagrams.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    mean_brier_score = float(np.mean(list(brier_scores.values())))

    print("Brier scores by class:")

    for room_class, brier in brier_scores.items():
        print(f"  {room_class}: {brier:.4f}")

    print(f"Mean Brier score: {mean_brier_score:.4f}")

    # --------------------------------------------------------
    print_section("SAVING DEPLOYMENT ARTIFACT")

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    comparison = pd.DataFrame(
        [
            {
                "Stage": "Baseline RF (CV)",
                "Macro F1": baseline_scores["f1_mean"],
                "Accuracy": baseline_scores["accuracy_mean"],
            },
            {
                "Stage": (f"Best {best_params['model_type']} (CV)"),
                "Macro F1": best_cv_f1,
                "Accuracy": np.nan,
            },
            {
                "Stage": ("Calibrated, argmax (OOF)"),
                "Macro F1": oof_argmax_f1,
                "Accuracy": np.nan,
            },
            {
                "Stage": ("Calibrated, tuned weights (OOF)"),
                "Macro F1": oof_weighted_f1,
                "Accuracy": np.nan,
            },
            {
                "Stage": "Final model (TEST)",
                "Macro F1": test_macro_f1,
                "Accuracy": test_accuracy,
            },
        ]
    )

    comparison.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    metrics = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "baseline_cv_macro_f1": baseline_scores["f1_mean"],
        "best_cv_macro_f1": float(best_cv_f1),
        "oof_macro_f1_argmax": float(oof_argmax_f1),
        "oof_macro_f1_weighted": float(oof_weighted_f1),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "test_macro_f1_argmax": float(test_macro_f1_argmax),
        "per_class_f1": {
            str(c): float(s)
            for c, s in zip(
                final_model.classes_,
                f1_score(
                    y_test,
                    y_pred,
                    average=None,
                    labels=final_model.classes_,
                ),
            )
        },
        "mean_brier_score": mean_brier_score,
        "brier_scores_by_class": brier_scores,
        "decision_weights": class_weights,
        "best_parameters": best_params,
        "model_path": str(MODEL_PATH),
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
            default=str,
        )

    print(f"Model saved:   {MODEL_PATH}")

    print(f"Metrics saved: {METRICS_PATH}")

    print_section("PIPELINE COMPLETED SUCCESSFULLY")

    print(f"Baseline CV Macro-F1: {baseline_scores['f1_mean']:.4f}")

    print(f"Best CV Macro-F1:     {best_cv_f1:.4f}")

    print(f"Test Accuracy:        {test_accuracy:.4f}")

    print(f"Test Macro-F1:        {test_macro_f1:.4f}")

    print(f"Mean Brier Score:     {mean_brier_score:.4f}")


if __name__ == "__main__":
    main()
