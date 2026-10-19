"""
German Credit Risk Predictor — Core Logic
==========================================

Non-UI module containing:
- Dataset discovery / loading / cleaning
- Feature engineering
- Preprocessing pipeline construction
- Model definitions
- Training + cross-validation
- Evaluation metrics
- Threshold analysis

This module has no Streamlit dependency and can be imported,
tested, or reused independently of the app UI.

Expected target column:
    Risk

Target:
    0 = Good
    1 = Bad
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_CANDIDATES = [
    "German Credit Risk - With Target.csv",
    "German Credit Risk - With Target.xlsx",
    "german_credit_data.csv",
    "german_credit_data.xlsx",
    "GermanCredit.csv",
    "German Credit Risk.csv",
    "credit_risk.csv",
]

TARGET = "Risk"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# =============================================================================
# DATASET FUNCTIONS
# =============================================================================


def find_dataset():
    base = Path.cwd()

    for filename in DATA_CANDIDATES:
        path = base / filename

        if path.exists():
            return path

    for pattern in ["*.csv", "*.xlsx", "*.xls"]:
        for path in base.glob(pattern):
            name = path.name.lower()

            if "credit" in name or "german" in name:
                return path

    return None


def read_dataset(path):

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    return pd.read_excel(path)


def clean_dataset(df):

    df = df.copy()

    # Remove unnamed columns
    unnamed = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]

    if unnamed:
        df = df.drop(columns=unnamed)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Locate target
    if TARGET not in df.columns:
        matches = [c for c in df.columns if str(c).strip().lower() == TARGET.lower()]

        if matches:
            df = df.rename(columns={matches[0]: TARGET})

        else:
            raise ValueError(
                f"Target '{TARGET}' not found.\nAvailable columns: {list(df.columns)}"
            )

    # Remove empty rows/columns
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # Normalize target
    target = df[TARGET].astype(str).str.strip().str.lower()

    target_map = {
        "good": 0,
        "bad": 1,
        "0": 0,
        "1": 1,
    }

    df[TARGET] = target.map(target_map)

    df = df.dropna(subset=[TARGET])

    df[TARGET] = df[TARGET].astype(int)

    # Convert numeric-looking categorical columns
    for col in df.columns:
        if col == TARGET:
            continue

        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")

            original_count = df[col].notna().sum()
            converted_count = converted.notna().sum()

            if original_count > 0 and converted_count / original_count >= 0.95:
                df[col] = converted

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


def feature_engineering(df):

    df = df.copy()

    columns_lower = {str(c).lower(): c for c in df.columns}

    age = columns_lower.get("age")
    credit = columns_lower.get("credit amount")
    duration = columns_lower.get("duration")

    if age and credit:
        age_values = pd.to_numeric(df[age], errors="coerce").replace(0, np.nan)

        credit_values = pd.to_numeric(df[credit], errors="coerce")

        df["credit_age_ratio"] = credit_values / age_values

    if credit and duration:
        credit_values = pd.to_numeric(df[credit], errors="coerce")

        duration_values = pd.to_numeric(df[duration], errors="coerce").replace(
            0, np.nan
        )

        df["credit_per_month"] = credit_values / duration_values

    if duration:
        duration_values = pd.to_numeric(df[duration], errors="coerce")

        df["long_duration"] = (duration_values >= 36).astype(int)

    if age:
        age_values = pd.to_numeric(df[age], errors="coerce")

        df["age_group"] = pd.cut(
            age_values,
            bins=[0, 25, 35, 50, 65, 120],
            labels=[
                "Young",
                "Adult",
                "Middle",
                "Senior",
                "Older",
            ],
        )

    return df


# =============================================================================
# PREPROCESSOR
# =============================================================================


def build_preprocessor(X):

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "num",
                numeric_pipe,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                categorical_pipe,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def make_pipeline(model, X):

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", model),
        ]
    )


# =============================================================================
# MODELS
# =============================================================================


def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            probability=True,
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=20,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        ),
    }


# =============================================================================
# PROBABILITY
# =============================================================================


def get_probability(model, X):

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)

        return 1 / (1 + np.exp(-np.clip(raw, -50, 50)))

    return None


# =============================================================================
# MODEL EVALUATION
# =============================================================================


def evaluate_pipeline(pipeline, X_test, y_test):

    predictions = pipeline.predict(X_test)

    probabilities = get_probability(pipeline, X_test)

    metrics = {
        "Test Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
    }

    if probabilities is not None:
        metrics["ROC AUC"] = roc_auc_score(y_test, probabilities)

        metrics["PR AUC"] = average_precision_score(y_test, probabilities)

    else:
        metrics["ROC AUC"] = np.nan
        metrics["PR AUC"] = np.nan

    return (predictions, probabilities, metrics)


def train_all_models(df):

    X = df.drop(columns=[TARGET])

    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    trained_models = {}

    results = []

    progress = st.progress(0, text="Training models...")

    models_dict = get_models()

    total_models = len(models_dict)

    for index, (name, estimator) in enumerate(models_dict.items(), start=1):
        pipeline = make_pipeline(estimator, X_train)

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        pipeline.fit(X_train, y_train)

        predictions, probabilities, metrics = evaluate_pipeline(
            pipeline, X_test, y_test
        )

        trained_models[name] = pipeline

        results.append(
            {
                "Model": name,
                "CV Accuracy": cv_results["test_accuracy"].mean(),
                "CV Accuracy Std": cv_results["test_accuracy"].std(),
                "CV Precision": cv_results["test_precision"].mean(),
                "CV Recall": cv_results["test_recall"].mean(),
                "CV F1": cv_results["test_f1"].mean(),
                "CV ROC AUC": cv_results["test_roc_auc"].mean(),
                "CV PR AUC": cv_results["test_pr_auc"].mean(),
                "Test Accuracy": metrics["Test Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "ROC AUC": metrics["ROC AUC"],
                "PR AUC": metrics["PR AUC"],
            }
        )

        progress.progress(index / total_models, text=f"Training {name}...")

    progress.empty()

    results_df = pd.DataFrame(results)

    # Rank using CV F1 + ROC AUC rather than
    # accuracy alone.
    results_df["Model Score"] = (
        0.40 * results_df["CV F1"]
        + 0.35 * results_df["CV ROC AUC"]
        + 0.25 * results_df["CV Accuracy"]
    )

    results_df = results_df.sort_values(by="Model Score", ascending=False).reset_index(
        drop=True
    )

    return (
        trained_models,
        results_df,
        X_train,
        X_test,
        y_train,
        y_test,
    )


# =============================================================================
# THRESHOLD ANALYSIS
# =============================================================================


def threshold_analysis(model, X_test, y_test):

    probabilities = get_probability(model, X_test)

    if probabilities is None:
        return 0.50, pd.DataFrame()

    rows = []

    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
            }
        )

    threshold_df = pd.DataFrame(rows)

    # F1 is more useful than accuracy for
    # identifying a useful credit-risk threshold.
    best_row = threshold_df.loc[threshold_df["F1"].idxmax()]

    return (float(best_row["Threshold"]), threshold_df)


# =============================================================================
# CACHE
# =============================================================================


@st.cache_data(show_spinner=False)
def prepare_data(raw_df):

    df = clean_dataset(raw_df)

    df = feature_engineering(df)

    return df


@st.cache_resource(show_spinner=False)
def train_cached_models(cleaned_df):

    return train_all_models(cleaned_df)
