#!/usr/bin/env python
# coding: utf-8


"""
Customer Churn Prediction
- EDA with dark theme (plots are saved to ./eda_plots)
- Imbalance-aware model training and hyperparameter tuning
- Model selection by cross-validated ROC-AUC (not accuracy / F1)
- Best-model persistence
- Metrics export for the Streamlit dashboard (ui.py)
- Sanity checks and inference example
"""

import os

# Must be set BEFORE importing numpy, pandas, scikit-learn, etc.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
SHOW_PLOTS = False  # True = open plot windows; False = only save PNGs
N_JOBS = 1

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_churn_data.csv"
MODEL_PATH = BASE_DIR / "churn_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.csv"
PLOTS_DIR = BASE_DIR / "eda_plots"
PLOTS_DIR.mkdir(exist_ok=True)

FEMALE_CODE = 1  # must match FEMALE_CODE in ui.py

# -----------------------------
# Dark theme configuration
# -----------------------------
plt.style.use("dark_background")
sns.set_theme(style="darkgrid", rc=plt.style.library["dark_background"])
sns.set_palette("bright")

plt.rcParams.update(
    {
        "figure.facecolor": "#111111",
        "axes.facecolor": "#111111",
        "savefig.facecolor": "#111111",
        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": "#ffffff",
        "text.color": "#ffffff",
        "xtick.color": "#ffffff",
        "ytick.color": "#ffffff",
        "grid.color": "#333333",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    }
)


def finish_plot(name):
    """Save the current figure, optionally show it, then close it."""
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{name}.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# -----------------------------
# 1. Load and inspect data
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head())

print("\nDataset info:")
df.info()

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print("\nTotal missing values after handling:", df.isna().sum().sum())


# -----------------------------
# 2. Exploratory data analysis
# -----------------------------
numeric_columns = df.select_dtypes(include=["number"])

print("\nNumeric correlation matrix:")
print(numeric_columns.corr())

plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_columns.corr(),
    annot=True,
    fmt=".2f",
    cmap="mako",
    linewidths=0.5,
    linecolor="#111111",
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Matrix")
finish_plot("correlation_matrix")

print("\nChurn distribution:")
print(df["Churn"].value_counts())

churn_counts = df["Churn"].value_counts()

fig, ax = plt.subplots(figsize=(6, 5))
ax.pie(
    churn_counts,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#ff6b6b", "#4ecdc4"],
    wedgeprops={"edgecolor": "#111111", "linewidth": 1},
)
ax.set_title("Churn Distribution")
ax.set_ylabel("")
finish_plot("churn_distribution")

print("\nMean monthly charges by churn:")
print(df.groupby("Churn")["MonthlyCharges"].mean())

# FIXED: changed "Gender" to lowercase "gender" to match dataset column name
print("\nMean monthly charges by churn and gender:")
print(df.groupby(["Churn", "gender"])["MonthlyCharges"].mean())

print("\nMean tenure by churn:")
print(df.groupby("Churn")["tenure"].mean())


# -----------------------------
# 3. Feature and target preparation
# -----------------------------
y = df["Churn"].apply(lambda value: 1 if str(value).strip().lower() == "yes" else 0)

X = df.drop(columns=["customerID", "Churn"])

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

churn_share = y.mean()
majority_share = max(churn_share, 1 - churn_share)

print("\nPositive churn samples:", int(y.sum()), f"of {len(y)}")
print(f"Churn share: {churn_share:.1%}")
print(
    f"Majority-class baseline accuracy: {majority_share:.1%} "
    "(any model must beat this to be useful)"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ]
)


# -----------------------------
# 4. Evaluation helper
# -----------------------------
def evaluate_model(name, model, X_test, y_test, cv_roc_auc=np.nan):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy:.4f} | Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "CV_ROC_AUC": cv_roc_auc,
    }


# -----------------------------
# 5. Baseline
# -----------------------------
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_results = {
    "Model": "Baseline (always predict majority)",
    "Accuracy": accuracy_score(y_test, baseline.predict(X_test)),
    "Balanced_Accuracy": balanced_accuracy_score(y_test, baseline.predict(X_test)),
    "Precision": 0.0,
    "Recall": 0.0,
    "F1": 0.0,
    "ROC_AUC": 0.5,
    "PR_AUC": y_test.mean(),
    "CV_ROC_AUC": 0.5,
}


# -----------------------------
# 6. Models and hyperparameter grids
# -----------------------------
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

model_specs = {
    "Logistic Regression": (
        Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "Decision Tree": (
        Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        {
            "model__max_depth": [3, 5, 7, 10],
            "model__min_samples_split": [2, 10, 20],
            "model__min_samples_leaf": [5, 10, 20],
            "model__criterion": ["gini", "entropy"],
        },
    ),
    "Random Forest": (
        Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS,
                    ),
                ),
            ]
        ),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 8, 12],
            "model__min_samples_leaf": [5, 10],
        },
    ),
}


# -----------------------------
# 7. Train, tune and evaluate every model
# -----------------------------
fitted_searches = {}
results = [baseline_results]

for name, (estimator, param_grid) in model_specs.items():
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=CV,
        scoring="roc_auc",
        n_jobs=N_JOBS,
        pre_dispatch=2 * N_JOBS,
    )
    search.fit(X_train, y_train)

    print(f"\nBest {name} parameters:", search.best_params_)
    print(f"Best {name} cross-validation ROC-AUC: {search.best_score_:.4f}")

    fitted_searches[name] = search
    results.append(
        evaluate_model(name, search, X_test, y_test, cv_roc_auc=search.best_score_)
    )


# -----------------------------
# 8. Compare models
# -----------------------------
all_results = pd.DataFrame(results).sort_values(by="CV_ROC_AUC", ascending=False)

print("\n===== Model Comparison (sorted by cross-validated ROC-AUC) =====")
print(all_results.round(4).to_string(index=False))


# -----------------------------
# 9. Save best model and metrics
# -----------------------------
best_model_name = max(
    fitted_searches, key=lambda model_name: fitted_searches[model_name].best_score_
)
best_model = fitted_searches[best_model_name].best_estimator_

joblib.dump(best_model, MODEL_PATH)
all_results.to_csv(METRICS_PATH, index=False)

print(f"\nSaved best model: {best_model_name}")
print(f"Model saved to: {MODEL_PATH}")
print(f"Model metrics saved to: {METRICS_PATH}")


# -----------------------------
# 10. Inference example
# -----------------------------
loaded_model = joblib.load(MODEL_PATH)
sample = X_test.iloc[[0]]

prediction = loaded_model.predict(sample)[0]
churn_probability = loaded_model.predict_proba(sample)[0][1]

print("\n===== Sample Prediction =====")
print("Predicted churn:", "Yes" if prediction == 1 else "No")
print(f"Churn probability: {churn_probability:.4f}")
