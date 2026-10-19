import warnings

warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv("email.csv")

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where Message or Category is missing
df = df.dropna(subset=["Message", "Category"])

# Convert message to string
df["Message"] = df["Message"].astype(str)

# Normalize category
df["Category"] = df["Category"].str.lower().str.strip()

# Convert labels
df["Category"] = df["Category"].map({"ham": 0, "spam": 1})

# Remove any rows that could not be mapped
df = df.dropna(subset=["Category"])

# Convert labels to integer
df["Category"] = df["Category"].astype(int)

print("Cleaned shape:", df.shape)

print("\nClass distribution:")
print(df["Category"].value_counts())

print("\nClass percentage:")
print((df["Category"].value_counts(normalize=True) * 100).round(2))


# ============================================================
# 3. BASIC EDA
# ============================================================

df["Message_Length"] = df["Message"].str.len()

df["Word_Count"] = df["Message"].apply(lambda x: len(x.split()))

print("\nMessage length statistics:")
print(df["Message_Length"].describe())

print("\nAverage message length by class:")
print(df.groupby("Category")["Message_Length"].mean())

print("\nAverage word count by class:")
print(df.groupby("Category")["Word_Count"].mean())


# ============================================================
# 4. DATA VISUALIZATION
# ============================================================

plt.style.use("ggplot")

plt.figure(figsize=(6, 6))

df["Category"].map({0: "Ham", 1: "Spam"}).value_counts().plot(
    kind="pie", autopct="%1.1f%%"
)

plt.ylabel("")
plt.title("Email Category Distribution")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="Message_Length", hue="Category", bins=40)

plt.title("Distribution of Message Length")
plt.xlabel("Characters")
plt.tight_layout()
plt.show()


# ============================================================
# 5. FEATURES AND TARGET
# ============================================================

X = df["Message"]
y = df["Category"]

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(y.value_counts())


# ============================================================
# 6. TRAIN TEST SPLIT
# IMPORTANT:
# ONLY SPLIT ONCE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())


# ============================================================
# 7. TF-IDF VECTORIZATION
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF VECTORIZATION")
print("=" * 70)

tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))

# IMPORTANT:
# Fit ONLY on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform test data using SAME vectorizer
X_test_tfidf = tfidf.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape :", X_test_tfidf.shape)


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {
    "Naive Bayes": (MultinomialNB(), {"alpha": [0.1, 0.5, 1, 2, 5]}),
    "Logistic Regression": (
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        {"C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Linear SVM": (
        LinearSVC(class_weight="balanced", random_state=42),
        {"C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
    ),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {"max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10]},
    ),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9]}),
}


# ============================================================
# 9. TRAIN AND TUNE MODELS
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

results = []
best_models = {}

for name, (model, params) in models.items():
    print(f"\nTraining: {name}")

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=10,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
    )

    search.fit(X_train_tfidf, y_train)

    model_best = search.best_estimator_

    # Save fitted model
    best_models[name] = model_best

    # Predictions
    predictions = model_best.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions, zero_division=0)

    recall = recall_score(y_test, predictions, zero_division=0)

    f1 = f1_score(y_test, predictions, zero_division=0)

    # ROC-AUC
    try:
        if hasattr(model_best, "decision_function"):
            scores = model_best.decision_function(X_test_tfidf)

            auc = roc_auc_score(y_test, scores)

        elif hasattr(model_best, "predict_proba"):
            probabilities = model_best.predict_proba(X_test_tfidf)[:, 1]

            auc = roc_auc_score(y_test, probabilities)

        else:
            auc = np.nan

    except Exception:
        auc = np.nan

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": auc,
            "Best Params": search.best_params_,
        }
    )

    print("Best Parameters:", search.best_params_)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="F1", ascending=False)

print(results_df.to_string(index=False))


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.loc[results_df["F1"].idxmax(), "Model"]

best_model = best_models[best_model_name]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Best Model:", best_model_name)
print("Model:", best_model)


# ============================================================
# 12. FINAL TEST PREDICTIONS
# ============================================================

y_pred = best_model.predict(X_test_tfidf)

print("\nActual distribution:")
print(y_test.value_counts())

print("\nPredicted distribution:")
print(pd.Series(y_pred).value_counts())


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(y_test, y_pred, target_names=["Ham", "Spam"], zero_division=0)
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Ham", "Spam"],
    cmap="Blues",
    values_format="d",
    ax=ax,
)

ax.set_title(f"Confusion Matrix - {best_model_name}")

plt.tight_layout()
plt.show()


# ============================================================
# 15. TEST CUSTOM EMAILS
# ============================================================

print("\n" + "=" * 70)
print("CUSTOM EMAIL TESTING")
print("=" * 70)

emails = [
    "Congratulations! You have won a $500 Amazon gift card. Click here to claim your prize now.",
    "Hi Rahul, are we still meeting for lunch at 1 PM tomorrow?",
    "URGENT! Your account has won a $1,000 cash reward. Claim it before the offer expires.",
    "Hi team, please find the meeting agenda attached. We will discuss the project updates at 3 PM.",
    "You have been selected for an exclusive lottery prize of $10,000. Send your details to receive the money.",
    "Hey, can you please send me the notes from today's class? I missed the lecture.",
    "FREE iPhone 16 Pro! You are one of the lucky winners. Click the link below to claim your phone.",
    "Your electricity bill is due tomorrow. Please make the payment through the usual payment portal.",
    "WINNER! You have been chosen to receive a brand-new car. Call this number immediately to claim your prize.",
    "Hi, just checking if you received my email about the interview scheduled for Monday at 10 AM.",
]


# Transform custom emails
emails_vectorized = tfidf.transform(emails)

# Predict
predictions = best_model.predict(emails_vectorized)

for email, prediction in zip(emails, predictions):
    label = "Spam" if prediction == 1 else "Ham"

    print("\nEmail:")
    print(email)

    print("Prediction:")
    print(label)

    print("-" * 70)


# ============================================================
# 16. SAVE MODEL + TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

model_package = {"model": best_model, "tfidf": tfidf, "model_name": best_model_name}

with open("spam_classifier.pkl", "wb") as file:
    pickle.dump(model_package, file)

print("Saved successfully as spam_classifier.pkl")


# ============================================================
# 17. LOAD TEST
# ============================================================

print("\n" + "=" * 70)
print("TESTING SAVED MODEL")
print("=" * 70)

with open("spam_classifier.pkl", "rb") as file:
    saved_package = pickle.load(file)

loaded_model = saved_package["model"]
loaded_tfidf = saved_package["tfidf"]

test_email = ["Congratulations! You won a free iPhone. Click here to claim your prize."]

test_vector = loaded_tfidf.transform(test_email)

test_prediction = loaded_model.predict(test_vector)[0]

test_label = "Spam" if test_prediction == 1 else "Ham"

print("Test Email:")
print(test_email[0])

print("\nPrediction:")
print(test_label)

print("\nModel loading test completed successfully.")
