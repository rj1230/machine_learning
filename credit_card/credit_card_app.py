"""
German Credit Risk Predictor — Streamlit App
=============================================

UI layer for:
- Multiple ML model comparison
- 5-fold stratified cross-validation
- Hold-out test evaluation
- ROC-AUC / PR-AUC
- Confusion matrix
- Classification report
- ROC curve
- Precision-Recall curve
- Threshold analysis
- Single customer prediction
- Batch CSV prediction
- Prediction history

All data loading, cleaning, feature engineering, model training,
and evaluation logic lives in credit_risk_core.py.

Expected target column:
    Risk

Target:
    0 = Good
    1 = Bad
"""

from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from credit_card.credit_risk_core import (
    TARGET,
    find_dataset,
    read_dataset,
    prepare_data,
    train_cached_models,
    get_probability,
    threshold_analysis,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="German Credit Risk",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# STYLE
# =============================================================================

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.25rem;
    }

    .app-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }

    .app-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 2px;
    }

    .result-card {
        border-radius: 16px;
        padding: 28px 30px;
        margin-top: 15px;
        border: 1px solid;
    }

    .result-good {
        background: linear-gradient(
            135deg,
            #f0fdf4 0%,
            #ecfdf5 100%
        );
        border-color: #bbf7d0;
    }

    .result-bad {
        background: linear-gradient(
            135deg,
            #fef2f2 0%,
            #fff1f2 100%
        );
        border-color: #fecaca;
    }

    .result-label {
        font-size: 1.6rem;
        font-weight: 800;
    }

    .result-good .result-label {
        color: #15803d;
    }

    .result-bad .result-label {
        color: #b91c1c;
    }

    .result-caption {
        color: #475569;
        font-size: 0.9rem;
        margin-top: 6px;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# LOAD DATASET
# =============================================================================

dataset_path = find_dataset()

if dataset_path is None:
    st.error(
        "Dataset not found.\n\n"
        "Place your German Credit dataset next to credit_risk_app.py."
    )

    st.stop()


try:
    raw_df = read_dataset(dataset_path)

    df = prepare_data(raw_df)

except Exception as exc:
    st.error(f"Could not load dataset: {exc}")

    st.stop()


if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' is missing.")

    st.stop()


# =============================================================================
# TRAIN
# =============================================================================

(
    models,
    results_df,
    X_train,
    X_test,
    y_train,
    y_test,
) = train_cached_models(df)


# =============================================================================
# BEST MODEL
# =============================================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = models[best_model_name]

best_test_accuracy = results_df.iloc[0]["Test Accuracy"]

best_cv_f1 = results_df.iloc[0]["CV F1"]

best_cv_auc = results_df.iloc[0]["CV ROC AUC"]


# =============================================================================
# SESSION STATE
# =============================================================================

if "history" not in st.session_state:
    st.session_state.history = []


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### 💳 German Credit Risk")

    st.caption("Credit risk classification")

    st.divider()

    st.success(f"Dataset: {dataset_path.name}")

    st.markdown("#### Model Selection")

    model_options = results_df["Model"].tolist()

    selected_model_name = st.selectbox(
        "Prediction model",
        model_options,
        index=model_options.index(best_model_name),
    )

    selected_model = models[selected_model_name]

    selected_row = results_df[results_df["Model"] == selected_model_name].iloc[0]

    st.markdown(f"**CV F1:** {selected_row['CV F1']:.2%}")

    st.markdown(f"**CV ROC AUC:** {selected_row['CV ROC AUC']:.2%}")

    st.markdown(f"**Test F1:** {selected_row['F1']:.2%}")

    st.markdown(f"**Test ROC AUC:** {selected_row['ROC AUC']:.2%}")

    st.divider()

    st.markdown("#### Prediction Threshold")

    threshold = st.slider(
        "Bad-risk threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
    )

    if st.session_state.history:
        st.divider()

        st.write(f"Predictions: **{len(st.session_state.history)}**")

        if st.button(
            "Clear history",
            width="stretch",
        ):
            st.session_state.history = []

            st.rerun()


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="app-header">

        <div style="font-size:2.1rem;">
            💳
        </div>

        <div>

            <p class="app-title">
                German Credit Risk Predictor
            </p>

            <p class="app-subtitle">
                Machine learning credit-risk classification
                with cross-validation, ROC-AUC, PR-AUC,
                threshold analysis and batch prediction.
            </p>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# TOP METRICS
# =============================================================================

total_customers = len(df)

good_count = int((df[TARGET] == 0).sum())

bad_count = int((df[TARGET] == 1).sum())

bad_rate = bad_count / total_customers if total_customers else 0

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Dataset Rows", f"{total_customers:,}")

c2.metric("Features", f"{len(df.columns) - 1}")

c3.metric("Good Risk", f"{good_count:,}")

c4.metric("Bad Risk", f"{bad_count:,}")

c5.metric("Best Test Accuracy", f"{best_test_accuracy:.2%}")


# =============================================================================
# TABS
# =============================================================================

(
    tab_predict,
    tab_models,
    tab_evaluation,
    tab_batch,
    tab_data,
    tab_history,
) = st.tabs(
    [
        "👤 Prediction",
        "🏆 Model Comparison",
        "📈 Evaluation",
        "📄 Batch Prediction",
        "📊 Dataset",
        "🕘 History",
    ]
)


# =============================================================================
# SINGLE CUSTOMER PREDICTION
# =============================================================================

with tab_predict:
    st.markdown("#### Customer information")

    columns_lower = {str(c).lower(): c for c in df.columns}

    age_col = columns_lower.get("age")
    sex_col = columns_lower.get("sex")
    job_col = columns_lower.get("job")
    housing_col = columns_lower.get("housing")
    saving_col = columns_lower.get("saving accounts")
    checking_col = columns_lower.get("checking account")
    credit_col = columns_lower.get("credit amount")
    duration_col = columns_lower.get("duration")
    purpose_col = columns_lower.get("purpose")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("##### Customer Details")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
        )

        if sex_col:
            sex_options = sorted(df[sex_col].dropna().astype(str).unique().tolist())

        else:
            sex_options = ["male", "female"]

        sex = st.selectbox("Sex", sex_options)

        if job_col:
            job_values = (
                pd.to_numeric(df[job_col], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )

            job_options = sorted(job_values.tolist())

        else:
            job_options = [0, 1, 2, 3]

        job = st.selectbox("Job", job_options)

        if housing_col:
            housing_options = sorted(
                df[housing_col].dropna().astype(str).unique().tolist()
            )

        else:
            housing_options = ["own", "rent", "free"]

        housing = st.selectbox("Housing", housing_options)

        if saving_col:
            saving_options = sorted(
                df[saving_col].dropna().astype(str).unique().tolist()
            )

        else:
            saving_options = ["little", "moderate", "rich", "quite rich"]

        saving = st.selectbox("Saving accounts", saving_options)

    with right:
        st.markdown("##### Financial Details")

        if checking_col:
            checking_options = sorted(
                df[checking_col].dropna().astype(str).unique().tolist()
            )

        else:
            checking_options = ["little", "moderate", "rich"]

        checking = st.selectbox("Checking account", checking_options)

        credit_amount = st.number_input(
            "Credit amount",
            min_value=0,
            max_value=100000,
            value=3000,
            step=100,
        )

        duration = st.number_input(
            "Duration (months)",
            min_value=1,
            max_value=120,
            value=24,
        )

        if purpose_col:
            purpose_options = sorted(
                df[purpose_col].dropna().astype(str).unique().tolist()
            )

        else:
            purpose_options = ["radio/TV", "car", "furniture/equipment"]

        purpose = st.selectbox("Purpose", purpose_options)

        predict_clicked = st.button(
            f"Predict with {selected_model_name}",
            type="primary",
            width="stretch",
        )

    if predict_clicked:
        customer = {}

        if age_col:
            customer[age_col] = age

        if sex_col:
            customer[sex_col] = sex

        if job_col:
            customer[job_col] = job

        if housing_col:
            customer[housing_col] = housing

        if saving_col:
            customer[saving_col] = saving

        if checking_col:
            customer[checking_col] = checking

        if credit_col:
            customer[credit_col] = credit_amount

        if duration_col:
            customer[duration_col] = duration

        if purpose_col:
            customer[purpose_col] = purpose

        # Fill engineered/other features
        for col in X_train.columns:
            if col in customer:
                continue

            if pd.api.types.is_numeric_dtype(X_train[col]):
                median_value = pd.to_numeric(X_train[col], errors="coerce").median()

                customer[col] = 0 if pd.isna(median_value) else float(median_value)

            else:
                mode = X_train[col].dropna().mode()

                customer[col] = mode.iloc[0] if not mode.empty else ""

        customer_df = pd.DataFrame([customer], columns=X_train.columns)

        probability = get_probability(selected_model, customer_df)

        if probability is not None:
            bad_probability = float(probability[0])

            label = "Bad" if bad_probability >= threshold else "Good"

        else:
            prediction = int(selected_model.predict(customer_df)[0])

            label = "Bad" if prediction == 1 else "Good"

            bad_probability = np.nan

        result_class = "result-bad" if label == "Bad" else "result-good"

        icon = "⚠️" if label == "Bad" else "✅"

        probability_display = (
            f"{bad_probability:.1%}" if not np.isnan(bad_probability) else "N/A"
        )

        st.markdown(
            f"""
            <div class="result-card {result_class}">

                <div class="result-label">
                    {icon} {label} Credit Risk
                </div>

                <div class="result-caption">

                    Model:
                    <b>{selected_model_name}</b>

                    · Bad-risk probability:
                    <b>{probability_display}</b>

                    · Threshold:
                    <b>{threshold:.0%}</b>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if not np.isnan(bad_probability):
            st.progress(
                bad_probability, text=(f"Bad-risk probability: {bad_probability:.1%}")
            )

        st.session_state.history.append(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": selected_model_name,
                "prediction": label,
                "bad_probability": bad_probability,
                "threshold": threshold,
                "age": age,
                "sex": sex,
                "credit_amount": credit_amount,
                "duration": duration,
                "purpose": purpose,
            }
        )


# =============================================================================
# MODEL COMPARISON
# =============================================================================

with tab_models:
    st.markdown("#### Model performance comparison")

    st.caption(
        "Models are ranked using a combined score based on "
        "cross-validation F1, ROC-AUC and accuracy."
    )

    display_df = results_df.copy()

    st.dataframe(
        display_df.style.format(
            {
                "CV Accuracy": "{:.2%}",
                "CV Accuracy Std": "{:.2%}",
                "CV Precision": "{:.2%}",
                "CV Recall": "{:.2%}",
                "CV F1": "{:.2%}",
                "CV ROC AUC": "{:.2%}",
                "CV PR AUC": "{:.2%}",
                "Test Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1": "{:.2%}",
                "ROC AUC": "{:.2%}",
                "PR AUC": "{:.2%}",
                "Model Score": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.success(
        f"🏆 Best model: **{best_model_name}**\n\n"
        f"CV F1: **{best_cv_f1:.2%}** · "
        f"CV ROC AUC: **{best_cv_auc:.2%}** · "
        f"Test Accuracy: **{best_test_accuracy:.2%}**"
    )

    metric = st.selectbox(
        "Comparison metric",
        [
            "CV Accuracy",
            "CV F1",
            "CV ROC AUC",
            "CV PR AUC",
            "Test Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC AUC",
            "PR AUC",
        ],
    )

    chart_df = results_df[["Model", metric]].copy()

    chart = (
        alt.Chart(chart_df)
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
        )
        .encode(
            x=alt.X(
                "Model:N",
                sort="-y",
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y(
                f"{metric}:Q",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                "Model",
                alt.Tooltip(metric, format=".2%"),
            ],
        )
        .properties(height=400)
    )

    st.altair_chart(chart, width="stretch")


# =============================================================================
# EVALUATION
# =============================================================================

with tab_evaluation:
    st.markdown(f"#### Detailed evaluation — {selected_model_name}")

    row = results_df[results_df["Model"] == selected_model_name].iloc[0]

    e1, e2, e3, e4, e5, e6 = st.columns(6)

    e1.metric("CV F1", f"{row['CV F1']:.2%}")

    e2.metric("Test Accuracy", f"{row['Test Accuracy']:.2%}")

    e3.metric("Precision", f"{row['Precision']:.2%}")

    e4.metric("Recall", f"{row['Recall']:.2%}")

    e5.metric("F1", f"{row['F1']:.2%}")

    e6.metric("ROC AUC", f"{row['ROC AUC']:.2%}")

    st.write("")

    predictions = selected_model.predict(X_test)

    probabilities = get_probability(selected_model, X_test)

    # -------------------------------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------------------------------

    st.markdown("##### Confusion Matrix")

    cm = confusion_matrix(y_test, predictions)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Good", "Actual Bad"],
        columns=["Predicted Good", "Predicted Bad"],
    )

    st.dataframe(cm_df, width="stretch")

    tn, fp, fn, tp = cm.ravel()

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("True Negatives", f"{tn:,}")

    m2.metric("False Positives", f"{fp:,}")

    m3.metric("False Negatives", f"{fn:,}")

    m4.metric("True Positives", f"{tp:,}")

    # -------------------------------------------------------------------------
    # CLASSIFICATION REPORT
    # -------------------------------------------------------------------------

    st.markdown("##### Classification Report")

    report = classification_report(
        y_test,
        predictions,
        target_names=["Good", "Bad"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.style.format(
            {
                "precision": "{:.2%}",
                "recall": "{:.2%}",
                "f1-score": "{:.2%}",
            }
        ),
        width="stretch",
    )

    if probabilities is not None:
        # ---------------------------------------------------------------------
        # ROC CURVE
        # ---------------------------------------------------------------------

        st.markdown("##### ROC Curve")

        fpr, tpr, _ = roc_curve(y_test, probabilities)

        roc_df = pd.DataFrame(
            {
                "False Positive Rate": fpr,
                "True Positive Rate": tpr,
            }
        )

        roc_chart = (
            alt.Chart(roc_df)
            .mark_line(strokeWidth=3)
            .encode(
                x=alt.X(
                    "False Positive Rate:Q",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y(
                    "True Positive Rate:Q",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=True,
            )
            .properties(height=350)
        )

        st.altair_chart(roc_chart, width="stretch")

        st.info(f"ROC AUC: **{row['ROC AUC']:.2%}**")

        # ---------------------------------------------------------------------
        # PRECISION RECALL CURVE
        # ---------------------------------------------------------------------

        st.markdown("##### Precision-Recall Curve")

        precision_values, recall_values, _ = precision_recall_curve(
            y_test, probabilities
        )

        pr_df = pd.DataFrame(
            {
                "Recall": recall_values,
                "Precision": precision_values,
            }
        )

        pr_chart = (
            alt.Chart(pr_df)
            .mark_line(strokeWidth=3)
            .encode(
                x=alt.X(
                    "Recall:Q",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y(
                    "Precision:Q",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=True,
            )
            .properties(height=350)
        )

        st.altair_chart(pr_chart, width="stretch")

        st.info(f"PR AUC: **{row['PR AUC']:.2%}**")

        # ---------------------------------------------------------------------
        # THRESHOLD ANALYSIS
        # ---------------------------------------------------------------------

        st.markdown("##### Probability Threshold Analysis")

        best_threshold, threshold_df = threshold_analysis(
            selected_model, X_test, y_test
        )

        st.info(f"Best threshold by test-set F1: **{best_threshold:.2f}**")

        threshold_metric = st.selectbox(
            "Threshold metric",
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
            ],
            key="threshold_metric",
        )

        threshold_chart = (
            alt.Chart(threshold_df)
            .mark_line(strokeWidth=3)
            .encode(
                x=alt.X(
                    "Threshold:Q",
                    scale=alt.Scale(domain=[0.1, 0.9]),
                ),
                y=alt.Y(
                    f"{threshold_metric}:Q",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=[
                    alt.Tooltip("Threshold", format=".2f"),
                    alt.Tooltip(threshold_metric, format=".2%"),
                ],
            )
            .properties(height=350)
        )

        st.altair_chart(threshold_chart, width="stretch")

        st.caption(
            "Note: threshold selection should ideally be "
            "validated on a separate validation set rather "
            "than optimized directly on the final test set."
        )


# =============================================================================
# BATCH PREDICTION
# =============================================================================

with tab_batch:
    st.markdown("#### Upload customers for batch prediction")

    st.caption(
        "Upload a CSV containing the same feature columns "
        "used by the training dataset. The Risk column is optional."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
        key="batch_upload",
    )

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)

            # ---------------------------------------------------------------
            # Remove unnamed index columns
            # ---------------------------------------------------------------

            unnamed_columns = [
                c
                for c in batch_df.columns
                if str(c).strip().lower().startswith("unnamed")
            ]

            if unnamed_columns:
                batch_df = batch_df.drop(columns=unnamed_columns)

            # Clean column names
            batch_df.columns = [str(c).strip() for c in batch_df.columns]

            # ---------------------------------------------------------------
            # Remove target if supplied
            # ---------------------------------------------------------------

            prediction_df = batch_df.drop(columns=[TARGET], errors="ignore")

            # ---------------------------------------------------------------
            # Check required columns
            # ---------------------------------------------------------------

            missing_columns = [
                c for c in X_train.columns if c not in prediction_df.columns
            ]

            extra_columns = [
                c for c in prediction_df.columns if c not in X_train.columns
            ]

            if missing_columns:
                st.error(
                    "❌ Missing required columns:\n\n"
                    + "\n".join(f"- {c}" for c in missing_columns)
                )

            else:
                # Keep exact training order
                prediction_df = prediction_df[X_train.columns]

                if extra_columns:
                    st.info("ℹ️ Extra columns ignored: " + ", ".join(extra_columns))

                st.markdown("##### Uploaded Data")

                st.dataframe(batch_df.head(10), width="stretch")

                st.caption(f"{len(batch_df):,} rows ready for prediction.")

                if st.button(
                    f"Run Batch Prediction with {selected_model_name}",
                    type="primary",
                    width="stretch",
                    key="run_batch_prediction",
                ):
                    with st.spinner(f"Predicting {len(prediction_df):,} customers..."):
                        probabilities = get_probability(selected_model, prediction_df)

                        if probabilities is not None:
                            predictions = (probabilities >= threshold).astype(int)

                        else:
                            predictions = selected_model.predict(prediction_df)

                        output_df = batch_df.copy()

                        output_df["Predicted Risk"] = np.where(
                            predictions == 1, "Bad", "Good"
                        )

                        if probabilities is not None:
                            output_df["Bad Risk Probability"] = np.round(
                                probabilities, 6
                            )

                            output_df["Good Risk Probability"] = np.round(
                                1 - probabilities, 6
                            )

                        total = len(output_df)

                        bad = int((output_df["Predicted Risk"] == "Bad").sum())

                        good = total - bad

                        st.success(f"✅ Predicted {total:,} customers.")

                        b1, b2, b3, b4 = st.columns(4)

                        b1.metric("Total", f"{total:,}")

                        b2.metric("Good Risk", f"{good:,}")

                        b3.metric("Bad Risk", f"{bad:,}")

                        b4.metric(
                            "Bad Risk Rate", f"{bad / total:.1%}" if total else "0.0%"
                        )

                        # ---------------------------------------------------
                        # Chart
                        # ---------------------------------------------------

                        chart_data = (
                            output_df["Predicted Risk"]
                            .value_counts()
                            .rename_axis("Risk")
                            .reset_index(name="Count")
                        )

                        chart = (
                            alt.Chart(chart_data)
                            .mark_bar(
                                cornerRadiusTopLeft=6,
                                cornerRadiusTopRight=6,
                            )
                            .encode(
                                x=alt.X("Risk:N", title=None),
                                y=alt.Y("Count:Q", title="Customers"),
                                tooltip=["Risk", "Count"],
                            )
                            .properties(height=280)
                        )

                        st.altair_chart(chart, width="stretch")

                        # ---------------------------------------------------
                        # Results
                        # ---------------------------------------------------

                        st.markdown("##### Prediction Results")

                        st.dataframe(output_df, width="stretch", height=450)

                        # ---------------------------------------------------
                        # Download
                        # ---------------------------------------------------

                        csv_bytes = output_df.to_csv(index=False).encode("utf-8")

                        st.download_button(
                            "⬇️ Download Prediction Results",
                            data=csv_bytes,
                            file_name=("german_credit_predictions.csv"),
                            mime="text/csv",
                            width="stretch",
                        )

        except Exception as exc:
            st.error(f"❌ Batch prediction failed: {exc}")

            with st.expander("Technical error details"):
                st.exception(exc)


# =============================================================================
# DATASET
# =============================================================================

with tab_data:
    st.markdown("#### Dataset overview")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Rows", f"{len(df):,}")

    d2.metric("Columns", f"{len(df.columns):,}")

    d3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    d4.metric("Bad Risk", f"{bad_rate:.1%}")

    st.write("")

    st.markdown("##### Dataset Preview")

    st.dataframe(df.head(100), width="stretch", height=430)

    st.markdown("##### Target Distribution")

    target_chart = (
        df[TARGET]
        .map({0: "Good", 1: "Bad"})
        .value_counts()
        .rename_axis("Risk")
        .reset_index(name="Count")
    )

    chart = (
        alt.Chart(target_chart)
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
        )
        .encode(
            x=alt.X("Risk:N", title=None),
            y=alt.Y("Count:Q", title="Customers"),
            tooltip=["Risk", "Count"],
        )
        .properties(height=300)
    )

    st.altair_chart(chart, width="stretch")


# =============================================================================
# HISTORY
# =============================================================================

with tab_history:
    st.markdown("#### Prediction history")

    if not st.session_state.history:
        st.info("No predictions yet. Make a prediction in the Prediction tab.")

    else:
        hist_df = (
            pd.DataFrame(st.session_state.history).iloc[::-1].reset_index(drop=True)
        )

        st.dataframe(hist_df, width="stretch", height=400)

        st.markdown("##### Prediction Distribution")

        history_counts = (
            hist_df["prediction"]
            .value_counts()
            .rename_axis("Risk")
            .reset_index(name="Count")
        )

        chart = (
            alt.Chart(history_counts)
            .mark_bar(
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6,
            )
            .encode(
                x=alt.X("Risk:N", title=None),
                y=alt.Y("Count:Q", title="Predictions"),
                tooltip=["Risk", "Count"],
            )
            .properties(height=280)
        )

        st.altair_chart(chart, width="stretch")
