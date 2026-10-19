from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StayType AI | NYC Airbnb Room Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Best_Model_Pipeline.pkl"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

# Required to load custom classes serialized in the model.
sys.path.insert(0, str(BASE_DIR))


# ============================================================
# STYLING
# Design rule: never hardcode body/text colours. Text inherits from the
# active Streamlit theme and surfaces use translucent neutrals, so every
# component stays readable in both dark and light mode. Only brand and
# status accents (blue / green / amber / red) are fixed.
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --st-surface: rgba(128, 128, 128, 0.09);
        --st-surface-strong: rgba(128, 128, 128, 0.14);
        --st-border: rgba(128, 128, 128, 0.30);
        --st-muted-opacity: 0.72;
        --st-accent: #3B82F6;
        --st-accent-strong: #2563EB;
    }

    /* ---------- Layout ---------- */
    .block-container {
        padding-top: 2.2rem;
        max-width: 1280px;
    }

    /* ---------- Headings and captions (inherit theme text colour) ---------- */
    .section-title {
        font-size: 1.4rem;
        font-weight: 750;
        margin: 18px 0 6px;
        padding-left: 12px;
        border-left: 4px solid var(--st-accent);
        color: inherit;
    }

    .section-caption {
        color: inherit;
        opacity: var(--st-muted-opacity);
        margin: 0 0 18px 16px;
    }

    /* ---------- Hero ---------- */
    .hero {
        padding: 30px 32px;
        margin-bottom: 22px;
        border-radius: 18px;
        background: linear-gradient(120deg, #1E293B 0%, #1D4ED8 100%);
        border: 1px solid rgba(255, 255, 255, 0.10);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }

    .hero h1,
    .hero p,
    .hero div {
        color: #FFFFFF !important;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin: 4px 0 8px;
        padding: 0;
    }

    .hero p {
        opacity: 0.92;
        max-width: 860px;
        margin: 0;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.75rem;
        font-weight: 700;
        opacity: 0.85;
    }

    /* ---------- Streamlit components ---------- */
    div[data-testid="stMetric"],
    div[data-testid="stMetricContainer"] {
        background: var(--st-surface);
        border: 1px solid var(--st-border);
        border-radius: 12px;
        padding: 14px 16px;
    }

    div[data-testid="stMetricLabel"] {
        opacity: var(--st-muted-opacity);
    }

    div[data-testid="stForm"] {
        background: var(--st-surface);
        border: 1px solid var(--st-border);
        border-radius: 16px;
        padding: 22px 24px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--st-border);
        border-radius: 10px;
        overflow: hidden;
    }

    button[data-baseweb="tab"] p {
        font-weight: 600;
        font-size: 1rem;
    }

    /* Primary buttons: solid brand blue with white text (readable in any theme) */
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-primaryFormSubmit"] {
        background-color: var(--st-accent-strong) !important;
        border: 1px solid var(--st-accent-strong) !important;
        color: #FFFFFF !important;
        font-weight: 650;
        border-radius: 10px;
    }

    button[kind="primary"] *,
    button[data-testid="stBaseButton-primary"] *,
    button[data-testid="stBaseButton-primaryFormSubmit"] * {
        color: #FFFFFF !important;
    }

    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }

    /* ---------- Result card ---------- */
    .result-card {
        border-radius: 16px;
        padding: 22px 24px;
        margin: 8px 0 16px;
        border: 1px solid var(--st-border);
        background: var(--st-surface);
        color: inherit;
    }

    .prediction-high {
        border-left: 6px solid #16A34A;
        background: linear-gradient(90deg, rgba(22, 163, 74, 0.14), var(--st-surface) 65%);
    }

    .prediction-medium {
        border-left: 6px solid #D97706;
        background: linear-gradient(90deg, rgba(217, 119, 6, 0.16), var(--st-surface) 65%);
    }

    .prediction-low {
        border-left: 6px solid #DC2626;
        background: linear-gradient(90deg, rgba(220, 38, 38, 0.16), var(--st-surface) 65%);
    }

    .prediction-title {
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0 0 10px;
        color: inherit;
    }

    .prediction-description {
        color: inherit;
        opacity: 0.85;
        margin: 6px 0;
    }

    /* ---------- Insight cards ---------- */
    .insight-card {
        background: var(--st-surface);
        border: 1px solid var(--st-border);
        border-radius: 12px;
        padding: 16px 18px;
        margin: 10px 0;
    }

    .insight-label {
        color: inherit;
        opacity: var(--st-muted-opacity);
        font-size: 0.85rem;
    }

    .insight-value {
        color: inherit;
        font-size: 1.6rem;
        font-weight: 800;
    }

    .insight-detail {
        color: inherit;
        opacity: var(--st-muted-opacity);
        font-size: 0.85rem;
    }

    /* ---------- Pipeline flow ---------- */
    .pipeline-row {
        display: flex;
        flex-wrap: wrap;
        align-items: stretch;
        gap: 8px;
        margin: 20px 0;
    }

    .pipeline-stage {
        flex: 1 1 135px;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid var(--st-border);
        background: var(--st-surface);
        color: inherit;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    .pipeline-stage b {
        display: block;
        margin-bottom: 7px;
        color: inherit;
        font-size: 0.92rem;
    }

    .pipeline-stage.active {
        border-color: var(--st-accent);
        background: rgba(59, 130, 246, 0.16);
    }

    .pipeline-arrow {
        align-self: center;
        color: inherit;
        opacity: 0.5;
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PIPELINE INFORMATION
# ============================================================

PIPELINE_STAGES = [
    ("1. Raw data", "Load and inspect NYC Airbnb data"),
    ("2. Clean and split", "Remove identifier columns and split data"),
    ("3. Engineer", "Create logs, ratios, and binary indicators"),
    ("4. Encode", "Impute missing values and encode categories"),
    ("5. Tune", "Compare candidate models using cross-validation"),
    ("6. Calibrate", "Calibrate class probability estimates"),
    ("7. Decision policy", "Apply tuned class decision weights"),
    ("8. Evaluate", "Evaluate against held-out test data"),
]


# ============================================================
# CACHED RESOURCES
# ============================================================


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# HELPERS
# ============================================================


def model_display_name(model_type: str | None) -> str:
    names = {
        "random_forest": "Random Forest",
        "hist_gradient_boosting": "Histogram Gradient Boosting",
    }
    return names.get(model_type or "", model_type or "Unknown")


def get_prediction_input(
    neighbourhood_group: str,
    neighbourhood: str,
    latitude: float,
    longitude: float,
    price: float,
    minimum_nights: int,
    number_of_reviews: int,
    reviews_per_month: float,
    calculated_host_listings_count: int,
    availability_365: int,
) -> pd.DataFrame:
    """Build a one-row DataFrame using the model's expected raw inputs."""

    return pd.DataFrame(
        [
            {
                "neighbourhood_group": neighbourhood_group,
                "neighbourhood": neighbourhood.strip() or "Unknown",
                "latitude": latitude,
                "longitude": longitude,
                "price": price,
                "minimum_nights": minimum_nights,
                "number_of_reviews": number_of_reviews,
                "reviews_per_month": reviews_per_month,
                "calculated_host_listings_count": calculated_host_listings_count,
                "availability_365": availability_365,
            }
        ]
    )


def render_pipeline_flow() -> None:
    parts = []

    for index, (title, description) in enumerate(PIPELINE_STAGES):
        active = " active" if index in {4, 5, 6, 7} else ""
        parts.append(
            f'<div class="pipeline-stage{active}"><b>{title}</b>{description}</div>'
        )

        if index < len(PIPELINE_STAGES) - 1:
            parts.append('<div class="pipeline-arrow">→</div>')

    st.markdown(
        f'<div class="pipeline-row">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_insight_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_label(probability: float) -> tuple[str, str, str]:
    """
    UI guidance based on the probability of the selected class.
    These thresholds are heuristic, not validated accuracy guarantees.
    """
    if probability >= 0.80:
        return (
            "High selected-class probability",
            "prediction-high",
            "The selected class has a high estimated probability. "
            "This does not guarantee the prediction is correct.",
        )

    if probability >= 0.55:
        return (
            "Moderate selected-class probability",
            "prediction-medium",
            "The selected class has a moderate estimated probability. "
            "Review alternative classes and listing details.",
        )

    return (
        "Low selected-class probability",
        "prediction-low",
        "The selected class has a low estimated probability. "
        "Manual review is recommended.",
    )


def validate_probabilities(
    classes: list,
    probabilities,
) -> list[float]:
    """Check that model output is a valid multiclass probability vector."""

    values = [float(value) for value in probabilities]

    if len(classes) != len(values):
        raise ValueError("Class labels and probability outputs have different lengths.")

    if not values:
        raise ValueError("The model returned no class probabilities.")

    if not all(pd.notna(value) for value in values):
        raise ValueError("The model returned missing probabilities.")

    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("A model probability falls outside [0, 1].")

    if abs(sum(values) - 1.0) > 1e-5:
        raise ValueError("The class probabilities do not sum to 1.")

    return values


# ============================================================
# FILE CHECKS AND MODEL LOADING
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found.\n\n"
        "Expected file:\n"
        f"`{MODEL_PATH}`\n\n"
        "Check that your training run has generated the saved model."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(
        "The model could not be loaded. Check that `nyc_features.py` "
        "is present and compatible with the saved model artifact."
    )
    st.exception(exc)
    st.stop()

try:
    metrics = load_metrics()
except (OSError, json.JSONDecodeError) as exc:
    st.warning(f"The metrics report could not be read: {exc}")
    metrics = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning Decision Support</div>
        <h1>🏠 StayType AI</h1>
        <p>
            Predict whether an NYC Airbnb listing most closely matches an
            Entire home/apt, Private room, or Shared room profile using a
            trained and calibrated multiclass classification pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# KPI ROW
# ============================================================

best_parameters = metrics.get("best_parameters", {}) if metrics else {}
best_model_name = model_display_name(best_parameters.get("model_type"))

test_accuracy = float(metrics.get("test_accuracy", 0)) if metrics else 0.0
test_macro_f1 = float(metrics.get("test_macro_f1", 0)) if metrics else 0.0
best_cv_macro_f1 = float(metrics.get("best_cv_macro_f1", 0)) if metrics else 0.0
mean_brier_score = float(metrics.get("mean_brier_score", 0)) if metrics else 0.0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Test Accuracy", f"{test_accuracy:.2%}" if metrics else "N/A")

with kpi2:
    st.metric("Test Macro-F1", f"{test_macro_f1:.4f}" if metrics else "N/A")

with kpi3:
    st.metric("Selected Model", best_model_name if metrics else "See model artifact")

with kpi4:
    st.metric(
        "Mean Brier Score",
        f"{mean_brier_score:.4f}" if metrics else "N/A",
        help="Lower is generally better; interpret with class prevalence and evaluation setup.",
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🏠 StayType AI")
    st.caption("NYC Airbnb room-type classification")

    st.divider()
    st.markdown("#### Model status")

    st.success("Model artifact loaded")

    if metrics:
        st.caption(f"Best CV Macro-F1: {best_cv_macro_f1:.4f}")
        st.caption(f"Test Macro-F1: {test_macro_f1:.4f}")
    else:
        st.warning("Metrics report unavailable")

    st.divider()
    st.markdown("#### Responsible use")
    st.caption(
        "Predictions estimate historical listing patterns. They are not "
        "verified Airbnb listing attributes and should not replace human review."
    )


# ============================================================
# TABS
# ============================================================

predict_tab, performance_tab, pipeline_tab = st.tabs(
    [
        "🔮 Predict Room Type",
        "📊 Model Performance",
        "🧠 Pipeline Intelligence",
    ]
)


# ============================================================
# PREDICTION TAB
# ============================================================

with predict_tab:
    st.markdown(
        '<div class="section-title">Listing classification</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Enter listing details and submit the '
        "form to estimate the most likely room category.</div>",
        unsafe_allow_html=True,
    )

    with st.form("listing_prediction_form", clear_on_submit=False):
        st.markdown("#### Location and pricing")

        location_col, coordinates_col, pricing_col = st.columns(3)

        with location_col:
            neighbourhood_group = st.selectbox(
                "Neighbourhood group",
                [
                    "Manhattan",
                    "Brooklyn",
                    "Queens",
                    "Bronx",
                    "Staten Island",
                ],
            )

            neighbourhood = st.text_input(
                "Neighbourhood",
                value="Harlem",
                help="Examples: Harlem, Williamsburg, Astoria, Chelsea.",
            )

        with coordinates_col:
            latitude = st.number_input(
                "Latitude",
                min_value=40.0,
                max_value=41.0,
                value=40.8116,
                step=0.0001,
                format="%.4f",
            )

            longitude = st.number_input(
                "Longitude",
                min_value=-74.5,
                max_value=-73.0,
                value=-73.9465,
                step=0.0001,
                format="%.4f",
            )

        with pricing_col:
            price = st.number_input(
                "Price per night ($)",
                min_value=0.0,
                value=150.0,
                step=5.0,
            )

            minimum_nights = st.number_input(
                "Minimum nights",
                min_value=1,
                value=2,
                step=1,
            )

        st.markdown("#### Listing activity and host profile")

        activity_col, host_col, availability_col = st.columns(3)

        with activity_col:
            number_of_reviews = st.number_input(
                "Number of reviews",
                min_value=0,
                value=10,
                step=1,
            )

            reviews_per_month = st.number_input(
                "Reviews per month",
                min_value=0.0,
                value=1.0,
                step=0.1,
            )

        with host_col:
            calculated_host_listings_count = st.number_input(
                "Host listing count",
                min_value=1,
                value=1,
                step=1,
            )

        with availability_col:
            availability_365 = st.slider(
                "Availability over next 365 days",
                min_value=0,
                max_value=365,
                value=180,
            )

        submitted = st.form_submit_button(
            "🔮 Predict Room Type",
            type="primary",
            width="stretch",
        )

    if submitted:
        input_df = get_prediction_input(
            neighbourhood_group=neighbourhood_group,
            neighbourhood=neighbourhood,
            latitude=latitude,
            longitude=longitude,
            price=price,
            minimum_nights=minimum_nights,
            number_of_reviews=number_of_reviews,
            reviews_per_month=reviews_per_month,
            calculated_host_listings_count=calculated_host_listings_count,
            availability_365=availability_365,
        )

        if price == 0:
            st.warning(
                "The price is set to $0. This may be unusual; verify the "
                "value if it does not reflect the intended listing."
            )

        try:
            with st.spinner("Analyzing listing and generating prediction..."):
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)

            probabilities = validate_probabilities(classes, probabilities)

        except Exception as exc:
            st.error(
                "Prediction failed. Verify the inputs and confirm that the "
                "saved model is compatible with the current feature code."
            )
            st.exception(exc)
            st.stop()

        probability_map = {
            str(room_type): probability
            for room_type, probability in zip(classes, probabilities)
        }

        if str(prediction) not in probability_map:
            st.error("The predicted class was not found in the model classes.")
            st.stop()

        top_probability = max(probability_map.values())
        selected_class_probability = probability_map[str(prediction)]

        confidence, confidence_class, confidence_message = confidence_label(
            selected_class_probability
        )

        probability_df = (
            pd.DataFrame(
                {
                    "Room Type": list(probability_map.keys()),
                    "Probability": list(probability_map.values()),
                }
            )
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )

        prediction_descriptions = {
            "Entire home/apt": (
                "The listing profile most closely matches an entire "
                "property or apartment rental."
            ),
            "Private room": (
                "The listing profile most closely matches a private-room rental."
            ),
            "Shared room": (
                "The listing profile most closely matches a shared-room rental."
            ),
        }

        prediction_text = prediction_descriptions.get(
            str(prediction), "The model selected this room category."
        )

        st.divider()
        result_col, probability_col = st.columns([1.15, 1])

        with result_col:
            st.markdown("### Prediction result")

            st.markdown(
                f"""
                <div class="result-card {confidence_class}">
                    <p class="prediction-title">🏠 {prediction}</p>
                    <p class="prediction-description">{prediction_text}</p>
                    <p class="prediction-description">
                        <b>{confidence}:</b> {confidence_message}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "Highest class probability",
                    f"{top_probability:.2%}",
                    help="The largest probability across all classes. "
                    "It may belong to a different class from the weighted prediction.",
                )

                st.metric(
                    "Selected class probability",
                    f"{selected_class_probability:.2%}",
                    help="The calibrated probability assigned to the class "
                    "selected by the model's decision rule.",
                )

            with metric2:
                st.metric("Predicted room type", str(prediction))

            with metric3:
                st.metric("Price per night", f"${price:,.0f}")

            st.progress(
                min(max(selected_class_probability, 0.0), 1.0),
                text="Selected-class probability (not a guarantee of correctness)",
            )

            if selected_class_probability < 0.55:
                st.warning(
                    "The selected class has a low estimated probability. "
                    "Review the alternative classes and verify the listing manually."
                )
            else:
                st.info(
                    "This is a model estimate based on historical listing "
                    "patterns, not verified Airbnb listing data."
                )

            if abs(top_probability - selected_class_probability) > 0.05:
                st.caption(
                    "Decision-policy note: the class selected by the weighted "
                    "decision rule differs in probability from the highest-probability "
                    "class. Review the probability table below."
                )

        with probability_col:
            st.markdown("### Class probabilities")

            st.dataframe(
                probability_df.style.format({"Probability": "{:.2%}"}),
                width="stretch",
                hide_index=True,
            )

            probability_chart = px.bar(
                probability_df,
                x="Probability",
                y="Room Type",
                orientation="h",
                text="Probability",
                color="Probability",
                color_continuous_scale="Blues",
                range_color=[0, 1],
            )

            probability_chart.update_traces(
                texttemplate="%{text:.1%}",
                textposition="outside",
            )

            probability_chart.update_layout(
                height=300,
                margin=dict(l=10, r=35, t=10, b=10),
                coloraxis_showscale=False,
                xaxis=dict(tickformat=".0%", range=[0, 1]),
            )

            st.plotly_chart(probability_chart, width="stretch")

        st.markdown(
            '<div class="section-title">Listing profile used for prediction</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(input_df, width="stretch", hide_index=True)

        history_item = {
            "Timestamp (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Prediction": str(prediction),
            "Selected class probability": selected_class_probability,
            "Highest class probability": top_probability,
            "Neighbourhood Group": neighbourhood_group,
            "Neighbourhood": neighbourhood,
            "Price": price,
        }

        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []

        st.session_state.prediction_history.insert(0, history_item)

        # Limit memory usage during long sessions.
        st.session_state.prediction_history = st.session_state.prediction_history[:50]

    else:
        st.info(
            "Enter listing details and click **Predict Room Type** to see "
            "the predicted class and probability distribution."
        )

    if st.session_state.get("prediction_history"):
        st.markdown(
            '<div class="section-title">Session prediction history</div>',
            unsafe_allow_html=True,
        )

        history_df = pd.DataFrame(st.session_state.prediction_history)

        for column in [
            "Highest class probability",
            "Selected class probability",
        ]:
            if column in history_df.columns:
                history_df[column] = history_df[column].map(
                    lambda value: f"{value:.2%}"
                )

        st.dataframe(history_df, width="stretch", hide_index=True)


# ============================================================
# MODEL PERFORMANCE TAB
# ============================================================

with performance_tab:
    st.markdown(
        '<div class="section-title">Model performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">Metrics loaded from '
        "<code>reports/metrics.json</code>.</div>",
        unsafe_allow_html=True,
    )

    if metrics is None:
        st.warning(
            "Metrics report not found. The prediction app can still load, "
            "but the performance dashboard needs reports/metrics.json."
        )

    else:
        perf1, perf2, perf3, perf4 = st.columns(4)

        with perf1:
            st.metric("Test Accuracy", f"{test_accuracy:.2%}")

        with perf2:
            st.metric("Test Macro-F1", f"{test_macro_f1:.4f}")

        with perf3:
            st.metric("Best CV Macro-F1", f"{best_cv_macro_f1:.4f}")

        with perf4:
            st.metric("Mean Brier Score", f"{mean_brier_score:.4f}")

        per_class_f1 = metrics.get("per_class_f1", {})

        left, right = st.columns([1.1, 1])

        with left:
            st.markdown("### Per-class F1 score")

            if per_class_f1:
                f1_df = pd.DataFrame(
                    {
                        "Room Type": list(per_class_f1.keys()),
                        "F1 Score": list(per_class_f1.values()),
                    }
                )

                fig = px.bar(
                    f1_df,
                    x="Room Type",
                    y="F1 Score",
                    text="F1 Score",
                    color="F1 Score",
                    color_continuous_scale="Blues",
                    range_color=[0, 1],
                )

                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                )

                fig.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False,
                    yaxis=dict(range=[0, 1]),
                )

                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Per-class F1 metrics are unavailable.")

        with right:
            st.markdown("### Evaluation interpretation")

            shared_room_f1 = float(per_class_f1.get("Shared room", 0))

            render_insight_card(
                "Test Macro-F1",
                f"{test_macro_f1:.4f}",
                "Macro-F1 gives each class equal weight in the aggregate score.",
            )

            render_insight_card(
                "Shared-room F1",
                f"{shared_room_f1:.4f}",
                "Shared room is the rarest class and remains an important "
                "improvement opportunity.",
            )

        st.markdown("### Calibration quality")

        brier_scores = metrics.get("brier_scores_by_class", {})

        if brier_scores:
            brier_df = pd.DataFrame(
                {
                    "Room Type": list(brier_scores.keys()),
                    "Brier Score": list(brier_scores.values()),
                }
            )

            calibration_col, table_col = st.columns([1.1, 1])

            with calibration_col:
                fig = px.bar(
                    brier_df,
                    x="Room Type",
                    y="Brier Score",
                    text="Brier Score",
                    color="Brier Score",
                    color_continuous_scale="RdYlGn_r",
                )

                fig.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                )

                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False,
                )

                st.plotly_chart(fig, width="stretch")

            with table_col:
                st.markdown(
                    """
                    **What is a Brier score?**

                    It measures the difference between predicted probabilities
                    and observed outcomes. Lower scores are generally better
                    under the same evaluation setup.

                    Class-specific scores can be affected by class prevalence,
                    so do not interpret their differences as a direct ranking
                    of class performance.
                    """
                )

                st.dataframe(
                    brier_df.style.format({"Brier Score": "{:.4f}"}),
                    width="stretch",
                    hide_index=True,
                )

        st.markdown("### Decision policy")

        decision_weights = metrics.get("decision_weights", {})
        weight_col, params_col = st.columns([1, 1.2])

        with weight_col:
            if decision_weights:
                weights_df = pd.DataFrame(
                    {
                        "Room Type": list(decision_weights.keys()),
                        "Decision Weight": list(decision_weights.values()),
                    }
                )

                st.dataframe(
                    weights_df.style.format({"Decision Weight": "{:.2f}"}),
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "Decision weights can change the predicted class without "
                    "changing the underlying probability estimates."
                )
            else:
                st.info("Decision weights are unavailable in the metrics report.")

        with params_col:
            st.markdown("#### Best model configuration")

            if best_parameters:
                st.json(best_parameters)
            else:
                st.info("Best hyperparameters are unavailable.")


# ============================================================
# PIPELINE INTELLIGENCE TAB
# ============================================================

with pipeline_tab:
    st.markdown(
        '<div class="section-title">Training pipeline intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">Overview of the training and inference '
        "design documented for this project.</div>",
        unsafe_allow_html=True,
    )

    render_pipeline_flow()

    architecture_col, feature_col = st.columns(2)

    with architecture_col:
        st.markdown("### Pipeline design")

        st.markdown(
            """
            - **Model selection:** Candidate models are compared during training.
            - **Preprocessing:** Numeric and categorical fields are prepared
              by the fitted pipeline.
            - **Feature engineering:** Derived features are calculated from
              listing information and learned training statistics.
            - **Calibration:** The fitted model produces calibrated probability
              estimates.
            - **Evaluation:** Reported test metrics come from the saved
              training report.
            """
        )

    with feature_col:
        st.markdown("### Decision intelligence")

        st.markdown(
            """
            - **Class probabilities:** Review the probability assigned to each
              room category.
            - **Decision weights:** The final class may differ from the class
              with the largest raw probability.
            - **Rare class:** Shared rooms are underrepresented in the dataset.
            - **Responsible output:** Predictions are estimates, not verified
              listing attributes.
            """
        )

    st.markdown("### Input features")

    feature_table = pd.DataFrame(
        {
            "Feature": [
                "neighbourhood_group",
                "neighbourhood",
                "latitude",
                "longitude",
                "price",
                "minimum_nights",
                "number_of_reviews",
                "reviews_per_month",
                "calculated_host_listings_count",
                "availability_365",
            ],
            "Purpose": [
                "Borough context",
                "Local neighbourhood context",
                "Geographic coordinate",
                "Geographic coordinate",
                "Nightly price",
                "Minimum stay requirement",
                "Historical review volume",
                "Review activity rate",
                "Host portfolio size",
                "Availability over the next year",
            ],
        }
    )

    st.dataframe(feature_table, width="stretch", hide_index=True)

    st.warning(
        "Use this model as decision support only. Verify listing information "
        "before relying on a prediction for business, housing, or regulatory "
        "decisions."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "StayType AI · NYC Airbnb Room Type Classification · "
    "Calibrated multiclass machine-learning pipeline"
)
