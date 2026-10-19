"""
Spam Classifier — Streamlit UI
Loads a model bundle saved as: {"model": ..., "tfidf": ..., "model_name": ...}
"""

import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "spam_classifier.pkl"

# ------------------------------------------------------------------
# GLOBAL STYLE
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        #MainMenu, footer, header {visibility: hidden;}

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

        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #94a3b8;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 2px;
        }

        .result-card {
            border-radius: 16px;
            padding: 28px 30px;
            margin-top: 10px;
            border: 1px solid;
        }
        .result-spam {
            background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
            border-color: #fecaca;
        }
        .result-ham {
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
            border-color: #bbf7d0;
        }
        .result-label {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.01em;
        }
        .result-spam .result-label { color: #b91c1c; }
        .result-ham .result-label { color: #15803d; }

        .result-caption {
            color: #475569;
            font-size: 0.9rem;
            margin-top: 4px;
        }

        .pill {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .pill-spam { background: #fee2e2; color: #b91c1c; }
        .pill-ham { background: #dcfce7; color: #15803d; }

        section[data-testid="stSidebar"] {
            background-color: #0f172a;
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] .stMarkdown hr {
            border-color: #334155;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 12px;
            border: 1px solid #cbd5e1;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1.4rem;
            border: none;
        }
        .stButton > button[kind="primary"] {
            background-color: #4f46e5;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #4338ca;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 18px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        package = pickle.load(f)
    return package


def predict_messages(package, messages: list[str]) -> pd.DataFrame:
    model = package["model"]
    tfidf = package["tfidf"]

    vectors = tfidf.transform(messages)
    preds = model.predict(vectors)

    if hasattr(model, "predict_proba"):
        confidences = model.predict_proba(vectors)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(vectors)
        confidences = 1 / (1 + np.exp(-raw))  # sigmoid squash for display only
    else:
        confidences = np.full(len(messages), np.nan)

    return pd.DataFrame(
        {
            "message": messages,
            "prediction": np.where(preds == 1, "Spam", "Ham"),
            "spam_probability": confidences,
        }
    )


package = load_model(MODEL_PATH)

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: message, prediction, confidence, time

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Spam Classifier")
    st.caption("Email / SMS text classification")
    st.divider()

    if package is not None:
        st.success("Model loaded", icon="✅")
        st.markdown(f"**Model:** {package.get('model_name', 'Unknown')}")
        vocab_size = getattr(package.get("tfidf"), "vocabulary_", {})
        st.markdown(f"**Vocabulary size:** {len(vocab_size):,}")
    else:
        st.error("Model file not found", icon="⚠️")
        st.caption(f"Expected `{MODEL_PATH}` in the working directory.")

    st.divider()
    threshold = st.slider(
        "Spam decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Messages with spam probability above this value are flagged as Spam.",
    )

    st.divider()
    if st.session_state.history:
        st.markdown(f"**Predictions this session:** {len(st.session_state.history)}")
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.caption("Built with Streamlit · scikit-learn · TF-IDF")

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div style="font-size:2.1rem;">🛡️</div>
        <div>
            <p class="app-title">Spam Classifier</p>
            <p class="app-subtitle">Classify emails or SMS messages as Spam or Ham using a trained TF-IDF + ML pipeline.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if package is None:
    st.warning(
        "No trained model found. Run your training script first so it saves "
        f"`{MODEL_PATH}` next to this app, then refresh the page.",
        icon="🧩",
    )
    st.stop()

# ------------------------------------------------------------------
# TOP METRICS
# ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_preds = len(st.session_state.history)
spam_count = sum(1 for h in st.session_state.history if h["prediction"] == "Spam")
ham_count = total_preds - spam_count
spam_rate = f"{(spam_count / total_preds * 100):.0f}%" if total_preds else "—"

for col, label, value in zip(
    [col1, col2, col3, col4],
    ["Active Model", "Session Predictions", "Flagged Spam", "Spam Rate"],
    [package.get("model_name", "—"), total_preds, spam_count, spam_rate],
):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_single, tab_batch, tab_history = st.tabs(
    ["✉️ Single Message", "📄 Batch (CSV)", "📊 History"]
)

# --- TAB 1: SINGLE MESSAGE -----------------------------------------
with tab_single:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("#### Enter a message")
        sample_choice = st.selectbox(
            "Try a sample message",
            [
                "— Choose a sample —",
                "Congratulations! You have won a $500 Amazon gift card. Click here to claim your prize now.",
                "Hi Rahul, are we still meeting for lunch at 1 PM tomorrow?",
                "URGENT! Your account has won a $1,000 cash reward. Claim it before the offer expires.",
                "Hi team, please find the meeting agenda attached. We will discuss the project updates at 3 PM.",
            ],
        )
        default_text = "" if sample_choice.startswith("—") else sample_choice

        user_input = st.text_area(
            "Message text",
            value=default_text,
            height=180,
            placeholder="Paste an email or SMS message here...",
            label_visibility="collapsed",
        )

        predict_clicked = st.button(
            "Classify Message", type="primary", use_container_width=True
        )

    with right:
        st.markdown("#### Result")
        if predict_clicked:
            if not user_input.strip():
                st.info("Enter a message on the left, then click Classify.")
            else:
                result_df = predict_messages(package, [user_input])
                prob = result_df.loc[0, "spam_probability"]
                label = "Spam" if prob >= threshold else "Ham"

                st.session_state.history.append(
                    {
                        "message": user_input,
                        "prediction": label,
                        "spam_probability": float(prob) if not np.isnan(prob) else None,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                css_class = "result-spam" if label == "Spam" else "result-ham"
                icon = "🚫" if label == "Spam" else "✅"
                prob_display = f"{prob * 100:.1f}%" if not np.isnan(prob) else "N/A"

                st.markdown(
                    f"""
                    <div class="result-card {css_class}">
                        <div class="result-label">{icon} {label}</div>
                        <div class="result-caption">Spam probability: <b>{prob_display}</b> · Threshold: {threshold:.0%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if not np.isnan(prob):
                    st.write("")
                    st.progress(
                        min(max(prob, 0.0), 1.0),
                        text=f"Spam confidence — {prob * 100:.1f}%",
                    )
        else:
            st.markdown(
                """
                <div class="result-card" style="border-color:#e2e8f0; background:#f8fafc;">
                    <div style="color:#64748b; font-weight:600;">No message classified yet</div>
                    <div class="result-caption">Results will appear here after you click Classify.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# --- TAB 2: BATCH CSV -----------------------------------------------
with tab_batch:
    st.markdown("#### Upload a CSV of messages")
    st.caption(
        "File must contain a column named **Message** (or select the column below)."
    )

    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"], label_visibility="collapsed"
    )

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            batch_df = None

        if batch_df is not None:
            default_col = (
                "Message" if "Message" in batch_df.columns else batch_df.columns[0]
            )
            text_col = st.selectbox(
                "Text column",
                batch_df.columns,
                index=list(batch_df.columns).index(default_col),
            )

            if st.button("Run batch classification", type="primary"):
                messages = batch_df[text_col].astype(str).tolist()
                with st.spinner(f"Classifying {len(messages)} messages..."):
                    preds_df = predict_messages(package, messages)
                    preds_df["prediction"] = np.where(
                        preds_df["spam_probability"] >= threshold, "Spam", "Ham"
                    )

                out_df = pd.concat(
                    [
                        batch_df.reset_index(drop=True),
                        preds_df[["prediction", "spam_probability"]],
                    ],
                    axis=1,
                )

                st.success(f"Classified {len(out_df)} messages.", icon="✅")

                m1, m2, m3 = st.columns(3)
                m1.metric("Total", len(out_df))
                m2.metric("Spam", int((out_df["prediction"] == "Spam").sum()))
                m3.metric("Ham", int((out_df["prediction"] == "Ham").sum()))

                chart_data = out_df["prediction"].value_counts().reset_index()
                chart_data.columns = ["prediction", "count"]
                chart = (
                    alt.Chart(chart_data)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("prediction:N", title=None),
                        y=alt.Y("count:Q", title="Messages"),
                        color=alt.Color(
                            "prediction:N",
                            scale=alt.Scale(
                                domain=["Ham", "Spam"], range=["#22c55e", "#ef4444"]
                            ),
                            legend=None,
                        ),
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart, use_container_width=True)

                st.dataframe(out_df, use_container_width=True, height=350)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_bytes,
                    file_name="spam_classification_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.markdown(
            """
            <div class="result-card" style="border-color:#e2e8f0; background:#f8fafc;">
                <div style="color:#64748b; font-weight:600;">No file uploaded</div>
                <div class="result-caption">Upload a CSV with a text column to classify many messages at once.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- TAB 3: HISTORY --------------------------------------------------
with tab_history:
    st.markdown("#### Session prediction history")
    if not st.session_state.history:
        st.info(
            "No predictions yet. Classify a message in the first tab to see it here."
        )
    else:
        hist_df = (
            pd.DataFrame(st.session_state.history).iloc[::-1].reset_index(drop=True)
        )

        def _pill(row):
            cls = "pill-spam" if row == "Spam" else "pill-ham"
            return f'<span class="pill {cls}">{row}</span>'

        display_df = hist_df.copy()
        display_df["spam_probability"] = display_df["spam_probability"].apply(
            lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "N/A"
        )
        st.dataframe(
            display_df[["time", "message", "prediction", "spam_probability"]],
            use_container_width=True,
            height=400,
        )

        trend = hist_df.copy()
        trend["idx"] = range(len(trend), 0, -1)
        chart = (
            alt.Chart(trend)
            .mark_circle(size=90)
            .encode(
                x=alt.X(
                    "idx:O",
                    title="Prediction order (oldest → newest)",
                    sort="descending",
                ),
                y=alt.Y(
                    "spam_probability:Q",
                    title="Spam probability",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color(
                    "prediction:N",
                    scale=alt.Scale(
                        domain=["Ham", "Spam"], range=["#22c55e", "#ef4444"]
                    ),
                    legend=alt.Legend(title=None),
                ),
                tooltip=["message", "prediction", "spam_probability"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
