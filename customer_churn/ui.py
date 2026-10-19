#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------- Settings ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"
DATA_PATH = BASE_DIR / "customer_churn_data.csv"
METRICS_PATH = BASE_DIR / "model_metrics.csv"

LOW, HIGH = 0.30, 0.60

PLOT_BG = "#10141B"
GRID_COLOR = "#232A36"
TEXT_COLOR = "#E8EDF4"
MUTED = "#8B95A5"
ACCENT = "#3B82F6"

LOW_C = "#22C55E"
MID_C = "#EAB308"
HIGH_C = "#EF4444"
CHURN_COLORS = {"No": LOW_C, "Yes": HIGH_C}

st.set_page_config(
    page_title="Churn Risk Intelligence",
    page_icon="📈",
    layout="wide",
)


# ---------- CSS ----------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, label, button {{
        font-family: 'Inter', sans-serif !important;
    }}

    .stApp {{
        background: #0A0C10;
        color: {TEXT_COLOR};
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }}

    div[data-baseweb="input"], div[data-baseweb="select"] {{
        background: {PLOT_BG};
        border: 1px solid {GRID_COLOR};
        border-radius: 8px;
    }}

    input {{
        color: {TEXT_COLOR} !important;
        background: transparent !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {PLOT_BG};
        border: 1px solid {GRID_COLOR};
        border-radius: 14px;
    }}

    .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 1.15rem 1.5rem;
        background: {PLOT_BG};
        border: 1px solid {GRID_COLOR};
        border-radius: 14px;
        margin-bottom: 1.4rem;
    }}

    .topbar-left {{
        display: flex;
        align-items: center;
        gap: 0.9rem;
        min-width: 0;
    }}

    .logo {{
        width: 42px;
        height: 42px;
        flex: 0 0 42px;
        display: grid;
        place-items: center;
        border-radius: 10px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.28);
        color: {ACCENT};
        font-size: 1.15rem;
    }}

    .topbar h1 {{
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
    }}

    .topbar p {{
        margin: 0.15rem 0 0 0;
        color: {MUTED};
        font-size: 0.88rem;
    }}

    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.75rem;
        border: 1px solid {GRID_COLOR};
        border-radius: 999px;
        background: #161B24;
        color: {MUTED};
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
    }}

    .badge i {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: {ACCENT};
        display: inline-block;
    }}

    .panel-title {{
        font-size: 1rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0 0 0.2rem 0;
    }}

    .panel-subtitle {{
        color: {MUTED};
        font-size: 0.86rem;
        margin: 0 0 1rem 0;
        line-height: 1.5;
    }}

    .field-label {{
        color: {MUTED};
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }}

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1.4rem;
    }}

    .kpi {{
        min-width: 0;
        background: {PLOT_BG};
        border: 1px solid {GRID_COLOR};
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
    }}

    .kpi-label {{
        color: {MUTED};
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }}

    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.12;
        color: #FFFFFF;
        font-variant-numeric: tabular-nums;
        overflow-wrap: anywhere;
    }}

    .kpi-sub {{
        color: {MUTED};
        font-size: 0.82rem;
        margin-top: 0.4rem;
        line-height: 1.45;
    }}

    .risk-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid;
        margin-bottom: 0.25rem;
    }}

    .recommendation {{
        margin-top: 0.75rem;
        padding: 1rem 1.1rem;
        border: 1px solid {GRID_COLOR};
        border-left: 3px solid {ACCENT};
        border-radius: 10px;
        background: #161B24;
        font-size: 0.93rem;
        line-height: 1.55;
    }}

    .footnote {{
        color: {MUTED};
        font-size: 0.8rem;
        margin-top: 1.4rem;
        line-height: 1.6;
    }}

    [data-testid="stFormSubmitButton"] button {{
        width: 100%;
        background: {ACCENT};
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }}

    [data-testid="stFormSubmitButton"] button * {{
        color: #FFFFFF !important;
    }}

    @media (max-width: 1100px) {{
        .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .topbar {{ align-items: flex-start; }}
    }}

    @media (max-width: 700px) {{
        .kpi-grid {{ grid-template-columns: 1fr; }}
        .topbar {{ flex-direction: column; align-items: flex-start; }}
        .badge {{ white-space: normal; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Chart helpers ----------
def apply_chart_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", color=TEXT_COLOR, size=12),
        margin=dict(l=30, r=30, t=72, b=48),
        title=dict(font=dict(size=15, color="#FFFFFF"), x=0.02, xanchor="left"),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        autosize=True,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, automargin=True)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, automargin=True)
    return fig


def show_chart(fig, height=380):
    fig.update_layout(height=height, autosize=True)
    st.plotly_chart(
        apply_chart_theme(fig),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def panel_heading(title, subtitle=None):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="panel-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


# ---------- Load data and model ----------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    if "InternetService" in data.columns:
        data["InternetService"] = data["InternetService"].fillna("Unknown")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(
        0
    )
    data["ChurnLabel"] = data["Churn"].astype(str).str.strip().str.title()
    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("`churn_model.pkl` was not found. Run `customer_churn.py` first.")
    st.stop()

if not DATA_PATH.exists():
    st.error("`customer_churn_data.csv` was not found.")
    st.stop()

df = load_data()
model = load_model()


# ---------- Header ----------
model_name = (
    model.steps[-1][1].__class__.__name__
    if hasattr(model, "steps")
    else model.__class__.__name__
)

st.markdown(
    f"""
    <div class="topbar">
      <div class="topbar-left">
        <div class="logo">◈</div>
        <div>
          <h1>Churn Risk Intelligence</h1>
          <p>Interactive Telco churn prediction, customer analytics, and model performance monitoring.</p>
        </div>
      </div>
      <div class="badge"><i></i> Active model · {model_name}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Dataset KPIs ----------
total_customers = len(df)
churned = int((df["ChurnLabel"] == "Yes").sum())
churn_rate = churned / total_customers if total_customers else 0.0
avg_monthly = float(df["MonthlyCharges"].mean())
avg_tenure = float(df["tenure"].mean())

st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Customers</div>
        <div class="kpi-value">{total_customers:,}</div>
        <div class="kpi-sub">Telco records available</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Churned customers</div>
        <div class="kpi-value" style="color:{HIGH_C}">{churned:,}</div>
        <div class="kpi-sub">Customers labeled as churned</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Overall churn rate</div>
        <div class="kpi-value" style="color:{MID_C}">{churn_rate:.1%}</div>
        <div class="kpi-sub">Share of customers who churned</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Average monthly charges</div>
        <div class="kpi-value">${avg_monthly:,.2f}</div>
        <div class="kpi-sub">Average tenure: {avg_tenure:.1f} months</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Tabs ----------
tab_prediction, tab_insights, tab_model = st.tabs(
    ["🎯 Churn prediction", "📊 Data insights", "🧠 Model performance"]
)


# =============================
# Prediction tab
# =============================
with tab_prediction:
    with st.container(border=True):
        with st.form("prediction_form"):
            panel_heading(
                "Customer profile",
                "Enter complete Telco service details and run predictions.",
            )
            c1, c2, c3, c4 = st.columns(4, gap="medium")

            with c1:
                st.markdown(
                    '<div class="field-label">Gender</div>', unsafe_allow_html=True
                )
                gender = st.selectbox(
                    "gender", ["Female", "Male"], label_visibility="collapsed"
                )

                st.markdown(
                    '<div class="field-label">Senior citizen</div>',
                    unsafe_allow_html=True,
                )
                senior_citizen = st.selectbox(
                    "SeniorCitizen",
                    [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    label_visibility="collapsed",
                )

                st.markdown(
                    '<div class="field-label">Partner</div>', unsafe_allow_html=True
                )
                partner = st.selectbox(
                    "Partner", ["Yes", "No"], label_visibility="collapsed"
                )

            with c2:
                st.markdown(
                    '<div class="field-label">Dependents</div>', unsafe_allow_html=True
                )
                dependents = st.selectbox(
                    "Dependents", ["No", "Yes"], label_visibility="collapsed"
                )

                st.markdown(
                    '<div class="field-label">Tenure (months)</div>',
                    unsafe_allow_html=True,
                )
                tenure = st.slider("tenure", 0, 72, 12, label_visibility="collapsed")

                st.markdown(
                    '<div class="field-label">Phone service</div>',
                    unsafe_allow_html=True,
                )
                phone_service = st.selectbox(
                    "PhoneService", ["Yes", "No"], label_visibility="collapsed"
                )

            with c3:
                st.markdown(
                    '<div class="field-label">Multiple lines</div>',
                    unsafe_allow_html=True,
                )
                multiple_lines = st.selectbox(
                    "MultipleLines",
                    ["No", "Yes", "No phone service"],
                    label_visibility="collapsed",
                )

                st.markdown(
                    '<div class="field-label">Internet service</div>',
                    unsafe_allow_html=True,
                )
                internet_service = st.selectbox(
                    "InternetService",
                    ["Fiber optic", "DSL", "No"],
                    label_visibility="collapsed",
                )

                st.markdown(
                    '<div class="field-label">Online security</div>',
                    unsafe_allow_html=True,
                )
                online_security = st.selectbox(
                    "OnlineSecurity",
                    ["No", "Yes", "No internet service"],
                    label_visibility="collapsed",
                )

            with c4:
                st.markdown(
                    '<div class="field-label">Contract</div>', unsafe_allow_html=True
                )
                contract = st.selectbox(
                    "Contract",
                    ["Month-to-month", "One year", "Two year"],
                    label_visibility="collapsed",
                )

                st.markdown(
                    '<div class="field-label">Monthly charges ($)</div>',
                    unsafe_allow_html=True,
                )
                monthly_charges = st.number_input(
                    "MonthlyCharges",
                    min_value=0.0,
                    max_value=200.0,
                    value=70.0,
                    step=1.0,
                    label_visibility="collapsed",
                )

                st.markdown(
                    '<div class="field-label">Total charges ($)</div>',
                    unsafe_allow_html=True,
                )
                total_charges = st.number_input(
                    "TotalCharges",
                    min_value=0.0,
                    max_value=10000.0,
                    value=840.0,
                    step=10.0,
                    label_visibility="collapsed",
                )

            # Additional common options populated with defaults for inference
            device_protection = "No"
            online_backup = "No"
            tech_support = "No"
            streaming_tv = "No"
            streaming_movies = "No"
            paperless_billing = "Yes"
            payment_method = "Electronic check"

            submitted = st.form_submit_button(
                "Run prediction", type="primary", use_container_width=True
            )

    if "current" not in st.session_state:
        st.session_state.current = None
    if "history" not in st.session_state:
        st.session_state.history = []

    def get_inference_row():
        return {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

    if submitted:
        new_row = get_inference_row()
        input_df = pd.DataFrame([new_row])
        prob = float(model.predict_proba(input_df)[0][1])
        st.session_state.current = {"row": new_row, "p": prob}

    current = st.session_state.current

    if current is None:
        st.info(
            "Enter customer profile parameters above and select **Run prediction**."
        )
    else:
        row = current["row"]
        p = current["p"]

        if p < LOW:
            band, risk_color = "Low risk", LOW_C
            action = "No immediate retention action required. Customer exhibits strong stability indicators."
        elif p < HIGH:
            band, risk_color = "Medium risk", MID_C
            action = "Schedule proactive service check-in before the next billing cycle. Offer targeted package optimization."
        else:
            band, risk_color = "High risk", HIGH_C
            action = "High churn vulnerability. Prioritize immediate outreach, review contract choices, or offer retention incentives."

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=p * 100,
                number={"suffix": "%", "font": {"size": 42, "color": risk_color}},
                title={"text": "Predicted churn probability", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": MUTED},
                    "bar": {"color": risk_color, "thickness": 0.28},
                    "bgcolor": "#161B24",
                    "borderwidth": 1,
                    "bordercolor": GRID_COLOR,
                    "steps": [
                        {"range": [0, LOW * 100], "color": "rgba(34,197,94,0.18)"},
                        {
                            "range": [LOW * 100, HIGH * 100],
                            "color": "rgba(234,179,8,0.18)",
                        },
                        {"range": [HIGH * 100, 100], "color": "rgba(239,68,68,0.18)"},
                    ],
                },
            )
        )

        left, right = st.columns([1, 1.15], gap="medium")

        with left:
            with st.container(border=True):
                panel_heading(
                    "Risk assessment", "Classification & recommended retention strategy"
                )
                st.markdown(
                    f'<span class="risk-pill" style="color:{risk_color};border-color:{risk_color};background:rgba(255,255,255,0.03)">{band}</span>',
                    unsafe_allow_html=True,
                )
                show_chart(gauge_fig, height=310)
                st.markdown(
                    f'<div class="recommendation">{action}</div>',
                    unsafe_allow_html=True,
                )

        with right:
            with st.container(border=True):
                panel_heading(
                    "Customer Profile Summary", "Parameters sent to classifier pipeline"
                )
                summary_df = pd.DataFrame(
                    list(row.items()), columns=["Feature", "Value"]
                )
                st.dataframe(
                    summary_df, use_container_width=True, hide_index=True, height=330
                )

        if submitted and (
            not st.session_state.history or st.session_state.history[-1]["row"] != row
        ):
            st.session_state.history.append(
                {
                    "row": row,
                    "p": p,
                    "label": f"Tenure: {row['tenure']}m · Contract: {row['Contract']} · ${row['MonthlyCharges']}",
                }
            )

        if len(st.session_state.history) > 1:
            history_df = pd.DataFrame(st.session_state.history)
            history_df["Prediction"] = history_df["p"] * 100
            history_df["Run"] = range(1, len(history_df) + 1)

            history_fig = px.line(
                history_df,
                x="Run",
                y="Prediction",
                markers=True,
                title="Prediction history",
                labels={"Prediction": "Churn probability (%)", "Run": "Run index"},
            )
            history_fig.update_traces(line_color=ACCENT, marker_color=ACCENT)
            with st.container(border=True):
                show_chart(history_fig, height=300)


# =============================
# Data insights tab
# =============================
with tab_insights:
    panel_heading(
        "Customer Base Analytics",
        "Explore churn distribution across contracts, tenure, and charges.",
    )

    churn_counts = (
        df["ChurnLabel"]
        .value_counts()
        .reindex(["No", "Yes"], fill_value=0)
        .rename_axis("Churn")
        .reset_index(name="Customers")
    )
    churn_pie = px.pie(
        churn_counts,
        names="Churn",
        values="Customers",
        hole=0.62,
        color="Churn",
        color_discrete_map=CHURN_COLORS,
        title="Churn distribution",
    )

    contract_df = (
        df.groupby("Contract", dropna=False)["MonthlyCharges"].mean().reset_index()
    )
    contract_fig = px.bar(
        contract_df,
        x="Contract",
        y="MonthlyCharges",
        text_auto=".1f",
        title="Average monthly charges by contract type",
        color="MonthlyCharges",
        color_continuous_scale="Blues",
    )
    contract_fig.update_layout(coloraxis_showscale=False)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        with st.container(border=True):
            show_chart(churn_pie, height=360)
    with col2:
        with st.container(border=True):
            show_chart(contract_fig, height=360)

    tenure_fig = px.histogram(
        df,
        x="tenure",
        color="ChurnLabel",
        nbins=30,
        barmode="overlay",
        opacity=0.72,
        color_discrete_map=CHURN_COLORS,
        title="Tenure distribution by churn status",
    )
    charges_fig = px.histogram(
        df,
        x="MonthlyCharges",
        color="ChurnLabel",
        nbins=30,
        barmode="overlay",
        opacity=0.72,
        color_discrete_map=CHURN_COLORS,
        title="Monthly charges distribution by churn status",
    )

    col3, col4 = st.columns(2, gap="medium")
    with col3:
        with st.container(border=True):
            show_chart(tenure_fig, height=370)
    with col4:
        with st.container(border=True):
            show_chart(charges_fig, height=370)


# =============================
# Model performance tab
# =============================
with tab_model:
    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)
        metrics_df = metrics_df.sort_values("ROC_AUC", ascending=False)
        best = metrics_df.iloc[0]

        st.markdown(
            f"""
            <div class="kpi-grid">
              <div class="kpi">
                <div class="kpi-label">Best model</div>
                <div class="kpi-value" style="font-size:1.35rem">{best["Model"]}</div>
                <div class="kpi-sub">Selected via cross-validated ROC-AUC</div>
              </div>
              <div class="kpi">
                <div class="kpi-label">ROC-AUC</div>
                <div class="kpi-value" style="color:{ACCENT}">{best["ROC_AUC"]:.3f}</div>
                <div class="kpi-sub">Area under ROC curve</div>
              </div>
              <div class="kpi">
                <div class="kpi-label">Recall</div>
                <div class="kpi-value" style="color:{MID_C}">{best["Recall"]:.3f}</div>
                <div class="kpi-sub">Churn capture rate</div>
              </div>
              <div class="kpi">
                <div class="kpi-label">Precision</div>
                <div class="kpi-value" style="color:{LOW_C}">{best["Precision"]:.3f}</div>
                <div class="kpi-sub">Prediction accuracy for churners</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_fig = go.Figure()
        for metric, color in [
            ("ROC_AUC", ACCENT),
            ("Accuracy", LOW_C),
            ("Recall", MID_C),
            ("Precision", HIGH_C),
        ]:
            if metric in metrics_df.columns:
                metric_fig.add_trace(
                    go.Bar(
                        x=metrics_df["Model"],
                        y=metrics_df[metric],
                        name=metric,
                        marker_color=color,
                        text=[f"{v:.3f}" for v in metrics_df[metric]],
                        textposition="outside",
                    )
                )

        metric_fig.update_layout(
            barmode="group",
            height=430,
            title="Model Evaluation Comparison",
            yaxis=dict(range=[0, 1.15]),
        )

        with st.container(border=True):
            show_chart(metric_fig, height=430)

        with st.container(border=True):
            panel_heading(
                "Detailed metrics", "Test-set evaluation for all trained models."
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No `model_metrics.csv` found. Run `customer_churn.py` first.")

st.markdown(
    """
    <p class="footnote">
      Predictions are statistical estimates based on historical Telco customer datasets ([Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)). Risk thresholds: low below 30%, medium 30% to 60%, high at or above 60%.
    </p>
    """,
    unsafe_allow_html=True,
)
