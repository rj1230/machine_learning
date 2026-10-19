"""
TrendNet long-only strategy UI.

Run:
    streamlit run app.py
"""

import html as html_lib
import warnings
from string import Template


import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)


from main import (
    prepare_dataset,
    split_data,
    select_strategy,
    evaluate_long_strategy,
    simulate_long_trades,
    strategy_stats,
    apply_filter,
    make_shap_safe_matrix,
    CANDIDATE_LABELS,
    LONG_THRESHOLDS,
    STOP_ATR_MULT,
    TARGET_ATR_MULT,
    MAX_HOLD_BARS,
    COST_SCENARIOS_BPS,
    GAP,
)


warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="TrendNet | Long-only trend strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


CLASS_NAMES = {0: "Range", 1: "Downtrend", 2: "Uptrend"}


# ------------------------------------------------------------------ design
PAPER = "#0B0F17"
SURFACE = "#111826"
SURFACE_ALT = "#161F30"
INK = "#E7ECF5"
MUTED = "#9AA6B8"
LINE = "#253044"
GRID = "#1D2637"
ACCENT = "#5B8CFF"


UP = "#2DD4A7"
DOWN = "#FF6B6B"
RANGE = "#8B95A8"
REGIME_COLORS = {0: RANGE, 1: DOWN, 2: UP}


SERIF = "'Source Serif 4', Georgia, serif"
SANS = "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif"


RIBBON_BARS = 120


CSS = Template(
    """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');



.stApp {
    background:
        radial-gradient(1100px 500px at 12% -10%, rgba(91, 140, 255, 0.13), transparent 60%),
        radial-gradient(900px 500px at 90% 0%, rgba(45, 212, 167, 0.08), transparent 55%),
        $paper;
    color: $ink;
}


.stApp, .stApp .stMarkdown, .stApp label, .stApp button, .stApp input,
.stApp [data-testid="stMetric"], .stApp [data-testid="stSidebar"] {
    font-family: $sans;
}


.stApp .block-container {
    padding: 2.2rem 2.5rem 4rem;
    max-width: 1320px;
}


#MainMenu, footer, [data-testid="stDeployButton"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }


.stApp [data-testid="stSidebar"] {
    background: $surface;
    border-right: 1px solid $line;
}


.brand {
    font-family: $serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: $ink;
}


.brand-sub {
    color: $muted;
    font-size: 0.84rem;
    margin: 4px 0 8px;
}


.side-h {
    font-weight: 600;
    font-size: 0.92rem;
    color: $ink;
    margin: 0.6rem 0 0.1rem;
}


.hero {
    background: linear-gradient(145deg, $surface 0%, $surface_alt 100%);
    border: 1px solid $line;
    border-radius: 14px;
    padding: 26px 30px 22px;
    margin-bottom: 1.1rem;
}


.hero-title {
    font-family: $serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: $ink;
}


.hero-sub {
    color: $muted;
    margin-top: 0.5rem;
    max-width: 70ch;
}


.chips { display: flex; gap: 8px; flex-wrap: wrap; }


.chip {
    border: 1px solid $line;
    background: rgba(91, 140, 255, 0.08);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.8rem;
}


.ribbon {
    display: flex;
    gap: 2px;
    height: 14px;
    margin-top: 22px;
}


.ribbon span { flex: 1; border-radius: 2px; }


.ribbon-meta {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
    font-size: 0.8rem;
    color: $muted;
}


.key {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-right: 14px;
}


.key i {
    width: 10px;
    height: 10px;
    border-radius: 2px;
}


.sec { margin: 1.8rem 0 0.7rem; }


.sec-title {
    font-family: $serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: $ink;
}


.sec-desc {
    color: $muted;
    font-size: 0.9rem;
    margin-top: 0.25rem;
}


.stApp [data-testid="stMetric"] {
    background: $surface;
    border: 1px solid $line;
    border-radius: 12px;
    padding: 14px 18px;
}


.stApp [data-testid="stPlotlyChart"] {
    background: $surface;
    border: 1px solid $line;
    border-radius: 12px;
}


.stApp [data-testid="stDataFrame"] {
    border: 1px solid $line;
    border-radius: 12px;
}


.stApp [data-testid="stExpander"] {
    background: $surface;
    border: 1px solid $line;
    border-radius: 12px;
}


.stApp input,
.stApp [data-baseweb="select"] > div {
    background-color: $surface_alt !important;
    border-color: $line !important;
    color: $ink !important;
}


.stApp .stButton > button[kind="primary"] {
    background: $accent;
    border-color: $accent;
    color: #061024;
}


.foot {
    color: $muted;
    font-size: 0.8rem;
    border-top: 1px solid $line;
    margin-top: 2.5rem;
    padding-top: 1rem;
}
"""
).substitute(
    paper=PAPER,
    surface=SURFACE,
    surface_alt=SURFACE_ALT,
    ink=INK,
    muted=MUTED,
    line=LINE,
    accent=ACCENT,
    serif=SERIF,
    sans=SANS,
)


st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ cache
@st.cache_data(show_spinner="Downloading data and building features...")
def load_and_prepare(symbol: str, start_date: str, end_date: str | None):
    return prepare_dataset(
        symbol=symbol,
        start=start_date,
        end=end_date,
    )


@st.cache_resource(show_spinner="Selecting and training strategy...")
def run_selection(
    symbol: str,
    start_date: str,
    end_date: str | None,
):
    df, label_cols, feature_cols = prepare_dataset(
        symbol=symbol,
        start=start_date,
        end=end_date,
    )

    train_df, selection_df, test_df = split_data(df)

    train_df.attrs["symbol"] = symbol
    selection_df.attrs["symbol"] = symbol
    test_df.attrs["symbol"] = symbol

    (
        selected_label,
        selected_threshold,
        selected_filter,
        selection_results,
        model,
    ) = select_strategy(
        train_df=train_df,
        selection_df=selection_df,
        feature_cols=feature_cols,
        label_cols=CANDIDATE_LABELS,
        symbol=symbol,
    )

    return {
        "symbol": symbol,
        "df": df,
        "label_cols": label_cols,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "selection_df": selection_df,
        "test_df": test_df,
        "selected_label": selected_label,
        "selected_threshold": selected_threshold,
        "selected_filter": selected_filter,
        "selection_results": selection_results,
        "model": model,
    }


# ------------------------------------------------------------------ helpers
def section(title: str, desc: str | None = None):
    body = f'<div class="sec-title">{title}</div>'

    if desc:
        body += f'<div class="sec-desc">{desc}</div>'

    st.markdown(
        f'<div class="sec">{body}</div>',
        unsafe_allow_html=True,
    )


def side_heading(text: str):
    st.markdown(
        f'<div class="side-h">{text}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int, legend: bool = True):
    fig.update_layout(
        height=height,
        font=dict(family=SANS, size=12, color=MUTED),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=12, r=12, t=44 if legend else 16, b=12),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=INK),
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor=LINE,
        tickcolor=LINE,
        zeroline=False,
        tickfont=dict(color=MUTED),
    )

    fig.update_yaxes(
        gridcolor=GRID,
        linecolor=LINE,
        zeroline=False,
        tickfont=dict(color=MUTED),
    )

    return fig


def render_hero(r: dict, threshold: float, filter_name: str):
    df = r["df"]

    span = (
        f"{pd.Timestamp(df.index.min()):%d %b %Y} to "
        f"{pd.Timestamp(df.index.max()):%d %b %Y}"
    )

    symbol = html_lib.escape(str(r["symbol"]))

    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">TrendNet</div>'
        '<div class="hero-sub">Long-only ML trend strategy. The label, probability threshold, '
        "and regime filter are selected on a separate selection period, then evaluated once "
        "on the final untouched test period.</div>"
        f'<div class="chips">'
        f'<span class="chip">{symbol}</span>'
        f'<span class="chip">{span}</span>'
        f'<span class="chip">Label: {r["selected_label"]}</span>'
        f'<span class="chip">Threshold: {threshold:.2f}</span>'
        f'<span class="chip">Filter: {filter_name}</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def price_chart(df: pd.DataFrame, bars: int):
    recent = df.tail(bars)

    fig = go.Figure(
        go.Candlestick(
            x=recent.index,
            open=recent["Open"],
            high=recent["High"],
            low=recent["Low"],
            close=recent["Close"],
            name="OHLC",
            increasing_line_color=UP,
            increasing_fillcolor=UP,
            decreasing_line_color=DOWN,
            decreasing_fillcolor=DOWN,
        )
    )

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    return style_fig(fig, 500, legend=False)


def equity_chart(equity: pd.Series, buy_hold: pd.Series):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity,
            mode="lines",
            name="ML long-only",
            line=dict(color=ACCENT, width=2.2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=buy_hold.index,
            y=buy_hold,
            mode="lines",
            name="Buy & hold",
            line=dict(color="#7C8798", width=1.6, dash="dot"),
        )
    )

    fig.update_yaxes(title_text="Growth of 1 unit")
    fig.update_layout(hovermode="x unified")

    return style_fig(fig, 460)


def drawdown_chart(equity: pd.Series):
    drawdown = equity / equity.cummax() - 1

    fig = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line=dict(color=DOWN, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.12)",
        )
    )

    fig.update_yaxes(title_text="Drawdown", tickformat=".0%")

    return style_fig(fig, 280, legend=False)


def probability_chart(probabilities, index, threshold: float):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=index,
            y=probabilities,
            mode="lines",
            name="Uptrend probability",
            line=dict(color=ACCENT, width=1.8),
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=UP,
        annotation_text="Long threshold",
    )

    fig.update_yaxes(title_text="Probability", range=[0, 1])

    return style_fig(fig, 340)


def selection_chart(results: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["threshold"],
            y=results["sharpe"],
            mode="markers",
            name="Selection Sharpe",
            marker=dict(
                size=10,
                color=results["trades"],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Trades"),
            ),
        )
    )

    fig.update_xaxes(title_text="Threshold")
    fig.update_yaxes(title_text="Sharpe")

    return style_fig(fig, 420)


def confusion_chart(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig = px.imshow(
        cm_norm,
        text_auto=".2f",
        color_continuous_scale=[[0, "#141C2B"], [1, ACCENT]],
        zmin=0,
        zmax=1,
        x=[CLASS_NAMES[i] for i in range(3)],
        y=[CLASS_NAMES[i] for i in range(3)],
        labels=dict(x="Predicted", y="Actual", color="Row share"),
    )

    fig.update_coloraxes(showscale=False)

    return style_fig(fig, 400, legend=False)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(
        '<div class="brand">TrendNet</div>'
        '<div class="brand-sub">Long-only ML trend strategy with ATR-based exits.</div>',
        unsafe_allow_html=True,
    )

    side_heading("Data")

    symbol = st.text_input(
        "Ticker",
        value="ES=F",
        help="Examples: ES=F, NQ=F, YM=F, RTY=F",
    )

    start_date = st.date_input(
        "Start date",
        value=pd.Timestamp("2010-01-01"),
    )

    use_end_date = st.checkbox("Set custom end date", value=False)

    end_date = None

    if use_end_date:
        end_date = st.date_input("End date", value=pd.Timestamp.today())

    side_heading("Strategy overrides")

    use_manual_threshold = st.checkbox(
        "Override selected threshold",
        value=False,
    )

    manual_threshold = st.slider(
        "Long probability threshold",
        0.45,
        0.75,
        0.50,
        0.01,
    )

    use_manual_filter = st.checkbox(
        "Override selected filter",
        value=False,
    )

    manual_filter = st.selectbox(
        "Regime filter",
        ["none", "adx", "atr_rank", "adx_atr_rank"],
    )

    stop_atr = st.slider(
        "Stop-loss: ATR multiple",
        0.5,
        4.0,
        STOP_ATR_MULT,
        0.1,
    )

    target_atr = st.slider(
        "Profit target: ATR multiple",
        1.0,
        8.0,
        TARGET_ATR_MULT,
        0.1,
    )

    max_hold = st.slider(
        "Maximum holding period",
        3,
        30,
        MAX_HOLD_BARS,
        1,
    )

    cost_bps = st.slider(
        "Transaction cost: basis points per side",
        0.0,
        10.0,
        COST_SCENARIOS_BPS[0],
        0.5,
    )

    st.write("")

    run_button = st.button(
        "Run strategy",
        type="primary",
        use_container_width=True,
    )


# ------------------------------------------------------------------ pipeline
if "results" not in st.session_state:
    st.session_state.results = None


if run_button or st.session_state.results is None:
    try:
        with st.spinner("Preparing data and selecting strategy..."):
            base = run_selection(
                symbol=symbol,
                start_date=str(start_date),
                end_date=str(end_date) if end_date else None,
            )

        threshold = (
            manual_threshold if use_manual_threshold else base["selected_threshold"]
        )

        filter_name = manual_filter if use_manual_filter else base["selected_filter"]

        with st.spinner("Backtesting selection period..."):
            selection_evaluation = evaluate_long_strategy(
                df_period=base["selection_df"],
                X_period=base["selection_df"][base["feature_cols"]],
                model=base["model"],
                threshold=threshold,
                filter_name=filter_name,
                period_name="SELECTION",
            )

        with st.spinner("Backtesting final test period..."):
            test_evaluation = evaluate_long_strategy(
                df_period=base["test_df"],
                X_period=base["test_df"][base["feature_cols"]],
                model=base["model"],
                threshold=threshold,
                filter_name=filter_name,
                period_name="TEST",
            )

        st.session_state.results = {
            **base,
            "threshold": threshold,
            "filter_name": filter_name,
            "selection_evaluation": selection_evaluation,
            "test_evaluation": test_evaluation,
        }

    except Exception as error:
        st.error(f"The strategy stopped with an error: `{error}`")
        st.stop()


r = st.session_state.results
threshold = r["threshold"]
filter_name = r["filter_name"]


render_hero(r, threshold, filter_name)


test_result = r["test_evaluation"]["results_by_cost"][cost_bps]["stats"]


col1, col2, col3, col4 = st.columns(4)


col1.metric("Test trades", f"{test_result['trades']:,}")
col2.metric("Test return", f"{test_result['total_ret_%']:.2f}%")
col3.metric("Test Sharpe", f"{test_result['sharpe']:.2f}")
col4.metric("Test max drawdown", f"{test_result['max_dd_%']:.2f}%")


col5, col6, col7, col8 = st.columns(4)


col5.metric("Win rate", f"{test_result['win_rate_%']:.1f}%")
col6.metric("Median trade", f"{test_result['median_trade_%']:.2f}%")
col7.metric("Market exposure", f"{test_result['time_in_mkt_%']:.1f}%")
col8.metric(
    "Selected threshold",
    f"{r['selected_threshold']:.2f}",
    help="Threshold chosen on the selection period.",
)


# ------------------------------------------------------------------ tabs
tab_data, tab_strategy, tab_trades, tab_model, tab_diag = st.tabs(
    ["Data", "Strategy", "Trades", "Model", "Diagnostics"]
)


# ------------------------------------------------------------------ data
with tab_data:
    section(
        "Price history",
        "Daily candles for the selected ticker.",
    )

    chart_bars = st.slider(
        "Recent bars",
        60,
        1000,
        250,
        50,
        key="price_bars",
    )

    st.plotly_chart(
        price_chart(r["df"], chart_bars),
        width="stretch",
    )

    section(
        "Latest observations",
        "The most recent rows, including engineered features.",
    )

    st.dataframe(
        r["df"].tail(15),
        width="stretch",
    )


# ------------------------------------------------------------------ strategy
with tab_strategy:
    section(
        "Strategy selection",
        "Each point is a label, threshold, and filter configuration evaluated on the "
        "selection period. The final test period is not used for this choice.",
    )

    st.plotly_chart(
        selection_chart(r["selection_results"]),
        width="stretch",
    )

    with st.expander("View selection results"):
        st.dataframe(
            r["selection_results"].round(4),
            width="stretch",
            hide_index=True,
        )

    section(
        "Final-test equity and drawdown",
        "Long-only strategy equity compared with buy-and-hold.",
    )

    test_equity = r["test_evaluation"]["results_by_cost"][cost_bps]["equity"]
    test_buy_hold = r["test_df"]["Close"] / r["test_df"]["Close"].iloc[0]

    st.plotly_chart(
        equity_chart(test_equity, test_buy_hold),
        width="stretch",
    )

    st.plotly_chart(
        drawdown_chart(test_equity),
        width="stretch",
    )

    section(
        "Cost sensitivity",
        "The same final-test strategy at different transaction-cost assumptions.",
    )

    cost_rows = []

    for cost, result in r["test_evaluation"]["results_by_cost"].items():
        cost_rows.append(
            {
                "Cost per side": f"{cost:.1f} bp",
                **result["stats"],
            }
        )

    st.dataframe(
        pd.DataFrame(cost_rows).round(4),
        width="stretch",
        hide_index=True,
    )

    section(
        "Uptrend probability",
        "Model probability of an uptrend regime on the final test period.",
    )

    st.plotly_chart(
        probability_chart(
            r["test_evaluation"]["probabilities"],
            r["test_df"].index,
            threshold,
        ),
        width="stretch",
    )


# ------------------------------------------------------------------ trades
with tab_trades:
    section(
        "Final-test trades",
        "Every simulated long trade, including entry, exit, stop, target, holding period, "
        "and net return after transaction costs.",
    )

    trades = r["test_evaluation"]["results_by_cost"][cost_bps]["trades"]

    if trades.empty:
        st.info("No trades were generated with the current settings.")
    else:
        st.dataframe(
            trades.reset_index(drop=True),
            width="stretch",
        )

        csv = trades.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download trade history",
            csv,
            file_name="trendnet_long_trades_test.csv",
            mime="text/csv",
        )

    section(
        "Exit reasons",
        "A healthy strategy should have a meaningful mix of target, stop, and time-based exits.",
    )

    if trades.empty:
        st.info("No exits to display.")
    else:
        exit_counts = trades["exit_reason"].value_counts()

        fig = go.Figure(
            go.Bar(
                x=exit_counts.index,
                y=exit_counts.values,
                marker_color=[UP, DOWN, "#7C8798"][: len(exit_counts)],
                text=exit_counts.values,
                textposition="outside",
            )
        )

        fig.update_yaxes(title_text="Trades")

        st.plotly_chart(
            style_fig(fig, 320, legend=False),
            width="stretch",
        )


# ------------------------------------------------------------------ model
with tab_model:
    selection_probabilities = r["selection_evaluation"]["probabilities"]

    selection_signals = apply_filter(
        selection_probabilities,
        threshold,
        r["selection_df"],
        filter_name,
    )

    section(
        "Selection classification performance",
        "Model performance on the selection period for the selected label.",
    )

    y_selection = r["selection_df"][r["selected_label"]]

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Accuracy",
        f"{accuracy_score(y_selection, selection_signals):.4f}",
    )

    m2.metric(
        "Macro-F1",
        f"{f1_score(y_selection, selection_signals, average='macro'):.4f}",
    )

    m3.metric(
        "MCC",
        f"{matthews_corrcoef(y_selection, selection_signals):.4f}",
    )

    section(
        "Confusion matrix",
        "Each row is an actual regime; the diagonal shows the share correctly identified.",
    )

    st.plotly_chart(
        confusion_chart(y_selection, selection_signals),
        width="stretch",
    )

    section(
        "Classification report",
        "Precision, recall, and F1 for each regime on the selection period.",
    )

    report = classification_report(
        y_selection,
        selection_signals,
        target_names=[CLASS_NAMES[i] for i in range(3)],
        output_dict=True,
        zero_division=0,
    )

    st.dataframe(
        pd.DataFrame(report).T.round(3),
        width="stretch",
    )


# ------------------------------------------------------------------ diagnostics
with tab_diag:
    section(
        "Trade concentration",
        "High top-3 trade concentration means the result depends heavily on a few large winners.",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Mean trade",
        f"{test_result['avg_trade_%']:.2f}%",
    )

    c2.metric(
        "Median trade",
        f"{test_result['median_trade_%']:.2f}%",
    )

    c3.metric(
        "Top-3 trade share",
        f"{test_result['top3_trade_share_%']:.1f}%",
    )

    if test_result["top3_trade_share_%"] > 50:
        st.warning(
            "More than half of the strategy’s total return comes from its three best trades. "
            "Treat the result as fragile."
        )

    elif test_result["median_trade_%"] <= 0:
        st.warning(
            "The median trade is not positive. The strategy may rely on occasional large winners."
        )

    else:
        st.success(
            "The median trade is positive and outlier concentration is below 50%."
        )

    section(
        "Selected configuration",
        "This configuration was selected using the selection period, then evaluated on the "
        "final untouched test period.",
    )

    st.metric("Label", r["selected_label"])
    st.metric("Threshold", f"{r['selected_threshold']:.2f}")
    st.metric("Filter", r["selected_filter"])

    section(
        "Features",
        "Numeric features available to the model after sanitization.",
    )

    with st.expander(f"View all {len(r['feature_cols'])} features"):
        st.dataframe(
            pd.DataFrame({"feature": r["feature_cols"]}),
            width="stretch",
            hide_index=True,
        )


# ------------------------------------------------------------------ footer
st.markdown(
    '<div class="foot">TrendNet is a research tool, not investment advice. '
    "Backtests are net of modelled transaction costs, use simplified execution assumptions, "
    "and do not guarantee future results.</div>",
    unsafe_allow_html=True,
)
