"""
TrendNet: Long-only ML trend strategy

Research workflow:
    Train -> selection -> final untouched test

The model identifies potential uptrend regimes. Trades are long-only and use
ATR-based stop-loss, profit-target, and maximum-holding-period exits.


CLI:
    python main.py
"""

import hashlib
import os
import warnings


import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
import pandas_ta_classic as ta
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)


warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ config
PRIMARY_SYMBOL = "ES=F"
ROBUSTNESS_SYMBOLS = ["NQ=F", "YM=F", "RTY=F"]

START = "2010-01-01"
END = None

OUT_DIR = "outputs"
MODEL_DIR = "models"

LOOKAHEADS = [2, 4, 6, 8, 10]
THRESHOLDS = [0.01, 0.02]
ATR_MULTS = [1.0, 1.5]
GAP = max(LOOKAHEADS)

SEED = 42
BARS_PER_YEAR = 252

# Candidate labels are selected on the selection period, never on final test.
CANDIDATE_LABELS = [
    "label_la8_atr1.5",
    "label_la10_th0.010",
    "label_la2_th0.020",
    "label_la4_atr1.0",
    "label_la8_th0.010",
]

LONG_THRESHOLDS = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
MIN_SELECTION_TRADES = 20

STOP_ATR_MULT = 1.5
TARGET_ATR_MULT = 3.0
MAX_HOLD_BARS = 10

ADX_FILTER_MIN = 20.0
ATR_RANK_FILTER_MIN = 0.30

COST_SCENARIOS_BPS = [1.0, 2.0, 5.0]

FEATURE_VERSION = "v10-selection-split-filters"


os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------------ data
def load_data(
    symbol: str,
    start: str = START,
    end=None,
) -> pd.DataFrame:
    safe_symbol = symbol.replace("=", "_").replace("^", "").replace("/", "_")
    csv_path = f"{safe_symbol}_data.csv"

    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise RuntimeError(
            f"Download returned no data for {symbol}. "
            "Check the ticker or update yfinance."
        )

    df.to_csv(csv_path)
    return df


# ------------------------------------------------------------------ features
def _merge(df_main: pd.DataFrame, ta_obj) -> pd.DataFrame:
    if isinstance(ta_obj, pd.DataFrame):
        return pd.concat([df_main, ta_obj], axis=1)

    return pd.concat([df_main, ta_obj.rename(ta_obj.name)], axis=1)


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    for length in [5, 10, 15]:
        df[f"rsi_{length}"] = ta.rsi(df["Close"], length=length)

    df["roc_10"] = ta.roc(df["Close"], length=10)
    df["mom_10"] = ta.mom(df["Close"], length=10)

    df = _merge(df, ta.stochrsi(df["Close"]))
    df["cci_20"] = ta.cci(df["High"], df["Low"], df["Close"], length=20)
    df["wr_14"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)
    df = _merge(df, ta.kst(df["Close"]))
    df["macd"] = ta.macd(df["Close"])["MACD_12_26_9"]

    for length in [5, 10, 20]:
        df[f"sma_{length}"] = ta.sma(df["Close"], length=length)
        df[f"ema_{length}"] = ta.ema(df["Close"], length=length)

    df["vwma_20"] = ta.vwma(df["Close"], df["Volume"], length=20)

    df = _merge(df, ta.bbands(df["Close"], length=20))
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df = _merge(df, ta.kc(df["High"], df["Low"], df["Close"], length=20))

    df["adx_14"] = ta.adx(df["High"], df["Low"], df["Close"], length=14)["ADX_14"]

    return df


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ret_1"] = df["Close"].pct_change()
    df["ret_5"] = df["Close"].pct_change(5)

    for n in [5, 10, 20]:
        df[f"close_sma{n}"] = df["Close"] / df[f"sma_{n}"] - 1
        df[f"close_ema{n}"] = df["Close"] / df[f"ema_{n}"] - 1

    df["close_vwma20"] = df["Close"] / df["vwma_20"] - 1
    df["atr_pct"] = df["atr_14"] / df["Close"]
    df["atr_rank_252"] = (
        df["atr_pct"].rolling(252).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    )
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["vol_chg"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["relative_volume_20"] = df["Volume"] / df["Volume"].rolling(20).mean()

    return df


def select_feature_cols(df: pd.DataFrame) -> list:
    raw_price = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "vwma_20",
        "atr_14",
        "sma_5",
        "sma_10",
        "sma_20",
        "ema_5",
        "ema_10",
        "ema_20",
        "adx_14",
    ]

    keep_bb = {"BBP_20_2.0", "BBB_20_2.0"}
    drop = set(raw_price)
    drop |= {
        column
        for column in df.columns
        if column.startswith(("BB", "KC")) and column not in keep_bb
    }

    return [
        column
        for column in df.columns
        if not column.startswith("label_") and column not in drop
    ]


def convert_value_to_scalar(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (list, tuple, np.ndarray)):
        return value[-1] if len(value) else np.nan

    text = str(value).strip()

    if text.startswith("[") and text.endswith("]"):
        parts = [part for part in text[1:-1].split(",") if part.strip()]
        return float(parts[-1]) if parts else np.nan

    return float(text)


def sanitize_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    max_nan_share: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    kept: list[str] = []
    dropped: dict[str, str] = {}

    for col in feature_cols:
        series = df[col]

        if not pd.api.types.is_numeric_dtype(series):
            try:
                series = series.map(convert_value_to_scalar)
            except (TypeError, ValueError):
                dropped[col] = "non-numeric"
                continue

        series = pd.to_numeric(series, errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan)

        nan_share = series.isna().mean()

        if nan_share >= max_nan_share or series.nunique(dropna=True) <= 1:
            dropped[col] = f"nan_share={nan_share:.2f}"
            continue

        df[col] = series.astype(float)
        kept.append(col)

    if dropped:
        print("\nDropped invalid feature columns:")
        for col, reason in dropped.items():
            print(f"  {col}: {reason}")

    if not kept:
        raise RuntimeError("No usable numeric feature columns were found.")

    return df, kept


# ------------------------------------------------------------------ labels
def generate_label(
    data: pd.DataFrame,
    lookahead: int,
    thresh,
    col: str = "Close",
) -> pd.Series:
    future_mean = (
        data[col]
        .shift(-lookahead)
        .rolling(window=lookahead, min_periods=lookahead)
        .mean()
    )

    pct_change = (future_mean - data[col]) / data[col]

    labels = pd.Series(
        np.select(
            [pct_change >= thresh, pct_change <= -thresh],
            [2, 1],
            default=0,
        ),
        index=data.index,
        dtype="float",
    )

    labels[future_mean.isna()] = np.nan
    return labels


def build_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    label_cols = []

    for lookahead in LOOKAHEADS:
        for threshold in THRESHOLDS:
            name = f"label_la{lookahead}_th{threshold:.3f}"
            df[name] = generate_label(
                df,
                lookahead=lookahead,
                thresh=threshold,
            )
            label_cols.append(name)

        for multiplier in ATR_MULTS:
            name = f"label_la{lookahead}_atr{multiplier:.1f}"
            df[name] = generate_label(
                df,
                lookahead=lookahead,
                thresh=multiplier * df["atr_pct"],
            )
            label_cols.append(name)

    return df, label_cols


def prepare_dataset(
    symbol: str,
    start: str = START,
    end=None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    raw = load_data(symbol=symbol, start=start, end=end)
    df = add_relative_features(add_indicators(raw))
    df, label_cols = build_labels(df)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[label_cols] = df[label_cols].astype(int)

    feature_cols = select_feature_cols(df)
    df, feature_cols = sanitize_feature_columns(df, feature_cols)

    return df, label_cols, feature_cols


def split_data(
    df: pd.DataFrame,
    train_share: float = 0.55,
    selection_share: float = 0.20,
    gap: int = GAP,
):
    """
    Split into:
        train: model training
        selection: label, threshold, and filter selection
        test: final untouched evaluation
    """
    n = len(df)
    a = int(n * train_share)
    b = int(n * (train_share + selection_share))

    train_df = df.iloc[:a]
    selection_df = df.iloc[a + gap : b].copy()
    test_df = df.iloc[b + gap :].copy()

    return train_df, selection_df, test_df


# ------------------------------------------------------------------ model
def make_model(**kwargs) -> XGBClassifier:
    params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        n_jobs=-1,
        eval_metric="mlogloss",
        random_state=SEED,
    )

    params.update(kwargs)
    return XGBClassifier(**params)


def train_label_model(X_train, y_train) -> XGBClassifier:
    model = make_model()
    model.fit(
        X_train,
        y_train,
        sample_weight=compute_sample_weight("balanced", y_train),
    )
    return model


# ------------------------------------------------------------------ persistence
def make_model_key(symbol: str, label_col: str, filter_name: str) -> str:
    raw_key = (
        f"{symbol}|{START}|{END}|{SEED}|{FEATURE_VERSION}|{label_col}|{filter_name}"
    )

    return hashlib.md5(raw_key.encode()).hexdigest()[:12]


def get_model_path(model_key: str) -> str:
    return os.path.join(MODEL_DIR, f"trendnet_long_{model_key}.joblib")


def save_model(model, metadata: dict, model_key: str) -> str:
    path = get_model_path(model_key)

    joblib.dump(
        {
            "model": model,
            "metadata": metadata,
        },
        path,
    )

    return path


# ------------------------------------------------------------------ filters
def apply_filter(
    probabilities: np.ndarray,
    threshold: float,
    df: pd.DataFrame,
    filter_name: str,
) -> np.ndarray:
    """
    Convert uptrend probabilities into long-only entry signals.
    """
    signals = probabilities >= threshold

    if filter_name == "adx":
        signals &= df["adx_14"].to_numpy() >= ADX_FILTER_MIN

    elif filter_name == "atr_rank":
        signals &= df["atr_rank_252"].to_numpy() >= ATR_RANK_FILTER_MIN

    elif filter_name == "adx_atr_rank":
        signals &= df["adx_14"].to_numpy() >= ADX_FILTER_MIN
        signals &= df["atr_rank_252"].to_numpy() >= ATR_RANK_FILTER_MIN

    elif filter_name != "none":
        raise ValueError(f"Unknown filter: {filter_name}")

    return signals.astype(int)


# ------------------------------------------------------------------ trade simulation
def simulate_long_trades(
    df: pd.DataFrame,
    long_signals: np.ndarray,
    stop_atr_mult: float = STOP_ATR_MULT,
    target_atr_mult: float = TARGET_ATR_MULT,
    max_hold_bars: int = MAX_HOLD_BARS,
    cost_bps: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Simulate long-only trades.

    A signal on bar t enters at the next bar's open.
    Exit occurs at:
      - stop-loss, if low <= stop
      - take-profit, if high >= target
      - close of the final allowed holding bar

    The stop is checked before the target, which is conservative when both
    could be hit in the same bar.
    """
    required = {"Open", "High", "Low", "Close", "atr_14"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns for trade simulation: {missing}")

    open_prices = df["Open"].to_numpy()
    high_prices = df["High"].to_numpy()
    low_prices = df["Low"].to_numpy()
    close_prices = df["Close"].to_numpy()
    atr = df["atr_14"].to_numpy()

    n = len(df)
    signals = np.asarray(long_signals, dtype=bool)

    trades = []
    busy_until = -1

    for entry_signal_bar in range(n - 1):
        if not signals[entry_signal_bar]:
            continue

        if entry_signal_bar <= busy_until:
            continue

        entry_bar = entry_signal_bar + 1

        if entry_bar >= n or np.isnan(atr[entry_signal_bar]):
            continue

        entry_price = open_prices[entry_bar]

        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        stop_price = entry_price - stop_atr_mult * atr[entry_signal_bar]
        target_price = entry_price + target_atr_mult * atr[entry_signal_bar]

        exit_bar = None
        exit_price = None
        exit_reason = None

        last_allowed_bar = min(entry_bar + max_hold_bars - 1, n - 1)

        for current_bar in range(entry_bar, last_allowed_bar + 1):
            if low_prices[current_bar] <= stop_price:
                exit_bar = current_bar
                exit_price = stop_price
                exit_reason = "stop"
                break

            if high_prices[current_bar] >= target_price:
                exit_bar = current_bar
                exit_price = target_price
                exit_reason = "target"
                break

            if current_bar == last_allowed_bar:
                exit_bar = current_bar
                exit_price = close_prices[current_bar]
                exit_reason = "max_hold"

        if exit_bar is None:
            continue

        gross_return = exit_price / entry_price - 1
        net_return = gross_return - (2 * cost_bps / 1e4)

        trades.append(
            {
                "signal_date": df.index[entry_signal_bar],
                "entry_date": df.index[entry_bar],
                "exit_date": df.index[exit_bar],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "exit_reason": exit_reason,
                "holding_bars": exit_bar - entry_bar + 1,
                "gross_return_%": gross_return * 100,
                "net_return_%": net_return * 100,
                "cost_bps": cost_bps,
            }
        )

        busy_until = exit_bar

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return trades_df, pd.Series(dtype=float)

    trades_df = trades_df.set_index("exit_date")
    trade_returns = trades_df["net_return_%"] / 100

    daily_equity = (1 + trade_returns).groupby(level=0).prod().cumprod()
    daily_equity = daily_equity.reindex(df.index).ffill().fillna(1.0)

    return trades_df, daily_equity


def strategy_stats(
    trades_df: pd.DataFrame,
    daily_equity: pd.Series,
    df: pd.DataFrame,
) -> dict:
    if trades_df.empty:
        return {
            "trades": 0,
            "total_ret_%": 0.0,
            "cagr_%": 0.0,
            "sharpe": np.nan,
            "max_dd_%": 0.0,
            "win_rate_%": np.nan,
            "profit_factor": np.nan,
            "avg_trade_%": np.nan,
            "median_trade_%": np.nan,
            "top3_trade_share_%": np.nan,
            "avg_hold_bars": np.nan,
            "time_in_mkt_%": 0.0,
        }

    daily_returns = daily_equity.pct_change().fillna(0)
    std = daily_returns.std(ddof=1)

    sharpe = daily_returns.mean() / std * np.sqrt(BARS_PER_YEAR) if std > 0 else np.nan

    running_max = daily_equity.cummax()
    max_dd = (daily_equity / running_max - 1).min() * 100

    wins = trades_df[trades_df["net_return_%"] > 0]
    losses = trades_df[trades_df["net_return_%"] <= 0]

    gross_profit = wins["net_return_%"].sum()
    gross_loss = abs(losses["net_return_%"].sum())

    years = len(df) / BARS_PER_YEAR
    total_return = daily_equity.iloc[-1] - 1

    cagr = (daily_equity.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else np.nan

    trade_returns = trades_df["net_return_%"]
    total_trade_return = trade_returns.sum()

    if total_trade_return != 0:
        top3_share = (
            trade_returns.sort_values(ascending=False).head(3).sum()
            / total_trade_return
            * 100
        )
    else:
        top3_share = np.nan

    return {
        "trades": len(trades_df),
        "total_ret_%": total_return * 100,
        "cagr_%": cagr,
        "sharpe": sharpe,
        "max_dd_%": max_dd,
        "win_rate_%": len(wins) / len(trades_df) * 100,
        "profit_factor": (gross_profit / gross_loss if gross_loss > 0 else np.nan),
        "avg_trade_%": trade_returns.mean(),
        "median_trade_%": trade_returns.median(),
        "top3_trade_share_%": top3_share,
        "avg_hold_bars": trades_df["holding_bars"].mean(),
        "time_in_mkt_%": trades_df["holding_bars"].sum() / len(df) * 100,
    }


def print_trade_quality(
    trades_df: pd.DataFrame,
    symbol: str,
    period: str,
):
    if trades_df.empty:
        print(f"\n{symbol} {period}: no trades")
        return

    returns = trades_df["net_return_%"]

    print(f"\n===== TRADE QUALITY: {symbol} {period} =====")
    print(f"Trades: {len(trades_df)}")
    print(f"Mean trade: {returns.mean():.3f}%")
    print(f"Median trade: {returns.median():.3f}%")
    print(f"Best trade: {returns.max():.3f}%")
    print(f"Worst trade: {returns.min():.3f}%")
    print(f"Standard deviation: {returns.std():.3f}%")

    total_return = returns.sum()

    if total_return != 0:
        top_3_share = (
            returns.sort_values(ascending=False).head(3).sum() / total_return * 100
        )

        print(f"Top-3 trades' share of total return: {top_3_share:.1f}%")
    else:
        print("Top-3 trades' share of total return: undefined")

    print("\nExit reasons:")
    print(trades_df["exit_reason"].value_counts().to_string())


# ------------------------------------------------------------------ selection
def select_strategy(
    train_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    symbol: str,
) -> tuple[str, float, str, pd.DataFrame, object]:
    """
    Choose label, threshold, and filter using the selection period only.
    """
    X_train = train_df[feature_cols]
    X_selection = selection_df[feature_cols]

    selection_returns = selection_df["ret_1"].to_numpy()

    rows = []

    for label_col in label_cols:
        if label_col not in train_df.columns:
            continue

        y_train = train_df[label_col]

        if y_train.nunique() < 3:
            continue

        model = train_label_model(X_train, y_train)
        probabilities = model.predict_proba(X_selection)[:, 2]

        for threshold in LONG_THRESHOLDS:
            for filter_name in [
                "none",
                "adx",
                "atr_rank",
                "adx_atr_rank",
            ]:
                signals = apply_filter(
                    probabilities,
                    threshold,
                    selection_df,
                    filter_name,
                )

                trades_df, daily_equity = simulate_long_trades(
                    selection_df,
                    signals,
                    cost_bps=COST_SCENARIOS_BPS[0],
                )

                stats = strategy_stats(
                    trades_df,
                    daily_equity,
                    selection_df,
                )

                rows.append(
                    {
                        "label": label_col,
                        "threshold": threshold,
                        "filter": filter_name,
                        **stats,
                    }
                )

    results = pd.DataFrame(rows)

    eligible = results[
        (results["trades"] >= MIN_SELECTION_TRADES) & (results["median_trade_%"] > 0)
    ]

    if eligible.empty:
        eligible = results[results["trades"] >= MIN_SELECTION_TRADES]

    if eligible.empty:
        raise RuntimeError(
            "No strategy configuration produced enough selection trades."
        )

    eligible = eligible.sort_values(
        ["sharpe", "median_trade_%", "trades"],
        ascending=[False, False, False],
    )

    best = eligible.iloc[0]

    selected_label = str(best["label"])
    selected_threshold = float(best["threshold"])
    selected_filter = str(best["filter"])

    path = os.path.join(
        OUT_DIR,
        f"strategy_selection_{symbol.replace('=', '_')}.csv",
    )
    results.to_csv(path, index=False)
    print("\nSaved", path)

    print("\n===== STRATEGY SELECTION (SELECTION PERIOD) =====")
    print(eligible.head(15).to_string(index=False))

    print(
        f"\nSelected label: {selected_label}\n"
        f"Selected threshold: {selected_threshold:.2f}\n"
        f"Selected filter: {selected_filter}"
    )

    final_model = train_label_model(
        X_train,
        train_df[selected_label],
    )

    return (
        selected_label,
        selected_threshold,
        selected_filter,
        results,
        final_model,
    )


# ------------------------------------------------------------------ evaluation
def evaluate_long_strategy(
    df_period: pd.DataFrame,
    X_period: pd.DataFrame,
    model,
    threshold: float,
    filter_name: str,
    period_name: str,
) -> dict:
    probabilities = model.predict_proba(X_period)[:, 2]

    signals = apply_filter(
        probabilities,
        threshold,
        df_period,
        filter_name,
    )

    results_by_cost = {}

    for cost_bps in COST_SCENARIOS_BPS:
        trades_df, daily_equity = simulate_long_trades(
            df_period,
            signals,
            cost_bps=cost_bps,
        )

        stats = strategy_stats(trades_df, daily_equity, df_period)
        stats["cost_bps"] = cost_bps
        results_by_cost[cost_bps] = {
            "stats": stats,
            "trades": trades_df,
            "equity": daily_equity,
        }

    rows = []

    for cost_bps, result in results_by_cost.items():
        rows.append(
            {
                "period": period_name,
                **result["stats"],
            }
        )

    table = pd.DataFrame(rows)

    path = os.path.join(
        OUT_DIR,
        f"long_strategy_{period_name.lower()}.csv",
    )
    table.to_csv(path, index=False)
    print("Saved", path)

    trades_path = os.path.join(
        OUT_DIR,
        f"long_trades_{period_name.lower()}.csv",
    )

    base_trades = results_by_cost[COST_SCENARIOS_BPS[0]]["trades"]

    if not base_trades.empty:
        base_trades.to_csv(trades_path)
        print("Saved", trades_path)

    symbol = df_period.attrs.get("symbol", "symbol")

    print_trade_quality(
        base_trades,
        symbol,
        period_name,
    )

    return {
        "threshold": threshold,
        "filter": filter_name,
        "signals": signals,
        "probabilities": probabilities,
        "results_by_cost": results_by_cost,
        "table": table,
    }


# ------------------------------------------------------------------ plots
def plot_equity(
    equity: pd.Series,
    buy_hold_equity: pd.Series,
    title: str,
    path: str,
):
    plt.figure(figsize=(10, 5))

    plt.plot(
        equity.index,
        equity,
        label="ML long-only",
        lw=1.5,
    )

    plt.plot(
        buy_hold_equity.index,
        buy_hold_equity,
        label="Buy & hold",
        lw=1.2,
        alpha=0.8,
    )

    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print("Saved", path)


def make_shap_safe_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(
                X[col].map(convert_value_to_scalar),
                errors="coerce",
            )

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    return X


def plot_shap(model, X, path_prefix: str):
    try:
        import shap

        X_shap = make_shap_safe_matrix(X)
        feature_names = list(X_shap.columns)

        booster = model.get_booster()
        dmatrix = xgb.DMatrix(
            X_shap,
            feature_names=feature_names,
        )

        contributions = booster.predict(
            dmatrix,
            pred_contribs=True,
        )

        contributions = np.asarray(contributions)

        if contributions.ndim != 3:
            raise RuntimeError(
                f"Unexpected SHAP contribution shape: {contributions.shape}"
            )

        if contributions.shape[1] == 3:
            class_axis = 1
        else:
            class_axis = 2

        for class_index in (1, 2):
            if class_axis == 1:
                values = contributions[:, class_index, :-1]
            else:
                values = contributions[:, :-1, class_index]

            shap.summary_plot(
                values,
                X_shap,
                feature_names=feature_names,
                show=False,
            )

            plt.tight_layout()
            plt.savefig(
                f"{path_prefix}_class{class_index}.png",
                dpi=120,
            )
            plt.close()

        print("Saved SHAP plots")

    except Exception as error:
        import traceback

        print("Skipping SHAP:", error)
        traceback.print_exc()


# ------------------------------------------------------------------ symbol runner
def run_symbol(symbol: str) -> dict:
    print(f"\n\n================ {symbol} ================")

    df, label_cols, feature_cols = prepare_dataset(
        symbol=symbol,
        start=START,
        end=END,
    )

    print(f"Rows: {len(df)} | features: {len(feature_cols)}")

    train_df, selection_df, test_df = split_data(df)

    train_df.attrs["symbol"] = symbol
    selection_df.attrs["symbol"] = symbol
    test_df.attrs["symbol"] = symbol

    print(
        "train/selection/test sizes:",
        len(train_df),
        len(selection_df),
        len(test_df),
    )

    selected_label, selected_threshold, selected_filter, selection_results, model = (
        select_strategy(
            train_df=train_df,
            selection_df=selection_df,
            feature_cols=feature_cols,
            label_cols=CANDIDATE_LABELS,
            symbol=symbol,
        )
    )

    X_train = train_df[feature_cols]
    X_selection = selection_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df[selected_label]
    y_selection = selection_df[selected_label]
    y_test = test_df[selected_label]

    model_key = make_model_key(
        symbol,
        selected_label,
        selected_filter,
    )

    save_model(
        model=model,
        metadata={
            "symbol": symbol,
            "label": selected_label,
            "threshold": selected_threshold,
            "filter": selected_filter,
            "feature_version": FEATURE_VERSION,
            "feature_columns": list(X_train.columns),
        },
        model_key=model_key,
    )

    print(f"Model saved: {get_model_path(model_key)}")

    selection_predictions = model.predict(X_selection)
    test_predictions = model.predict(X_test)

    print("\n===== SELECTION CLASSIFICATION =====")
    print(f"Accuracy: {accuracy_score(y_selection, selection_predictions):.4f}")
    print(
        f"Macro F1: {f1_score(y_selection, selection_predictions, average='macro'):.4f}"
    )
    print(f"MCC: {matthews_corrcoef(y_selection, selection_predictions):.4f}")

    print("\n===== FINAL TEST CLASSIFICATION =====")
    print(f"Accuracy: {accuracy_score(y_test, test_predictions):.4f}")
    print(f"Macro F1: {f1_score(y_test, test_predictions, average='macro'):.4f}")
    print(f"MCC: {matthews_corrcoef(y_test, test_predictions):.4f}")

    selection_evaluation = evaluate_long_strategy(
        df_period=selection_df,
        X_period=X_selection,
        model=model,
        threshold=selected_threshold,
        filter_name=selected_filter,
        period_name="SELECTION",
    )

    test_evaluation = evaluate_long_strategy(
        df_period=test_df,
        X_period=X_test,
        model=model,
        threshold=selected_threshold,
        filter_name=selected_filter,
        period_name="TEST",
    )

    print("\n===== SELECTION LONG STRATEGY =====")
    print(selection_evaluation["table"].to_string(index=False))

    print("\n===== FINAL TEST LONG STRATEGY =====")
    print(test_evaluation["table"].to_string(index=False))

    base_cost = COST_SCENARIOS_BPS[0]

    selection_equity = selection_evaluation["results_by_cost"][base_cost]["equity"]
    test_equity = test_evaluation["results_by_cost"][base_cost]["equity"]

    selection_buy_hold = selection_df["Close"] / selection_df["Close"].iloc[0]
    test_buy_hold = test_df["Close"] / test_df["Close"].iloc[0]

    plot_equity(
        selection_equity,
        selection_buy_hold,
        f"{symbol}: selection long-only equity",
        os.path.join(
            OUT_DIR,
            f"equity_long_selection_{symbol.replace('=', '_')}.png",
        ),
    )

    plot_equity(
        test_equity,
        test_buy_hold,
        f"{symbol}: final-test long-only equity",
        os.path.join(
            OUT_DIR,
            f"equity_long_test_{symbol.replace('=', '_')}.png",
        ),
    )

    plot_shap(
        model,
        X_selection,
        os.path.join(
            OUT_DIR,
            f"shap_{symbol.replace('=', '_')}",
        ),
    )

    test_stats = test_evaluation["results_by_cost"][base_cost]["stats"]

    return {
        "symbol": symbol,
        "label": selected_label,
        "threshold": selected_threshold,
        "filter": selected_filter,
        **test_stats,
    }


# ------------------------------------------------------------------ CLI
def run_cli_pipeline():
    results = []

    for symbol in [PRIMARY_SYMBOL, *ROBUSTNESS_SYMBOLS]:
        try:
            result = run_symbol(symbol=symbol)
            results.append(result)

        except Exception as error:
            print(f"\nSkipping {symbol}: {error}")

    comparison = pd.DataFrame(results)

    path = os.path.join(
        OUT_DIR,
        "multi_asset_long_strategy_comparison.csv",
    )
    comparison.to_csv(path, index=False)

    print("\n===== MULTI-ASSET LONG-ONLY COMPARISON =====")
    print(comparison.to_string(index=False))
    print("\nSaved", path)


if __name__ == "__main__":
    run_cli_pipeline()
