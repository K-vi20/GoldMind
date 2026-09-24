"""
Feature Engineering module for GoldMind (Gold XAU/USD).
Ensures stationarity, scale-invariance, and prevents lookahead bias / data leakage.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DEFAULT_N_FEATURES = 20
DEFAULT_TARGET_HORIZON = 1
DEFAULT_RANDOM_STATE = 42


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Standard Relative Strength Index (RSI) using Wilder's EMA smoothing.
    Properly handles zero-loss (RSI=100) and zero-gain (RSI=0).
    """
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()

    # Vectorized safe division
    rs = np.where(loss == 0, np.nan, gain / loss)
    rsi = np.where(
        loss == 0,
        np.where(gain == 0, 50.0, 100.0),
        np.where(gain == 0, 0.0, 100.0 - (100.0 / (1.0 + rs))),
    )
    return pd.Series(rsi, index=close.index)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) in dollar units.
    """
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build stationary and scale-invariant technical indicators from hourly OHLCV data.

    CRITICAL: Raw price levels (e.g. ma_20, raw close, bb_upper) are excluded
    because tree models (Random Forest, XGBoost) cannot extrapolate to unseen price regimes.
    """
    f = pd.DataFrame(index=df.index)
    close, high, low, open_px, vol = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]

    # 1. Multi-horizon momentum returns (Stationary % change over window w)
    for w in [1, 2, 3, 5, 8, 13, 21, 34]:
        f[f"ret_{w}"] = close.pct_change(w)
    f["ret_240"] = close.pct_change(240)  # ~10 trading days cumulative return

    # 2. Autoregressive past 1-bar returns (Genuine lag shifts, not redundant pct_change)
    ret_1h = close.pct_change(1)
    for lag in [1, 2, 3, 5, 10]:
        f[f"ret_lag_{lag}h"] = ret_1h.shift(lag)

    # 3. Moving averages as RELATIVE price difference (Stationary: Close / MA - 1)
    for w in [5, 10, 20, 50, 100, 200]:
        ma = close.rolling(w).mean()
        f[f"px_over_ma_{w}"] = (close / ma) - 1.0

    # 4. Volatility (Stationary rolling standard deviation of percentage returns)
    for w in [5, 10, 20, 50]:
        f[f"vol_{w}"] = ret_1h.rolling(w).std()

    # 5. Normalized ATR & Candle Range (Stationary: divided by Close/Open)
    atr_14 = _atr(df, 14)
    atr_50 = _atr(df, 50)
    f["atr_pct_14"] = atr_14 / close
    f["atr_pct_50"] = atr_50 / close
    f["hl_range"] = (high - low) / close
    f["oc_range"] = (close - open_px) / open_px

    # 6. Normalized Bollinger Bands (Stationary: %B and BandWidth)
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_width_abs = 4.0 * bb_std
    f["bb_pct_b"] = (close - (bb_ma - 2.0 * bb_std)) / bb_width_abs.replace(0, np.nan)
    f["bb_width"] = bb_width_abs / bb_ma

    # 7. Wilder's RSI (Stationary: bounded 0-100)
    for p in [7, 14, 21]:
        f[f"rsi_{p}"] = _rsi(close, p)

    # 8. Normalized MACD (Stationary: MACD / Close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_norm = (ema12 - ema26) / close
    macd_signal_norm = macd_norm.ewm(span=9, adjust=False).mean()
    f["macd_norm"] = macd_norm
    f["macd_signal_norm"] = macd_signal_norm
    f["macd_hist_norm"] = macd_norm - macd_signal_norm

    # 9. Volume indicators (Stationary: relative volume changes and z-scores)
    f["vol_change"] = vol.pct_change()
    vol_mean_20 = vol.rolling(20).mean()
    vol_std_20 = vol.rolling(20).std()
    f["vol_zscore_20"] = (vol - vol_mean_20) / vol_std_20.replace(0, np.nan)
    for w in [5, 10, 20]:
        f[f"vol_ratio_ma_{w}"] = (vol / vol.rolling(w).mean().replace(0, np.nan)) - 1.0

    # 10. Candle geometry (Stationary: normalized by Open price)
    f["candle_body"] = (close - open_px).abs() / open_px
    f["candle_upper_wick"] = (high - df[["Open", "Close"]].max(axis=1)) / open_px
    f["candle_lower_wick"] = (df[["Open", "Close"]].min(axis=1) - low) / open_px

    # 11. Trading Session indicators
    f["is_asian_session"] = df.index.hour.isin(range(0, 8)).astype(int)
    f["is_london_session"] = df.index.hour.isin(range(7, 16)).astype(int)
    f["is_ny_session"] = df.index.hour.isin(range(12, 21)).astype(int)

    # 12. Cyclical Time encodings & Market Gap awareness
    hours = df.index.hour
    dow = df.index.dayofweek
    f["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    f["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    f["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    f["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    # Market reopen gap flag (e.g. weekend gap > 2 hours)
    time_diff_hours = (df.index.to_series().diff().dt.total_seconds() / 3600.0).fillna(1.0)
    f["is_market_gap"] = (time_diff_hours > 2.0).astype(int)

    # External series (if any, as percentage returns)
    for col in df.columns:
        if col.endswith("_close"):
            f[f"{col}_ret_1"] = df[col].pct_change()
            f[f"{col}_ret_5"] = df[col].pct_change(5)

    return f


def make_target(df: pd.DataFrame, horizon: int = DEFAULT_TARGET_HORIZON, kind: str = "regression") -> pd.Series:
    """
    Generate forward target variable:
    - 'regression': Forward percentage return over `horizon` bars: (Close[t+h] - Close[t]) / Close[t].
    - 'classification': Direction of forward return (1 if return > 0 else 0).
    """
    fwd_ret = df["Close"].pct_change(horizon).shift(-horizon)
    if kind == "classification":
        return (fwd_ret > 0).astype(int)
    return fwd_ret


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n: int = DEFAULT_N_FEATURES,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[List[str], pd.Series]:
    """
    Rank features by RandomForest importance using TRAINING DATA ONLY.
    Never pass validation or test data to this function to avoid lookahead bias.
    """
    mask = X_train.notna().all(axis=1) & y_train.notna()
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train.loc[mask], y_train.loc[mask])
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    top_features = importances.head(n).index.tolist()
    return top_features, importances
