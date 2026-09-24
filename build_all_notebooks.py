"""
Script to build clean, professional Jupyter Notebooks for GoldMind:
1. Eda.ipynb       - Exploratory Data Analysis & Autocorrelation Clustering
2. Features.ipynb  - Stationary Feature Engineering & Feature Selection Demo
3. Train.ipynb     - ML Training (RF, XGB Regressor & Classifier) + Quant Backtest
"""

import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

# ==========================================
# 1. Eda.ipynb
# ==========================================
eda_cells = [
    md_cell("""# 📊 GoldMind - Exploratory Data Analysis (EDA)
Comprehensive data quality checks, price trends, return distributions, session volatility patterns, and autocorrelation/volatility clustering for XAU/USD (Gold)."""),

    code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "data/XAU_1m_data.csv"
OUT_DIR = "eda_output"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"EDA setup ready. Output directory: {OUT_DIR}")"""),

    code_cell("""# ---------- 1. Load Data & Resample to Hourly ----------
print(f"Loading {CSV_PATH} ...")
raw = pd.read_csv(CSV_PATH, parse_dates=["Date"])
raw = raw.sort_values("Date").reset_index(drop=True)

print(f"Loaded {len(raw):,} 1-minute bars from {raw['Date'].min()} to {raw['Date'].max()}")

# Resample to 1-Hour bars
df_1h = (
    raw.set_index("Date")
    .resample("1h")
    .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    .dropna()
)
print(f"Resampled to {len(df_1h):,} 1-hour bars.")"""),

    code_cell("""# ---------- 2. Data Quality & Integrity Checks ----------
print("=== Data Quality Checks ===")
print("Missing values in raw data:\\n", raw.isna().sum())
print("Duplicate timestamps:", raw["Date"].duplicated().sum())

# OHLC consistency check
broken_ohlc = raw[
    (raw["High"] < raw["Low"])
    | (raw["Open"] > raw["High"]) | (raw["Open"] < raw["Low"])
    | (raw["Close"] > raw["High"]) | (raw["Close"] < raw["Low"])
]
print(f"Broken OHLC bars: {len(broken_ohlc)}")

# Non-positive prices or zero volume
print("Non-positive prices:", (raw[["Open", "High", "Low", "Close"]] <= 0).sum().sum())
print("Zero-volume bars:", (raw["Volume"] == 0).sum())

# Weekend and exchange holiday gaps (> 6 hours)
gaps = raw["Date"].diff()
big_gaps = gaps[gaps > pd.Timedelta(hours=6)]
print(f"Weekend / Market gaps (>6h): {len(big_gaps)}  |  Largest gap: {gaps.max()}")"""),

    code_cell("""# ---------- 3. Plot Price & Volume Trends ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(df_1h.index, df_1h["Close"], color="#d4af37", lw=1.2, label="XAU/USD Close")
ax1.set_title("XAU/USD (Gold) Hourly Price Trend", fontsize=14, fontweight='bold')
ax1.set_ylabel("Price (USD)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper left")

ax2.bar(df_1h.index, df_1h["Volume"], color="#4682b4", alpha=0.6, width=0.03, label="Volume")
ax2.set_title("Trading Volume", fontsize=11)
ax2.set_ylabel("Volume")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper left")

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "price_volume_trend.png")
plt.savefig(fig_path, dpi=200)
plt.show()
print(f"Saved price & volume plot to {fig_path}")"""),

    code_cell("""# ---------- 4. Return Distribution & Outlier Analysis ----------
df_1h["ret"] = df_1h["Close"].pct_change()
clean_ret = df_1h["ret"].dropna()

mean_ret = clean_ret.mean()
std_ret = clean_ret.std()
skew_ret = clean_ret.skew()
kurt_ret = clean_ret.kurtosis()
var_95 = clean_ret.quantile(0.05)

print("=== 1-Hour Return Statistics ===")
print(f"Mean Return      : {mean_ret:.6f} ({mean_ret*100:.4f}%)")
print(f"Hourly Vol (Std) : {std_ret:.6f} ({std_ret*100:.4f}%)")
print(f"Skewness         : {skew_ret:.4f}")
print(f"Excess Kurtosis  : {kurt_ret:.4f} (Fat tails / Leptokurtic)")
print(f"VaR 95% (1-Hour) : {var_95*100:.3f}%")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(clean_ret, bins=100, density=True, alpha=0.65, color="#2b5c8f", label="Empirical Returns")

# Overlay normal distribution
x = np.linspace(clean_ret.min(), clean_ret.max(), 500)
norm_pdf = (1 / (std_ret * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_ret) / std_ret) ** 2)
ax.plot(x, norm_pdf, 'r--', lw=1.5, label="Normal Distribution")

ax.set_title(f"1-Hour Return Distribution (Kurtosis = {kurt_ret:.2f})", fontsize=12, fontweight='bold')
ax.set_xlabel("Return")
ax.set_ylabel("Density")
ax.set_xlim(-0.02, 0.02)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
dist_path = os.path.join(OUT_DIR, "return_distribution.png")
plt.savefig(dist_path, dpi=200)
plt.show()
print(f"Saved return distribution plot to {dist_path}")"""),

    code_cell("""# ---------- 5. Session & Hourly Volatility Patterns ----------
df_1h["Hour"] = df_1h.index.hour
hourly_vol = df_1h.groupby("Hour")["ret"].std() * 100

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(hourly_vol.index, hourly_vol.values, color="#e67e22", alpha=0.8, edgecolor="#b95e09")

# Highlight London & NY overlap
ax.axvspan(7, 16, color="blue", alpha=0.08, label="London Session (07-16 UTC)")
ax.axvspan(12, 20, color="green", alpha=0.08, label="NY Session (12-20 UTC)")

ax.set_title("Hourly Volatility Profile (Average % Standard Deviation by Hour UTC)", fontsize=12, fontweight='bold')
ax.set_xlabel("Hour of Day (UTC)")
ax.set_ylabel("Volatility (%)")
ax.set_xticks(range(24))
ax.grid(True, alpha=0.3, axis="y")
ax.legend(loc="upper left")

plt.tight_layout()
vol_path = os.path.join(OUT_DIR, "hourly_volatility.png")
plt.savefig(vol_path, dpi=200)
plt.show()
print(f"Saved hourly volatility profile to {vol_path}")"""),

    code_cell("""# ---------- 6. Autocorrelation & Volatility Clustering (ARCH Effect) ----------
lags = range(1, 25)
acf_ret = [clean_ret.autocorr(lag=l) for l in lags]
acf_vol = [clean_ret.abs().autocorr(lag=l) for l in lags]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(lags, acf_ret, color="#34495e", alpha=0.8)
ax1.axhline(0, color="black", lw=0.8)
ax1.axhline(1.96 / np.sqrt(len(clean_ret)), color="red", linestyle="--", alpha=0.6, label="95% CI")
ax1.axhline(-1.96 / np.sqrt(len(clean_ret)), color="red", linestyle="--", alpha=0.6)
ax1.set_title("ACF of Raw Returns (Price Random Walk Check)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Lag (Hours)")
ax1.set_ylabel("Autocorrelation")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.bar(lags, acf_vol, color="#e74c3c", alpha=0.8)
ax2.axhline(0, color="black", lw=0.8)
ax2.axhline(1.96 / np.sqrt(len(clean_ret)), color="red", linestyle="--", alpha=0.6, label="95% CI")
ax2.set_title("ACF of Absolute Returns (Volatility Clustering / ARCH)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Lag (Hours)")
ax2.set_ylabel("Autocorrelation of |Return|")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
acf_path = os.path.join(OUT_DIR, "autocorrelation_clustering.png")
plt.savefig(acf_path, dpi=200)
plt.show()
print(f"Saved autocorrelation plot to {acf_path}")"""),

    code_cell("""# ---------- 7. Yearly Summary Table ----------
df_1h["Year"] = df_1h.index.year
yearly = df_1h.groupby("Year").agg(
    Open=("Open", "first"),
    Close=("Close", "last"),
    High=("High", "max"),
    Low=("Low", "min"),
    Bars=("Close", "count"),
    Avg_Volume=("Volume", "mean"),
    Annual_Return=("Close", lambda s: (s.iloc[-1] / s.iloc[0]) - 1.0)
)
yearly["Annual_Return_Pct"] = yearly["Annual_Return"].map(lambda x: f"{x*100:+.2f}%")
print("\\n=== Yearly Summary ===")
print(yearly[["Open", "High", "Low", "Close", "Bars", "Annual_Return_Pct"]])

yearly_path = os.path.join(OUT_DIR, "yearly_summary.csv")
yearly.to_csv(yearly_path)
print(f"Saved yearly summary to {yearly_path}")""")
]

with open("Eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(eda_cells), f, indent=1)
print("Updated Eda.ipynb successfully.")

# ==========================================
# 2. Features.ipynb
# ==========================================
features_cells = [
    md_cell("""# 🛠️ GoldMind - Stationary Feature Engineering
This notebook generates 45+ scale-invariant, stationary technical indicators for XAU/USD.

### ⚠️ Critical Note on Stationarity in Machine Learning
Tree models (Random Forest, XGBoost) split on raw numeric thresholds. 
If raw price levels (e.g. `ma_200`, `close_lag_1h`, `bb_upper`) are used as features:
- In Gold's bull market (e.g. \\$2,000 in train $\\rightarrow$ \\$5,000 in test), **every test sample falls outside the training range**.
- The tree maps all test data to a single extreme leaf node, causing catastrophic bias.
- **Solution:** All indicators MUST be normalized relative to current price or bounded (e.g. `Close/MA - 1`, `%B`, normalized ATR, return lags, cyclical time encodings)."""),

    code_cell("""import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Import from our modular package
from src.features import build_features, make_target, select_top_features

CSV_PATH = "data/XAU_1m_data.csv"
OUT_DIR = "model_output"
os.makedirs(OUT_DIR, exist_ok=True)
print("Features setup ready.")"""),

    code_cell("""# ---------- 1. Load 1-min Data and Resample to 1-Hour ----------
print(f"Loading {CSV_PATH} ...")
raw = pd.read_csv(CSV_PATH, parse_dates=["Date"]).set_index("Date").sort_index()

df_1h = (
    raw.resample("1h")
    .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    .dropna()
)
print(f"Hourly dataset: {len(df_1h):,} bars ({df_1h.index.min()} to {df_1h.index.max()})")"""),

    code_cell("""# ---------- 2. Build Stationary Features ----------
print("Building stationary features via src.features.build_features() ...")
feats = build_features(df_1h)
y_reg = make_target(df_1h, horizon=1, kind="regression")
y_clf = make_target(df_1h, horizon=1, kind="classification")

print(f"Generated {feats.shape[1]} features across {len(feats):,} bars.")
print("\\nFeature categories created:")
print("- Multi-horizon returns: ret_1 to ret_34, ret_240")
print("- Autoregressive past 1-bar returns: ret_lag_1h to ret_lag_10h (shift-based)")
print("- Relative MA distance: px_over_ma_5 to px_over_ma_200 (Close / MA - 1)")
print("- Volatility & Normalized ATR: vol_5 to vol_50, atr_pct_14, atr_pct_50")
print("- Normalized Bollinger Bands: bb_pct_b, bb_width")
print("- Wilder's RSI: rsi_7, rsi_14, rsi_21")
print("- Normalized MACD: macd_norm, macd_signal_norm, macd_hist_norm")
print("- Volume z-score & ratios: vol_zscore_20, vol_ratio_ma_5, etc.")
print("- Candle geometry: candle_body, candle_upper_wick, candle_lower_wick")
print("- Session & Cyclical Time: hour_sin, hour_cos, dow_sin, dow_cos, is_market_gap")"""),

    code_cell("""# ---------- 3. Verify Stationarity & Clean Data ----------
# Combine features and regression target
data = feats.join(y_reg.rename("target")).dropna()
X = data.drop(columns=["target"])
y = data["target"]

print(f"Clean samples after dropping warmup rolling windows: {len(data):,}")

# Check that NO raw absolute price columns exist
raw_price_cols = [c for c in X.columns if c in ["Close", "High", "Low", "Open", "ma_20", "bb_upper", "bb_lower"]]
if len(raw_price_cols) == 0:
    print("✅ Stationarity check passed: No raw dollar price levels found in feature set.")
else:
    print("⚠️ WARNING: Found raw price columns:", raw_price_cols)

print("\\nSample feature summary:")
print(X[["ret_1", "ret_lag_1h", "px_over_ma_20", "atr_pct_14", "rsi_14", "bb_pct_b", "hour_sin"]].describe().round(4))"""),

    code_cell("""# ---------- 4. Feature Selection Demo (Strictly on Train Split) ----------
# To prevent lookahead bias (data leakage), feature selection MUST be fitted
# strictly on the training partition, NOT on the whole dataset!
n_total = len(X)
train_size = int(n_total * 0.72)
X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

print(f"Running feature selection on training split ({len(X_train):,} samples) ...")
top_features, importances = select_top_features(X_train, y_train, n=20)

print(f"\\nTop 15 Most Informative Features (strictly from train set):")
for rank, feat in enumerate(top_features[:15], 1):
    print(f" {rank:2d}. {feat:<22} (Importance: {importances[feat]:.4f})")"""),

    code_cell("""# ---------- 5. Visualize Top Feature Importances ----------
top_imp = importances.head(15).iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_imp.index, top_imp.values, color="#3498db", edgecolor="#217dbb")
ax.set_title("Top 15 Feature Importances (Random Forest on Train Set)", fontsize=12, fontweight='bold')
ax.set_xlabel("Relative Importance")
ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
feat_plot_path = os.path.join(OUT_DIR, "feature_importances_demo.png")
plt.savefig(feat_plot_path, dpi=200)
plt.show()
print(f"Saved feature importance plot to {feat_plot_path}")""")
]

with open("Features.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(features_cells), f, indent=1)
print("Updated Features.ipynb successfully.")

# ==========================================
# 3. Train.ipynb
# ==========================================
train_cells = [
    md_cell("""# 🤖 GoldMind - Model Training, Evaluation & Strategy Backtest
Complete machine learning pipeline for XAU/USD hourly forecasting:
1. Resample 1-minute data $\\rightarrow$ 1-hour bars
2. Build 45+ stationary, scale-invariant features
3. **Time-series chronological split FIRST (Train 72%, Val 8%, Test 20%)**
4. **Feature selection SECOND (fitted strictly on Train set - NO Data Leakage)**
5. Train **Random Forest Regressor**, **XGBoost Regressor**, and **XGBoost Classifier**
6. Evaluate Regression & Directional metrics (MAE, RMSE, $R^2$, Directional Accuracy)
7. **Trading Simulation & Backtest (PnL, Sharpe Ratio, Max Drawdown, Bar vs Trade Win Rate, Profit Factor with spread costs)**
8. Export all metrics, predictions, and artifacts"""),

    code_cell("""import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

from src.features import build_features, make_target, select_top_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "data/XAU_1m_data.csv"
OUT_DIR = "model_output"
os.makedirs(OUT_DIR, exist_ok=True)

N_FEATURES_SELECTED = 20
TARGET_HORIZON = 1
RANDOM_STATE = 42

TEST_SIZE = 0.20          # Last 20% of time as out-of-sample test
VAL_SIZE = 0.10           # 10% of train portion for XGB early stopping
SPREAD_PCT = 0.0002       # 0.02% (~$0.40 - $0.80 per gold trade) transaction cost / spread

print("Configuration initialized.")"""),

    code_cell("""# ---------- 1. Load & Resample Data ----------
print(f"Loading {CSV_PATH} ...")
raw = (
    pd.read_csv(CSV_PATH, parse_dates=["Date"])
    .set_index("Date")
    .sort_index()
)
print(f"Loaded {len(raw):,} 1-minute bars ({raw.index.min()} to {raw.index.max()})")

df_1h = (
    raw.resample("1h")
    .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    .dropna()
)
print(f"Resampled to {len(df_1h):,} hourly bars.")"""),

    code_cell("""# ---------- 2. Build Stationary Features & Targets ----------
print("Building stationary features ...")
feats = build_features(df_1h)
y_reg = make_target(df_1h, horizon=TARGET_HORIZON, kind="regression")
y_clf = make_target(df_1h, horizon=TARGET_HORIZON, kind="classification")

# Join and drop warmup NaNs
data = feats.join(y_reg.rename("target_reg")).join(y_clf.rename("target_clf")).dropna()
X_all = data.drop(columns=["target_reg", "target_clf"])
y_reg_all = data["target_reg"]
y_clf_all = data["target_clf"]
close_series = df_1h.loc[data.index, "Close"]

print(f"Clean samples: {len(data):,}  |  Features: {X_all.shape[1]}")"""),

    code_cell("""# ---------- 3. Time-Series Split FIRST (No Data Leakage) ----------
n = len(X_all)
test_start = int(n * (1 - TEST_SIZE))
val_start = int(test_start * (1 - VAL_SIZE))

X_train, y_train_reg, y_train_clf = X_all.iloc[:val_start], y_reg_all.iloc[:val_start], y_clf_all.iloc[:val_start]
X_val, y_val_reg, y_val_clf = X_all.iloc[val_start:test_start], y_reg_all.iloc[val_start:test_start], y_clf_all.iloc[val_start:test_start]
X_test, y_test_reg, y_test_clf = X_all.iloc[test_start:], y_reg_all.iloc[test_start:], y_clf_all.iloc[test_start:]

test_close = close_series.iloc[test_start:]

print("Split sizes (Chronological):")
print(f"  Train : {len(X_train):,} bars ({X_train.index.min()} -> {X_train.index.max()})")
print(f"  Val   : {len(X_val):,} bars ({X_val.index.min()} -> {X_val.index.max()})")
print(f"  Test  : {len(X_test):,} bars ({X_test.index.min()} -> {X_test.index.max()})")"""),

    code_cell("""# ---------- 4. Feature Selection on Train Set Only ----------
print(f"Selecting top {N_FEATURES_SELECTED} features using Train Set ONLY ...")
top_feats, train_importances = select_top_features(
    X_train, y_train_reg, n=N_FEATURES_SELECTED, random_state=RANDOM_STATE
)
print("Top selected features:", top_feats[:10])

# Filter datasets to selected features
X_train_sel = X_train[top_feats]
X_val_sel = X_val[top_feats]
X_test_sel = X_test[top_feats]"""),

    code_cell("""# ---------- 5. Model Training ----------
print("Training models ...")

# 1. Random Forest Regressor
print("Training Random Forest Regressor ...")
rf = RandomForestRegressor(
    n_estimators=250,
    max_depth=10,
    min_samples_leaf=10,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train_sel, y_train_reg)
rf_pred = rf.predict(X_test_sel)

# 2. XGBoost Regressor
print("Training XGBoost Regressor (with early stopping) ...")
xgb_reg = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=40,
    eval_metric="rmse",
)
xgb_reg.fit(
    X_train_sel, y_train_reg,
    eval_set=[(X_val_sel, y_val_reg)],
    verbose=False,
)
xgb_reg_pred = xgb_reg.predict(X_test_sel)
print(f"XGBoost Regressor best iteration: {xgb_reg.best_iteration}")

# 3. XGBoost Classifier (Directional Probability)
print("Training XGBoost Classifier (Directional Probability) ...")
xgb_clf = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=40,
    eval_metric="logloss",
)
xgb_clf.fit(
    X_train_sel, y_train_clf,
    eval_set=[(X_val_sel, y_val_clf)],
    verbose=False,
)
xgb_clf_probs = xgb_clf.predict_proba(X_test_sel)[:, 1]
print(f"XGBoost Classifier best iteration: {xgb_clf.best_iteration}")
print("Training complete.")"""),

    code_cell("""# ---------- 6. Evaluation Metrics ----------
def calc_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "DirAcc": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "PctPredNeg": float(np.mean(y_pred < 0)),
    }

rf_metrics = calc_metrics(y_test_reg.values, rf_pred)
xgb_reg_metrics = calc_metrics(y_test_reg.values, xgb_reg_pred)

comparison = pd.DataFrame({"RandomForest": rf_metrics, "XGBoost_Reg": xgb_reg_metrics}).T
comparison_display = comparison.copy()
comparison_display["MAE"] = comparison_display["MAE"].map(lambda x: f"{x:.6f}")
comparison_display["RMSE"] = comparison_display["RMSE"].map(lambda x: f"{x:.6f}")
comparison_display["R2"] = comparison_display["R2"].map(lambda x: f"{x:.5f}")
comparison_display["DirAcc"] = comparison_display["DirAcc"].map(lambda x: f"{x*100:.2f}%")
comparison_display["PctPredNeg"] = comparison_display["PctPredNeg"].map(lambda x: f"{x*100:.2f}%")

print("\\n" + "=" * 65)
print("OUT-OF-SAMPLE TEST SET METRICS (Stationary & No Leakage)")
print("=" * 65)
print(comparison_display.to_string())

clf_acc = accuracy_score(y_test_clf.values, (xgb_clf_probs > 0.5).astype(int))
clf_auc = roc_auc_score(y_test_clf.values, xgb_clf_probs)
print(f"\\nXGBoost Classifier Directional Accuracy: {clf_acc*100:.2f}%  |  ROC-AUC: {clf_auc:.4f}")
actual_neg_pct = np.mean(y_test_reg.values < 0) * 100
print(f"Actual test market negative returns: {actual_neg_pct:.2f}%")"""),

    code_cell("""# ---------- 7. Trading Strategy Backtest & Simulation ----------
def run_backtest(y_true, signals, spread_cost=SPREAD_PCT):
    pos_changes = np.abs(np.diff(signals, prepend=0))
    gross_returns = signals * y_true
    net_returns = gross_returns - (pos_changes * spread_cost)
    equity_curve = (1.0 + net_returns).cumprod()
    total_return = (equity_curve[-1] - 1.0) * 100
    
    # Annualized Sharpe (hourly: ~6,000 trading hours per year)
    ann_factor = np.sqrt(6000)
    sharpe = (np.mean(net_returns) / (np.std(net_returns) + 1e-9)) * ann_factor
    
    # Max Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / peak
    max_dd = np.max(drawdowns) * 100
    
    # Hourly win rate (percentage of active bars with positive return)
    active_mask = (signals != 0)
    hourly_win_rate = np.mean(gross_returns[active_mask] > 0) * 100 if np.sum(active_mask) > 0 else 0.0

    # Trade-level statistics (from position entry to exit/flip)
    trades = []
    curr_pos = 0
    trade_ret = 0.0
    for s, r, chg in zip(signals, y_true, pos_changes):
        if chg > 0:
            if curr_pos != 0:
                trades.append(trade_ret)
                trade_ret = 0.0
            curr_pos = s
        if curr_pos != 0:
            trade_ret += (curr_pos * r) - (spread_cost if chg > 0 else 0)
    if curr_pos != 0:
        trades.append(trade_ret)

    trades = np.array(trades)
    trade_win_rate = np.mean(trades > 0) * 100 if len(trades) > 0 else 0.0
    gains = trades[trades > 0].sum() if np.any(trades > 0) else 0.0
    losses = np.abs(trades[trades < 0].sum()) if np.any(trades < 0) else 1e-9
    profit_factor = gains / losses if losses > 0 else np.nan

    return {
        "Total Return (%)": total_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd,
        "Hourly Win Rate (%)": hourly_win_rate,
        "Trade Win Rate (%)": trade_win_rate,
        "Profit Factor": profit_factor,
        "Total Trades": int(np.sum(pos_changes > 0)),
        "Equity Curve": equity_curve,
    }

# 1. RF Regressor Signals
sig_rf = np.where(rf_pred > 0, 1, -1)

# 2. XGB Regressor Signals
sig_xgb_reg = np.where(xgb_reg_pred > 0, 1, -1)

# 3. XGB Classifier Conviction Filter (Trade when probability > 0.52 or < 0.48, else stay flat)
sig_xgb_clf = np.zeros(len(xgb_clf_probs))
sig_xgb_clf[xgb_clf_probs > 0.52] = 1
sig_xgb_clf[xgb_clf_probs < 0.48] = -1

y_test_arr = y_test_reg.values
rf_bt = run_backtest(y_test_arr, sig_rf)
xgb_reg_bt = run_backtest(y_test_arr, sig_xgb_reg)
xgb_clf_bt = run_backtest(y_test_arr, sig_xgb_clf)

# Buy & Hold Benchmark
bnh_equity = (1.0 + y_test_arr).cumprod()
bnh_total = (bnh_equity[-1] - 1.0) * 100

bt_summary = pd.DataFrame({
    "RandomForest": {k: v for k, v in rf_bt.items() if k != "Equity Curve"},
    "XGBoost_Reg": {k: v for k, v in xgb_reg_bt.items() if k != "Equity Curve"},
    "XGBoost_Clf_Conviction": {k: v for k, v in xgb_clf_bt.items() if k != "Equity Curve"},
}).T
print("\\n" + "=" * 70)
print("TRADING BACKTEST RESULTS (Out-of-sample with transaction costs)")
print("=" * 70)
print(bt_summary.round(2).to_string())
print(f"Buy & Hold Return: {bnh_total:.2f}%")

# Plot Equity Curves
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test_reg.index, rf_bt["Equity Curve"], label=f"Random Forest (Sharpe: {rf_bt['Sharpe Ratio']:.2f})", color="#2980b9", lw=1.5)
ax.plot(y_test_reg.index, xgb_reg_bt["Equity Curve"], label=f"XGBoost Regressor (Sharpe: {xgb_reg_bt['Sharpe Ratio']:.2f})", color="#27ae60", lw=1.5)
ax.plot(y_test_reg.index, xgb_clf_bt["Equity Curve"], label=f"XGBoost Classifier Conviction (Sharpe: {xgb_clf_bt['Sharpe Ratio']:.2f})", color="#8e44ad", lw=1.5)
ax.plot(y_test_reg.index, bnh_equity, label=f"Buy & Hold ({bnh_total:.1f}%)", color="#7f8c8d", linestyle="--", alpha=0.7)

ax.set_title("Out-of-Sample Trading Strategy Equity Curve (XAU/USD Hourly)", fontsize=13, fontweight='bold')
ax.set_ylabel("Portfolio Value (Base = 1.0)")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")

plt.tight_layout()
curve_path = os.path.join(OUT_DIR, "strategy_equity_curve.png")
plt.savefig(curve_path, dpi=200)
plt.show()
print(f"Saved equity curve plot to {curve_path}")"""),

    code_cell("""# ---------- 8. Export Artifacts ----------
comparison_display.to_csv(os.path.join(OUT_DIR, "metrics_comparison.csv"))
bt_summary.to_csv(os.path.join(OUT_DIR, "backtest_summary.csv"))

rf_imp = pd.Series(rf.feature_importances_, index=top_feats)
xgb_imp = pd.Series(xgb_reg.feature_importances_, index=top_feats)
imp_df = pd.DataFrame({"RF": rf_imp, "XGB": xgb_imp}).sort_values("XGB", ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"))

pd.Series(top_feats).to_csv(os.path.join(OUT_DIR, "selected_features.csv"), index=False, header=["feature"])

pred_df = pd.DataFrame(
    {
        "y_true": y_test_reg.values,
        "rf_pred": rf_pred,
        "xgb_reg_pred": xgb_reg_pred,
        "xgb_clf_prob": xgb_clf_probs,
        "close": test_close.values,
    },
    index=y_test_reg.index,
)
pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"))

print(f"\\n✅ All artifacts successfully exported to ./{OUT_DIR}/")
print("Done.")""")
]

with open("Train.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(train_cells), f, indent=1)
print("Updated Train.ipynb successfully.")
