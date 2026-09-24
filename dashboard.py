# ================= dashboard.py =================
# รันด้วย: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import tensorflow as tf
from datetime import datetime
import MetaTrader5 as mt5

st.set_page_config(page_title="🥇 Gold AI Trader", layout="wide", page_icon="🥇")

# ================= Load Models =================
@st.cache_resource
def load_models():
    lstm = tf.keras.models.load_model('models/improved_lstm.keras')
    bilstm = tf.keras.models.load_model('models/improved_bilstm.keras')
    meta = joblib.load('models/improved_meta.pkl')
    scaler = joblib.load('models/improved_scaler.pkl')
    features = joblib.load('models/improved_features.pkl')
    config = joblib.load('models/improved_config.pkl')
    return lstm, bilstm, meta, scaler, features, config

# ================= Sidebar =================
st.sidebar.title("🥇 Gold AI Trader")
st.sidebar.markdown("---")

# MT5 Connection
if st.sidebar.button("🔌 Connect MT5"):
    try:
        if mt5.initialize():
            st.sidebar.success("✅ MT5 Connected")
            st.session_state['mt5_connected'] = True
        else:
            st.sidebar.error("❌ MT5 Connection Failed")
    except Exception as e:
        st.sidebar.error(f"❌ {e}")

symbol = st.sidebar.selectbox("Symbol", ["XAUUSD", "XAUUSDm"], index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["M5", "M15", "H1", "H4"], index=2)
demo_mode = st.sidebar.checkbox("🎭 Demo Mode", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
min_confidence = st.sidebar.slider("Min Confidence", 0.5, 0.9, 0.6, 0.05)

# ================= Main Dashboard =================
st.title("🥇 XAUUSD AI Trading Dashboard")
st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load models
try:
    lstm, bilstm, meta, scaler, features, config = load_models()
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# ================= Fetch Data =================
def fetch_data(symbol, timeframe, n_bars=500):
    tf_map = {'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15, 
              'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4}
    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, n_bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

if st.session_state.get('mt5_connected'):
    df = fetch_data(symbol, timeframe)
    
    # ================= Price Chart =================
    st.subheader(f"📈 {symbol} Price Chart")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price'
    ), row=1, col=1)
    
    # Volume
    colors = ['red' if c < o else 'green' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['time'], y=df['tick_volume'], 
                         marker_color=colors, name='Volume'), row=2, col=1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # ================= AI Signal =================
    st.subheader("🤖 AI Signal")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['close'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        # TODO: Run prediction here
        signal = "WAIT ⚪"
        st.metric("AI Signal", signal)
    
    with col3:
        confidence = 0.0
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col4:
        action = "No Trade"
        st.metric("Action", action)
    
    # ================= Recent Signals =================
    st.subheader("📋 Recent Signals")
    
    # Mock data - replace with actual predictions
    recent_signals = pd.DataFrame({
        'Time': df['time'].tail(10).values,
        'Price': df['close'].tail(10).values,
        'Signal': ['WAIT'] * 10,
        'Confidence': [0.0] * 10
    })
    st.dataframe(recent_signals, use_container_width=True)
    
    # ================= Trade History =================
    st.subheader("💼 Trade History")
    
    # Mock data
    trades = pd.DataFrame({
        'Time': [datetime.now()],
        'Type': ['BUY'],
        'Price': [current_price],
        'PnL': [0.0],
        'Status': ['Open']
    })
    st.dataframe(trades, use_container_width=True)
    
else:
    st.warning("⚠️ Please connect to MT5 first (see sidebar)")

# ================= Footer =================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<b>⚠️ Disclaimer:</b> This is for educational purposes only. 
Trading involves substantial risk of loss. Past performance does not guarantee future results.
</div>
""")