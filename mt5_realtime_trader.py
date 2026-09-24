import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import time
import os

# ================= IMPORT ICT FUNCTIONS =================
# (คัดลอกฟังก์ชัน detect_fvg, detect_order_block, detect_bos_choch, detect_candle_patterns จาก Cell 3)

def detect_fvg(df):
    high = df['Gold_High'].values
    low = df['Gold_Low'].values
    
    bullish_fvg = np.zeros(len(df))
    bearish_fvg = np.zeros(len(df))
    fvg_size = np.zeros(len(df))
    
    for i in range(2, len(df)):
        if low[i] > high[i-2]:
            bullish_fvg[i] = 1
            fvg_size[i] = low[i] - high[i-2]
        elif high[i] < low[i-2]:
            bearish_fvg[i] = 1
            fvg_size[i] = low[i-2] - high[i]
    
    return pd.DataFrame({
        'Bullish_FVG': bullish_fvg,
        'Bearish_FVG': bearish_fvg,
        'FVG_Size': fvg_size,
        'FVG_Net': bullish_fvg - bearish_fvg
    }, index=df.index)


def detect_order_block(df, lookback=10):
    high = df['Gold_High'].values
    low = df['Gold_Low'].values
    close = df['Gold_Close'].values
    open_price = df['Gold_Open'].values
    
    bullish_ob = np.zeros(len(df))
    bearish_ob = np.zeros(len(df))
    
    for i in range(lookback, len(df)):
        swing_high = np.max(high[i-lookback:i])
        swing_low = np.min(low[i-lookback:i])
        
        if close[i] > open_price[i] and close[i] > swing_high:
            for j in range(i-1, max(i-lookback, 0), -1):
                if close[j] < open_price[j]:
                    bullish_ob[i] = 1
                    break
        elif close[i] < open_price[i] and close[i] < swing_low:
            for j in range(i-1, max(i-lookback, 0), -1):
                if close[j] > open_price[j]:
                    bearish_ob[i] = 1
                    break
    
    return pd.DataFrame({
        'Bullish_OB': bullish_ob,
        'Bearish_OB': bearish_ob,
        'OB_Net': bullish_ob - bearish_ob
    }, index=df.index)


def detect_bos_choch(df, lookback=20):
    high = df['Gold_High'].values
    low = df['Gold_Low'].values
    close = df['Gold_Close'].values
    
    bos_bullish = np.zeros(len(df))
    bos_bearish = np.zeros(len(df))
    choch_bullish = np.zeros(len(df))
    choch_bearish = np.zeros(len(df))
    
    for i in range(lookback, len(df)):
        swing_high = np.max(high[i-lookback:i-1])
        swing_low = np.min(low[i-lookback:i-1])
        prev_trend = np.polyfit(range(lookback), close[i-lookback:i], 1)[0]
        
        if close[i] > swing_high and prev_trend > 0:
            bos_bullish[i] = 1
        elif close[i] < swing_low and prev_trend < 0:
            bos_bearish[i] = 1
        elif close[i] > swing_high and prev_trend < 0:
            choch_bullish[i] = 1
        elif close[i] < swing_low and prev_trend > 0:
            choch_bearish[i] = 1
    
    return pd.DataFrame({
        'BOS_Bullish': bos_bullish,
        'BOS_Bearish': bos_bearish,
        'CHoCH_Bullish': choch_bullish,
        'CHoCH_Bearish': choch_bearish,
        'Structure_Score': (bos_bullish + choch_bullish) - (bos_bearish + choch_bearish)
    }, index=df.index)


def detect_candle_patterns(df):
    open_price = df['Gold_Open'].values
    high = df['Gold_High'].values
    low = df['Gold_Low'].values
    close = df['Gold_Close'].values
    
    body = np.abs(close - open_price)
    upper_shadow = high - np.maximum(open_price, close)
    lower_shadow = np.minimum(open_price, close) - low
    total_range = high - low
    total_range = np.where(total_range == 0, 1e-10, total_range)
    
    patterns = {}
    
    bullish_engulfing = np.zeros(len(df))
    bearish_engulfing = np.zeros(len(df))
    for i in range(1, len(df)):
        if (close[i-1] < open_price[i-1] and close[i] > open_price[i] and 
            body[i] > body[i-1] and open_price[i] <= close[i-1] and close[i] >= open_price[i-1]):
            bullish_engulfing[i] = 1
        elif (close[i-1] > open_price[i-1] and close[i] < open_price[i] and 
              body[i] > body[i-1] and open_price[i] >= close[i-1] and close[i] <= open_price[i-1]):
            bearish_engulfing[i] = 1
    
    patterns['Bullish_Engulfing'] = bullish_engulfing
    patterns['Bearish_Engulfing'] = bearish_engulfing
    
    hammer = np.zeros(len(df))
    shooting_star = np.zeros(len(df))
    for i in range(len(df)):
        if (lower_shadow[i] > 2 * body[i] and upper_shadow[i] < body[i] and body[i] > 0):
            hammer[i] = 1
        elif (upper_shadow[i] > 2 * body[i] and lower_shadow[i] < body[i] and body[i] > 0):
            shooting_star[i] = 1
    
    patterns['Hammer'] = hammer
    patterns['Shooting_Star'] = shooting_star
    patterns['Doji'] = np.where(body < 0.1 * total_range, 1, 0)
    
    inside_bar = np.zeros(len(df))
    outside_bar = np.zeros(len(df))
    for i in range(1, len(df)):
        if high[i] < high[i-1] and low[i] > low[i-1]:
            inside_bar[i] = 1
        if high[i] > high[i-1] and low[i] < low[i-1]:
            outside_bar[i] = 1
    
    patterns['Inside_Bar'] = inside_bar
    patterns['Outside_Bar'] = outside_bar
    patterns['Body_Ratio'] = body / total_range
    patterns['Upper_Shadow_Ratio'] = upper_shadow / total_range
    patterns['Lower_Shadow_Ratio'] = lower_shadow / total_range
    
    return pd.DataFrame(patterns, index=df.index)


# ================= CREATE FEATURES FUNCTION =================
def create_features_realtime(df):
    """สร้าง features เหมือนตอนเทรน"""
    feat = df.copy()
    
    # Traditional Indicators
    feat['Gold_RSI'] = ta.momentum.RSIIndicator(feat['Gold_Close'], window=14).rsi()
    feat['Gold_MACD'] = ta.trend.MACD(feat['Gold_Close']).macd()
    feat['Gold_BB_High'] = ta.volatility.BollingerBands(feat['Gold_Close']).bollinger_hband()
    feat['Gold_BB_Low'] = ta.volatility.BollingerBands(feat['Gold_Close']).bollinger_lband()
    
    feat['Gold_ATR'] = ta.volatility.AverageTrueRange(
        high=feat['Gold_High'], low=feat['Gold_Low'],
        close=feat['Gold_Close'], window=14
    ).average_true_range()
    
    feat['Gold_Stoch'] = ta.momentum.StochasticOscillator(
        feat['Gold_High'], feat['Gold_Low'], feat['Gold_Close']
    ).stoch()
    
    feat['Gold_Vol_Change'] = feat['Gold_Volume'].pct_change().replace([np.inf, -np.inf], 0)
    
    for window in [10, 20, 50, 200]:
        feat[f'Gold_SMA_{window}'] = feat['Gold_Close'].rolling(window).mean()
        feat[f'Gold_EMA_{window}'] = feat['Gold_Close'].ewm(span=window, adjust=False).mean()
    
    # Cross-asset (ถ้าไม่มี DXY/VIX/SP500 ใน MT5 ให้ใช้ค่าคงที่หรือดึงจาก yfinance)
    feat['DXY_Change'] = 0  # หรือดึงจาก yfinance
    feat['VIX_Change'] = 0
    feat['Gold_DXY_Corr'] = 0
    
    # ICT Patterns
    feat = pd.concat([feat, detect_fvg(df)], axis=1)
    feat = pd.concat([feat, detect_order_block(df, lookback=10)], axis=1)
    feat = pd.concat([feat, detect_bos_choch(df, lookback=20)], axis=1)
    feat = pd.concat([feat, detect_candle_patterns(df)], axis=1)
    
    # Rolling ICT
    feat['FVG_Count_10'] = (feat['Bullish_FVG'] + feat['Bearish_FVG']).rolling(10).sum()
    feat['FVG_Net_10'] = feat['FVG_Net'].rolling(10).sum()
    feat['OB_Count_20'] = (feat['Bullish_OB'] + feat['Bearish_OB']).rolling(20).sum()
    feat['Structure_Events_30'] = (
        feat['BOS_Bullish'] + feat['BOS_Bearish'] +
        feat['CHoCH_Bullish'] + feat['CHoCH_Bearish']
    ).rolling(30).sum()
    
    # แก้ inf
    feat = feat.replace([np.inf, -np.inf], np.nan)
    
    return feat.dropna()


# ================= REAL-TIME TRADER CLASS =================
class GoldRealtimeTrader:
    def __init__(self, symbol='XAUUSD', demo_mode=True):
        self.symbol = symbol
        self.demo_mode = demo_mode
        
        # Load models
        print("📦 Loading models...")
        self.lstm = tf.keras.models.load_model('models/lstm_model.h5')
        self.bilstm = tf.keras.models.load_model('models/bilstm_model.h5')
        self.meta = joblib.load('models/meta_model.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.selected_features = joblib.load('models/feature_names.pkl')
        
        self.time_steps = 24
        
        # Connect MT5
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
        
        print(f"✅ MT5 Connected | Symbol: {symbol} | Demo Mode: {demo_mode}")
    
    def get_mt5_data(self, n_bars=300):
        """ดึงข้อมูลจาก MT5"""
        rates = mt5.copy_rates_from_pos(
            self.symbol,
            mt5.TIMEFRAME_H1,
            0,
            n_bars
        )
        
        if rates is None or len(rates) == 0:
            raise ValueError("No data received from MT5")
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        df.rename(columns={
            'open': 'Gold_Open',
            'high': 'Gold_High',
            'low': 'Gold_Low',
            'close': 'Gold_Close',
            'tick_volume': 'Gold_Volume'
        }, inplace=True)
        
        return df[['Gold_Open', 'Gold_High', 'Gold_Low', 'Gold_Close', 'Gold_Volume']]
    
    def predict_signal(self):
        """ทำนายสัญญาณ"""
        # ดึงข้อมูล
        df = self.get_mt5_data(n_bars=300)
        
        # สร้าง features
        df_features = create_features_realtime(df)
        
        # เลือก features ที่ต้องการ
        X = df_features[self.selected_features].iloc[-self.time_steps:]
        
        # สร้าง sequence
        X_seq = X.values.reshape(1, self.time_steps, len(self.selected_features))
        
        # Scale
        X_scaled = self.scaler.transform(X_seq.reshape(-1, X_seq.shape[2])).reshape(X_seq.shape)
        
        # Predict
        lstm_pred = self.lstm.predict(X_scaled, verbose=0)
        bilstm_pred = self.bilstm.predict(X_scaled, verbose=0)
        
        # Stacking
        stacked = np.hstack([lstm_pred, bilstm_pred])
        pred_class = self.meta.predict(stacked)[0]
        confidence = np.max(stacked)
        
        # Mapping: 0=Down, 1=Neutral, 2=Up
        signals = {0: 'SELL 🔴', 1: 'WAIT ⚪', 2: 'BUY 🟢'}
        
        return signals[pred_class], confidence, pred_class
    
    def run_once(self):
        """รันทำนายครั้งเดียว"""
        try:
            signal, confidence, signal_class = self.predict_signal()
            
            print(f"\n{'='*60}")
            print(f"📊 {self.symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            print(f"🎯 Signal: {signal}")
            print(f"📈 Confidence: {confidence:.2%}")
            
            if self.demo_mode:
                if signal_class == 0:
                    print("💡 [DEMO] Would SELL")
                elif signal_class == 2:
                    print("💡 [DEMO] Would BUY")
                else:
                    print("💡 [DEMO] No action (WAIT)")
            else:
                # ส่งออเดอร์จริง (ต้อง implement เพิ่ม)
                if signal_class != 1:
                    self.execute_trade(signal_class, confidence)
            
            return signal, confidence
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, 0
    
    def run_loop(self, interval_sec=3600):
        """รันลูปทุก interval"""
        print(f"\n🔄 Starting real-time loop (every {interval_sec}s)...")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.run_once()
                print(f"\n⏳ Waiting {interval_sec} seconds...")
                time.sleep(interval_sec)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
        finally:
            mt5.shutdown()
            print("✅ MT5 disconnected")


# ================= MAIN =================
if __name__ == "__main__":
    # เริ่มแบบ Demo Mode (ไม่ส่งออเดอร์จริง)
    trader = GoldRealtimeTrader(symbol='XAUUSD', demo_mode=True)
    
    # รันครั้งเดียวเพื่อทดสอบ
    trader.run_once()
    
    # หรือรันลูปทุก 1 ชั่วโมง
    # trader.run_loop(interval_sec=3600)