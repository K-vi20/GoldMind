import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import time
import os
import yfinance as yf
import csv
from pathlib import Path

# ================= ICT FUNCTIONS =================

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


# ================= CREATE FEATURES =================
def create_features_realtime(df, cross_asset_df=None):
    feat = df.copy()
    
    # Traditional Indicators
    feat['Gold_RSI'] = ta.momentum.RSIIndicator(feat['Gold_Close'], window=14).rsi()
    feat['Gold_MACD'] = ta.trend.MACD(feat['Gold_Close']).macd()
    feat['Gold_MACD_Signal'] = ta.trend.MACD(feat['Gold_Close']).macd_signal()
    
    bb = ta.volatility.BollingerBands(feat['Gold_Close'])
    feat['Gold_BB_Width'] = bb.bollinger_hband() - bb.bollinger_lband()
    
    feat['Gold_ATR'] = ta.volatility.AverageTrueRange(
        high=feat['Gold_High'], low=feat['Gold_Low'],
        close=feat['Gold_Close'], window=14
    ).average_true_range()
    feat['Gold_ATR_Pct'] = feat['Gold_ATR'] / feat['Gold_Close'] * 100
    
    feat['Gold_Stoch'] = ta.momentum.StochasticOscillator(
        feat['Gold_High'], feat['Gold_Low'], feat['Gold_Close']
    ).stoch()
    
    # Returns
    for window in [4, 12, 24, 72]:
        feat[f'Gold_Return_{window}'] = feat['Gold_Close'].pct_change(window)
    
    # Cross-Asset Features
    if cross_asset_df is not None and len(cross_asset_df) > 0:
        if cross_asset_df.index.tz is not None:
            print("   ⚠️ Fixing timezone mismatch in cross-asset data...")
            cross_asset_df.index = cross_asset_df.index.tz_localize(None)
        
        cross_asset_df = cross_asset_df.reindex(df.index, method='ffill')
        
        feat['DXY_Return_12'] = cross_asset_df['DXY_Close'].pct_change(12)
        feat['VIX_Return_12'] = cross_asset_df['VIX_Close'].pct_change(12)
        feat['SP500_Return_12'] = cross_asset_df['SP500_Close'].pct_change(12)
    else:
        feat['DXY_Return_12'] = 0.0
        feat['VIX_Return_12'] = 0.0
        feat['SP500_Return_12'] = 0.0
    
    # ICT Patterns
    feat = pd.concat([feat, detect_fvg(df)], axis=1)
    feat = pd.concat([feat, detect_order_block(df, lookback=10)], axis=1)
    feat = pd.concat([feat, detect_bos_choch(df, lookback=20)], axis=1)
    feat = pd.concat([feat, detect_candle_patterns(df)], axis=1)
    
    # Rolling ICT
    feat['FVG_Net_10'] = feat['FVG_Net'].rolling(10, min_periods=1).sum()
    feat['OB_Net_20'] = (feat['Bullish_OB'] - feat['Bearish_OB']).rolling(20, min_periods=1).sum()
    feat['Structure_Score_30'] = feat['Structure_Score'].rolling(30, min_periods=1).sum()
    
    # Clean inf/nan
    feat = feat.replace([np.inf, -np.inf], np.nan)
    
    return feat


def get_cross_asset_data(n_bars=300):
    try:
        tickers = {'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'SP500': '^GSPC'}
        data_dict = {}
        first_index = None
        
        for name, ticker in tickers.items():
            df = yf.download(ticker, period='30d', interval='1h', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                continue
            if first_index is None:
                first_index = df.index
            data_dict[f'{name}_Close'] = df['Close'].squeeze()
        
        if data_dict:
            result = pd.DataFrame(data_dict, index=first_index).ffill()
            print(f"   ✅ Cross-asset data: {len(result)} rows")
            return result
        else:
            print("   ⚠️ No cross-asset data available, using zeros")
            return None
    except Exception as e:
        print(f"   ️ Warning: Could not fetch cross-asset data: {e}")
        return None


# ================= REAL-TIME TRADER CLASS =================
class GoldRealtimeTrader:
    def __init__(self, symbol='XAUUSD', demo_mode=True, models_dir='models', 
                 interval_sec=3600, min_confidence=0.55):
        self.symbol = symbol
        self.demo_mode = demo_mode
        self.models_dir = models_dir
        self.interval_sec = interval_sec
        self.min_confidence = min_confidence
        
        # Statistics tracking
        self.stats = {
            'total_predictions': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'wait_signals': 0,
            'trades_executed': 0,
            'start_time': datetime.now()
        }
        
        # Load improved models
        print("📦 Loading improved models...")
        
        lstm_path = f'{models_dir}/improved_lstm.keras'
        bilstm_path = f'{models_dir}/improved_bilstm.keras'
        meta_path = f'{models_dir}/improved_meta.pkl'
        scaler_path = f'{models_dir}/improved_scaler.pkl'
        features_path = f'{models_dir}/improved_features.pkl'
        config_path = f'{models_dir}/improved_config.pkl'
        
        for path, name in [(lstm_path, 'LSTM'), (bilstm_path, 'BiLSTM'), 
                           (meta_path, 'Meta'), (scaler_path, 'Scaler'),
                           (features_path, 'Features')]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ {name} not found: {path}")
        
        self.lstm = tf.keras.models.load_model(lstm_path)
        self.bilstm = tf.keras.models.load_model(bilstm_path)
        self.meta = joblib.load(meta_path)
        self.scaler = joblib.load(scaler_path)
        self.selected_features = joblib.load(features_path)
        
        if os.path.exists(config_path):
            config = joblib.load(config_path)
            self.time_steps = config.get('time_steps', 48)
            self.threshold = config.get('threshold', 0.003)
        else:
            self.time_steps = 48
            self.threshold = 0.003
        
        print(f"✅ Models loaded:")
        print(f"   LSTM:      {lstm_path}")
        print(f"   BiLSTM:    {bilstm_path}")
        print(f"   Meta:      {meta_path}")
        print(f"   Features:  {len(self.selected_features)}")
        print(f"   Time Steps: {self.time_steps}")
        
        # Connect MT5
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
        
        print(f"✅ MT5 Connected | Symbol: {symbol} | Demo Mode: {demo_mode}")
        print(f"⏱️  Check interval: {interval_sec}s ({interval_sec/3600:.1f}h)")
        print(f" Min confidence: {min_confidence:.0%}")
        
        # Setup log file
        self.log_file = Path('signal_log.csv')
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'symbol', 'signal', 'confidence', 
                               'price', 'prob_down', 'prob_neutral', 'prob_up', 'action'])
    
    def get_mt5_data(self, n_bars=500):
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
    
    def is_market_open(self):
        """ตรวจสอบว่าตลาดเปิดอยู่ไหม (ทองเปิด 23 ชม./วัน ปิดเสาร์-อาทิตย์)"""
        now = datetime.now()
        # ตลาดทองปิด: วันเสาร์ 00:00 - วันอาทิตย์ 23:00 (เวลา UTC)
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        if weekday == 5:  # Saturday
            return False
        if weekday == 6 and now.hour < 23:  # Sunday before 23:00
            return False
        
        return True
    
    def predict_signal(self):
        print("\n🔮 Starting prediction...")
        
        print("📊 Fetching MT5 data...")
        df = self.get_mt5_data(n_bars=500)
        print(f"   Got {len(df)} bars from MT5")
        
        print("📊 Fetching cross-asset data...")
        cross_asset_df = get_cross_asset_data()
        
        print("🔧 Creating features...")
        df_features = create_features_realtime(df, cross_asset_df)
        print(f"   Features created: {len(df_features)} rows")
        
        # Check features
        available_features = [f for f in self.selected_features if f in df_features.columns]
        missing_features = [f for f in self.selected_features if f not in df_features.columns]
        
        if len(available_features) == 0:
            raise ValueError("No features available!")
        
        if len(missing_features) > 0:
            print(f"   ⚠️ Filling {len(missing_features)} missing features with 0...")
            for feat in missing_features:
                df_features[feat] = 0.0
        
        X = df_features[self.selected_features].iloc[-self.time_steps:]
        
        if len(X) < self.time_steps:
            raise ValueError(f"Not enough data! Need {self.time_steps} rows, got {len(X)}")
        
        X_seq = X.values.reshape(1, self.time_steps, len(self.selected_features))
        
        if not np.all(np.isfinite(X_seq)):
            print("   ⚠️ Found NaN/Inf, replacing with 0...")
            X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_scaled = self.scaler.transform(X_seq.reshape(-1, X_seq.shape[2])).reshape(X_seq.shape)
        
        print("🧠 Running models...")
        lstm_pred = self.lstm.predict(X_scaled, verbose=0)
        bilstm_pred = self.bilstm.predict(X_scaled, verbose=0)
        
        stacked = np.hstack([lstm_pred, bilstm_pred])
        pred_class = int(self.meta.predict(stacked)[0])
        confidence = float(np.max(stacked))
        
        signals = {0: 'SELL 🔴', 1: 'WAIT ', 2: 'BUY 🟢'}
        
        print(f"\n✅ Prediction complete!")
        print(f"   Signal: {signals[pred_class]}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Probabilities: Down={stacked[0][0]:.3f}, Neutral={stacked[0][1]:.3f}, Up={stacked[0][2]:.3f}")
        
        return signals[pred_class], confidence, pred_class, stacked[0], df['Gold_Close'].iloc[-1]
    
    def log_signal(self, signal, confidence, pred_class, probabilities, price, action):
        """บันทึกสัญญาณลง CSV"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                self.symbol,
                signal,
                f"{confidence:.4f}",
                f"{price:.2f}",
                f"{probabilities[0]:.4f}",
                f"{probabilities[1]:.4f}",
                f"{probabilities[2]:.4f}",
                action
            ])
    
    def execute_trade(self, signal_class, confidence, price):
        """ส่งออเดอร์ไปที่ MT5"""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"❌ Symbol {self.symbol} not found")
            return None
        
        if not symbol_info.visible:
            mt5.symbol_select(self.symbol, True)
        
        tick = mt5.symbol_info_tick(self.symbol)
        lot = 0.01
        
        if signal_class == 2:  # BUY
            order_type = mt5.ORDER_TYPE_BUY
            entry_price = tick.ask
            sl = entry_price - 10.0
            tp = entry_price + 20.0
        else:  # SELL
            order_type = mt5.ORDER_TYPE_SELL
            entry_price = tick.bid
            sl = entry_price + 10.0
            tp = entry_price - 20.0
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": order_type,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": f"AI_{signal_class}_conf{confidence:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Order executed successfully!")
            print(f"   Deal: {result.deal}")
            print(f"   Price: {entry_price:.2f}")
            print(f"   SL: {sl:.2f}, TP: {tp:.2f}")
            self.stats['trades_executed'] += 1
        else:
            print(f"❌ Order failed: {result.retcode} - {result.comment}")
        
        return result
    
    def run_once(self):
        """รันทำนายครั้งเดียว"""
        try:
            # ตรวจสอบเวลาตลาด
            if not self.is_market_open():
                print(f"\n️  Market closed. Waiting {self.interval_sec}s...")
                return None, 0
            
            signal, confidence, signal_class, probabilities, price = self.predict_signal()
            
            # Update stats
            self.stats['total_predictions'] += 1
            if signal_class == 0:
                self.stats['sell_signals'] += 1
            elif signal_class == 1:
                self.stats['wait_signals'] += 1
            elif signal_class == 2:
                self.stats['buy_signals'] += 1
            
            print(f"\n{'='*70}")
            print(f"📊 {self.symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")
            print(f" Signal: {signal}")
            print(f"📈 Confidence: {confidence:.2%}")
            print(f"💰 Current Price: ${price:.2f}")
            print(f"📊 Stats: {self.stats['total_predictions']} predictions | "
                  f"{self.stats['buy_signals']} BUY | {self.stats['sell_signals']} SELL | "
                  f"{self.stats['wait_signals']} WAIT")
            
            # ตัดสินใจว่าจะเทรดไหม
            action = "NO_ACTION"
            if confidence >= self.min_confidence and signal_class != 1:
                if self.demo_mode:
                    action = f"[DEMO] Would {'BUY' if signal_class == 2 else 'SELL'}"
                    print(f"💡 {action}")
                else:
                    print(f" Executing trade...")
                    self.execute_trade(signal_class, confidence, price)
                    action = "TRADE_EXECUTED"
            else:
                if confidence < self.min_confidence:
                    print(f"⏸️  Confidence too low ({confidence:.2%} < {self.min_confidence:.0%})")
                    action = "LOW_CONFIDENCE"
                else:
                    print(f"⏸️  No action (WAIT)")
                    action = "WAIT_SIGNAL"
            
            # บันทึกสัญญาณ
            self.log_signal(signal, confidence, signal_class, probabilities, price, action)
            print(f"📝 Signal logged to {self.log_file}")
            
            return signal, confidence
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    def print_summary(self):
        """พิมพ์สรุปสถิติ"""
        runtime = datetime.now() - self.stats['start_time']
        hours = runtime.total_seconds() / 3600
        
        print(f"\n{'='*70}")
        print(f"📊 TRADING SESSION SUMMARY")
        print(f"{'='*70}")
        print(f"️  Runtime: {runtime}")
        print(f"🔮 Total Predictions: {self.stats['total_predictions']}")
        print(f"🟢 BUY Signals: {self.stats['buy_signals']}")
        print(f" SELL Signals: {self.stats['sell_signals']}")
        print(f"⚪ WAIT Signals: {self.stats['wait_signals']}")
        print(f"💼 Trades Executed: {self.stats['trades_executed']}")
        
        if self.stats['total_predictions'] > 0:
            buy_pct = self.stats['buy_signals'] / self.stats['total_predictions'] * 100
            sell_pct = self.stats['sell_signals'] / self.stats['total_predictions'] * 100
            wait_pct = self.stats['wait_signals'] / self.stats['total_predictions'] * 100
            print(f"\n Signal Distribution:")
            print(f"   BUY:  {buy_pct:.1f}%")
            print(f"   SELL: {sell_pct:.1f}%")
            print(f"   WAIT: {wait_pct:.1f}%")
        
        print(f"{'='*70}\n")
    
    def run_loop(self):
        """รันลูปตรวจสอบอย่างต่อเนื่อง"""
        print(f"\n{'='*70}")
        print(f"🔄 STARTING CONTINUOUS MONITORING")
        print(f"{'='*70}")
        print(f"⏱️  Check interval: {self.interval_sec}s ({self.interval_sec/3600:.1f}h)")
        print(f"🎯 Min confidence: {self.min_confidence:.0%}")
        print(f"💰 Mode: {'DEMO' if self.demo_mode else 'LIVE'}")
        print(f"📊 Symbol: {self.symbol}")
        print(f"📝 Logging to: {self.log_file}")
        print(f"\n⚠️  Press Ctrl+C to stop\n")
        
        try:
            while True:
                # รันทำนาย
                self.run_once()
                
                # พิมพ์สรุปทุก 6 ชั่วโมง
                if self.stats['total_predictions'] % 6 == 0 and self.stats['total_predictions'] > 0:
                    self.print_summary()
                
                # รอ interval
                print(f"\n⏳ Next check in {self.interval_sec}s ({self.interval_sec/60:.0f} min)...")
                
                # แบ่ง sleep เป็นช่วงสั้นๆ เพื่อตอบสนอง Ctrl+C ได้เร็ว
                for _ in range(self.interval_sec):
                    time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
            self.print_summary()
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            self.print_summary()
        finally:
            mt5.shutdown()
            print("✅ MT5 disconnected")
            print("👋 Goodbye!")


# ================= MAIN =================
if __name__ == "__main__":
    # การตั้งค่า
    SYMBOL = 'XAUUSD'
    DEMO_MODE = True  # ⚠️ เปลี่ยนเป็น False เมื่อต้องการเทรดจริง
    INTERVAL_SEC = 350  # ตรวจสอบทุก 5 นาที (300 วินาที)
    MIN_CONFIDENCE = 0.80  # ความมั่นใจขั้นต่ำ 80%

    # สร้าง trader
    trader = GoldRealtimeTrader(
        symbol=SYMBOL,
        demo_mode=DEMO_MODE,
        models_dir='models',
        interval_sec=INTERVAL_SEC,
        min_confidence=MIN_CONFIDENCE
    )
    
    # รันแบบวนลูป
    trader.run_loop()
    
    # หรือถ้าต้องการทดสอบแค่ครั้งเดียว ให้ใช้:
    # trader.run_once()