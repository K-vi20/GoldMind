import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict
import os

# ================= BACKTEST ENGINE CLASS =================

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str


class GoldBacktester:
    def __init__(self, 
                 spread_pips=0.3,
                 commission_per_lot=5.0,
                 sl_pips=10.0,
                 tp_pips=20.0,
                 risk_per_trade=0.01,
                 initial_capital=10000):
        
        self.spread = spread_pips
        self.commission = commission_per_lot
        self.sl_distance = sl_pips
        self.tp_distance = tp_pips
        self.risk_per_trade = risk_per_trade
        self.initial_capital = initial_capital
        
    def run(self, df: pd.DataFrame, signals: np.ndarray) -> Dict:
        trades: List[Trade] = []
        capital = self.initial_capital
        equity_curve = [capital]
        position = None
        
        signal_map = {0: 'SELL', 1: 'WAIT', 2: 'BUY'}
        
        for i in range(len(df) - 1):
            current_price = df['Close'].iloc[i]
            next_time = df.index[i + 1]
            signal = signal_map.get(int(signals[i]), 'WAIT')
            
            # Check SL/TP
            if position is not None:
                high = df['High'].iloc[i]
                low = df['Low'].iloc[i]
                
                if position['direction'] == 'BUY':
                    if low <= position['entry_price'] - self.sl_distance:
                        exit_price = position['entry_price'] - self.sl_distance
                        pnl = self._close_trade(position, exit_price, next_time, 'sl', trades, capital)
                        capital += pnl
                        position = None
                    elif high >= position['entry_price'] + self.tp_distance:
                        exit_price = position['entry_price'] + self.tp_distance
                        pnl = self._close_trade(position, exit_price, next_time, 'tp', trades, capital)
                        capital += pnl
                        position = None
                        
                elif position['direction'] == 'SELL':
                    if high >= position['entry_price'] + self.sl_distance:
                        exit_price = position['entry_price'] + self.sl_distance
                        pnl = self._close_trade(position, exit_price, next_time, 'sl', trades, capital)
                        capital += pnl
                        position = None
                    elif low <= position['entry_price'] - self.tp_distance:
                        exit_price = position['entry_price'] - self.tp_distance
                        pnl = self._close_trade(position, exit_price, next_time, 'tp', trades, capital)
                        capital += pnl
                        position = None
            
            # Open new position
            if position is None and signal in ['BUY', 'SELL']:
                entry_price = current_price
                if signal == 'BUY':
                    entry_price += self.spread / 2
                else:
                    entry_price -= self.spread / 2
                
                risk_amount = capital * self.risk_per_trade
                size = risk_amount / self.sl_distance
                
                position = {
                    'entry_time': df.index[i],
                    'direction': signal,
                    'entry_price': entry_price,
                    'size': size,
                    'commission': self.commission * size / 0.01
                }
            
            # Close on opposite signal
            elif position is not None:
                if (position['direction'] == 'BUY' and signal == 'SELL') or \
                   (position['direction'] == 'SELL' and signal == 'BUY'):
                    pnl = self._close_trade(position, current_price, df.index[i], 'signal', trades, capital)
                    capital += pnl
                    position = None
            
            equity_curve.append(capital + (self._unrealized_pnl(position, current_price) if position else 0))
        
        if position is not None:
            pnl = self._close_trade(position, df['Close'].iloc[-1], df.index[-1], 'timeout', trades, capital)
            capital += pnl
        
        return self._compute_metrics(trades, equity_curve, df)
    
    def _unrealized_pnl(self, position, current_price):
        if position is None: return 0
        if position['direction'] == 'BUY':
            return (current_price - position['entry_price']) * position['size'] - position['commission']
        else:
            return (position['entry_price'] - current_price) * position['size'] - position['commission']
    
    def _close_trade(self, position, exit_price, exit_time, reason, trades, capital):
        """✅ แก้ไข: return ค่า pnl แทนการ set position['pnl']"""
        if position['direction'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['size'] - position['commission']
        else:
            pnl = (position['entry_price'] - exit_price) * position['size'] - position['commission']
        
        pnl_pct = pnl / capital * 100 if capital > 0 else 0
        
        trades.append(Trade(
            entry_time=position['entry_time'],
            exit_time=exit_time,
            direction=position['direction'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            size=position['size'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason
        ))
        
        return pnl  # ✅ return ค่า pnl
    
    def _compute_metrics(self, trades: List[Trade], equity_curve: List[float], df) -> Dict:
        if not trades:
            return {'error': 'No trades'}
        
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        max_dd = drawdown.min()
        
        returns = np.diff(equity) / equity[:-1]
        sharpe = (returns.mean() / returns.std() * np.sqrt(8760)) if returns.std() > 0 else 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(trades) * 100,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            'max_drawdown_pct': max_dd,
            'sharpe_ratio': sharpe,
            'final_capital': equity_curve[-1],
            'total_return_pct': (equity_curve[-1] / self.initial_capital - 1) * 100,
            'buy_hold_return_pct': buy_hold_return,
            'alpha': (equity_curve[-1] / self.initial_capital - 1) * 100 - buy_hold_return,
            'trades': trades
        }
    
    def print_report(self, metrics: Dict):
        print("\n" + "=" * 80)
        print("📊 BACKTEST REPORT")
        print("=" * 80)
        print(f"\n💰 Performance:")
        print(f"  Initial Capital:     ${self.initial_capital:,.2f}")
        print(f"  Final Capital:       ${metrics['final_capital']:,.2f}")
        print(f"  Total PnL:           ${metrics['total_pnl']:,.2f}")
        print(f"  Total Return:        {metrics['total_return_pct']:.2f}%")
        print(f"  Buy & Hold Return:   {metrics['buy_hold_return_pct']:.2f}%")
        print(f"   Alpha vs B&H:     {metrics['alpha']:+.2f}%")
        
        print(f"\n📈 Trade Statistics:")
        print(f"  Total Trades:        {metrics['total_trades']}")
        print(f"  Winning Trades:      {metrics['winning_trades']}")
        print(f"  Losing Trades:       {metrics['losing_trades']}")
        print(f"  Win Rate:            {metrics['win_rate']:.1f}%")
        print(f"  Avg Win:             ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss:            ${metrics['avg_loss']:.2f}")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
        
        print(f"\n⚠️ Risk Metrics:")
        print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        
        trades = metrics['trades']
        exit_reasons = pd.Series([t.exit_reason for t in trades]).value_counts()
        print(f"\n🚪 Exit Reasons:")
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count} ({count/len(trades)*100:.1f}%)")


# ================= MAIN EXECUTION =================

def predict_signals_from_csv(test_csv_path, models_dir='models'):
    """โหลด test data และทำนายสัญญาณด้วย improved models"""
    print("📦 Loading test data and improved models...")
    
    # โหลด test data
    test_df = pd.read_csv(test_csv_path)
    print(f"✅ Test data loaded: {len(test_df)} rows")
    print(f"   Columns: {list(test_df.columns[:5])}...")
    
    # ✅ โหลด improved models (.keras)
    lstm_path = f'{models_dir}/improved_lstm.keras'
    bilstm_path = f'{models_dir}/improved_bilstm.keras'
    meta_path = f'{models_dir}/improved_meta.pkl'
    scaler_path = f'{models_dir}/improved_scaler.pkl'
    features_path = f'{models_dir}/improved_features.pkl'
    config_path = f'{models_dir}/improved_config.pkl'
    
    # ตรวจสอบไฟล์
    for path, name in [(lstm_path, 'LSTM'), (bilstm_path, 'BiLSTM'), 
                       (meta_path, 'Meta'), (scaler_path, 'Scaler'),
                       (features_path, 'Features')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ {name} not found: {path}")
    
    print(f"\n Loading models from {models_dir}/...")
    lstm_model = tf.keras.models.load_model(lstm_path)
    bilstm_model = tf.keras.models.load_model(bilstm_path)
    meta_model = joblib.load(meta_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    
    # โหลด config (time_steps)
    time_steps = 24
    if os.path.exists(config_path):
        config = joblib.load(config_path)
        time_steps = config.get('time_steps', 24)
        print(f"   Config: time_steps={time_steps}")
    
    print(f"✅ Models loaded:")
    print(f"   LSTM:      {lstm_path}")
    print(f"   BiLSTM:    {bilstm_path}")
    print(f"   Meta:      {meta_path}")
    print(f"   Features:  {len(feature_names)}")
    
    # เตรียม features
    X_test = test_df[feature_names].values
    
    # สร้าง sequence
    print(f"\n🔢 Creating sequences (time_steps={time_steps})...")
    X_seq = []
    for i in range(len(X_test) - time_steps):
        X_seq.append(X_test[i:(i + time_steps)])
    X_seq = np.array(X_seq)
    
    print(f"   Sequences shape: {X_seq.shape}")
    
    # Scale
    n_samples, n_timesteps, n_features = X_seq.shape
    X_scaled = scaler.transform(X_seq.reshape(-1, n_features)).reshape(X_seq.shape)
    
    # Predict
    print("\n🔮 Predicting signals...")
    lstm_pred = lstm_model.predict(X_scaled, verbose=0)
    bilstm_pred = bilstm_model.predict(X_scaled, verbose=0)
    
    # Stacking
    stacked = np.hstack([lstm_pred, bilstm_pred])
    signals = meta_model.predict(stacked)
    
    print(f"✅ Predicted {len(signals)} signals")
    print(f"   Signal distribution: {np.bincount(signals.astype(int), minlength=3)}")
    print(f"   Down(0): {np.sum(signals == 0)}, Wait(1): {np.sum(signals == 1)}, Up(2): {np.sum(signals == 2)}")
    
    return signals


def main():
    print("=" * 80)
    print("📈 PRIORITY 2: BACKTESTING")
    print("=" * 80)
    
    # ตรวจสอบไฟล์
    test_csv = 'data/test_data.csv'
    full_features_csv = 'data/full_features.csv'
    
    if not os.path.exists(test_csv):
        print(f"❌ Error: {test_csv} not found!")
        print("💡 Please run Cell 17 in Notebook first.")
        return
    
    if not os.path.exists(full_features_csv):
        print(f"❌ Error: {full_features_csv} not found!")
        print("💡 Please run Cell 17 in Notebook first.")
        return
    
    # 1. ทำนายสัญญาณ
    signals = predict_signals_from_csv(test_csv)
    
    # 2. โหลดข้อมูลราคาจริง
    print("\n📊 Loading price data for backtesting...")
    
    # ✅ แก้ไขตรงนี้: ใช้ index_col=0 และ parse_dates=True
    full_df = pd.read_csv(full_features_csv, index_col=0, parse_dates=True)
    
    print(f"✅ Full features loaded: {len(full_df)} rows")
    print(f"   Columns: {list(full_df.columns[:5])}...")
    print(f"   Index name: {full_df.index.name}")
    
    # ตรวจสอบว่ามีคอลัมน์ Gold_Open, Gold_High, Gold_Low, Gold_Close ไหม
    required_cols = ['Gold_Open', 'Gold_High', 'Gold_Low', 'Gold_Close']
    missing_cols = [col for col in required_cols if col not in full_df.columns]
    
    if missing_cols:
        print(f"❌ Error: Missing columns: {missing_cols}")
        print(f"   Available columns: {list(full_df.columns)}")
        return
    
    # ดึงเฉพาะช่วง test set
    test_prices = full_df[required_cols].iloc[-len(signals):].copy()
    test_prices.rename(columns={
        'Gold_Open': 'Open',
        'Gold_High': 'High',
        'Gold_Low': 'Low',
        'Gold_Close': 'Close'
    }, inplace=True)
    
    print(f"✅ Price data: {len(test_prices)} rows")
    print(f"   Date range: {test_prices.index[0]} to {test_prices.index[-1]}")
    print(f"   Price range: ${test_prices['Close'].min():.2f} - ${test_prices['Close'].max():.2f}")
    
    # 3. รัน Backtest
    print("\n🏃 Running backtest...")
    backtester = GoldBacktester(
        spread_pips=0.3,
        commission_per_lot=5.0,
        sl_pips=10.0,
        tp_pips=20.0,
        risk_per_trade=0.01,
        initial_capital=10000
    )
    
    metrics = backtester.run(test_prices, signals)
    backtester.print_report(metrics)
    
    # 4. Plot Equity Curve
    print("\n📈 Plotting equity curve...")
    
    equity_curve = [backtester.initial_capital]
    for trade in metrics['trades']:
        equity_curve.append(equity_curve[-1] + trade.pnl)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    axes[0].plot(range(len(equity_curve)), equity_curve, 
                label='Strategy', color='blue', linewidth=2)
    
    bh_equity = backtester.initial_capital * (test_prices['Close'] / test_prices['Close'].iloc[0])
    axes[0].plot(range(len(bh_equity)), bh_equity, label='Buy & Hold', 
                color='gray', linewidth=1.5, linestyle='--')
    
    axes[0].set_title('Equity Curve: Strategy vs Buy & Hold')
    axes[0].set_ylabel('Capital ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    axes[1].fill_between(range(len(drawdown)), drawdown, 0, 
                        color='red', alpha=0.3)
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown %')
    axes[1].set_xlabel('Trade Number')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    print("✅ Chart saved to backtest_results.png")
    plt.show()
    
    print("\n🎉 Backtest completed!")


if __name__ == "__main__":
    main()