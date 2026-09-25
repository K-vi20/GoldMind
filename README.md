## 🧠 LSTM–BiLSTM Ensemble Classifier (ICT Pattern-Based) — โมดูลเสริมสำหรับ GoldMind

> ส่วนนี้สรุปจากโน้ตบุ๊กทดลอง (`.ipynb`) ที่ใช้แนวทาง **Deep Learning แบบ Sequence Classification**
> ต่างจากไปป์ไลน์หลักของ GoldMind ที่ใช้ Random Forest / XGBoost แบบ Regression บน 1-hour bar
> โมดูลนี้เป็นแนวทางเสริม (alternative track) ที่โฟกัส **ทำนายทิศทาง (Up / Down / Neutral)** โดยผสาน
> Technical Indicators แบบดั้งเดิมเข้ากับ **ICT (Inner Circle Trader) Price Action Patterns**

แนะนำให้เพิ่มเป็นหัวข้อใหม่ในโครงสร้างเดิม เช่น `## 🧬 Alternative Track: Deep Learning Directional Classifier` ต่อจากหัวข้อ "Summary of Out-of-Sample Results"

---

### 1. ภาพรวมแนวคิด

โมเดลนี้ตั้งเป้าทำนาย **ทิศทางราคาทองใน 3 ชั่วโมงข้างหน้า** เป็น 3 คลาส (Down / Neutral / Up)
โดยรวมสัญญาณจาก 2 กลุ่ม:

- **Technical Indicators แบบดั้งเดิม** — RSI, MACD, Bollinger Band Width, ATR (normalized), Stochastic, Return หลายช่วงเวลา (4h/12h/24h/72h), Cross-asset returns (DXY, VIX, SP500)
- **ICT Price Action Patterns** — Fair Value Gap (FVG), Order Block (OB), Break of Structure / Change of Character (BOS/CHoCH), และรูปแบบแท่งเทียน (Engulfing, Hammer, Shooting Star, Doji, Inside/Outside Bar)

จุดเด่นคือการนำแนวคิด ICT ซึ่งปกติใช้ในการเทรดด้วยตาเปล่า มาแปลงเป็นฟีเจอร์เชิงตัวเลขให้โมเดลเรียนรู้ได้

---

### 2. Pipeline

```
1. Data Collection      → yfinance: Gold (GC=F), DXY, VIX, SP500 (2 ปีย้อนหลัง, timeframe 1H)
2. Feature Engineering  → Indicators + ICT Patterns + Candlestick Patterns (48 features)
3. Target Labeling      → future_return (3-bar ahead) > 0.3% = Up, < -0.3% = Down, else Neutral
4. Feature Selection    → Mutual Information → เลือก Top 15 features
5. Sequencing           → sliding window 48 timesteps (≈ 2 วัน) ต่อ 1 sample
6. Train/Val/Test Split → Walk-forward แบบ chronological (80% / 10% / 10%)
7. Scaling              → StandardScaler (fit บน train เท่านั้น)
8. Model Training       → LSTM + BiLSTM (แยกเทรน)
9. Stacking Ensemble    → ใช้ output ของ LSTM + BiLSTM ป้อนต่อให้ GradientBoostingClassifier (meta-model)
10. Evaluation          → Accuracy, Classification Report, Confusion Matrix
```

---

### 3. สถาปัตยกรรมโมเดล

| ส่วนประกอบ | รายละเอียด |
|---|---|
| **LSTM** | LSTM(32, tanh) → BatchNorm → Dropout(0.3) → LSTM(16, tanh) → BatchNorm → Dropout(0.3) → Dense(8, relu) → BatchNorm → Dense(3, softmax) |
| **BiLSTM** | โครงสร้างเดียวกันแต่ห่อด้วย `Bidirectional` ทุกชั้น LSTM |
| **Regularization** | L2 (1e-4) ทุกชั้นหลัก, Dropout 0.3, Class weighting (แก้ปัญหา class imbalance) |
| **Optimizer / Loss** | Adam (lr=0.001), sparse_categorical_crossentropy |
| **Callbacks** | EarlyStopping (patience=20, monitor val_loss), ReduceLROnPlateau (factor=0.5, patience=8) |
| **Meta-model** | GradientBoostingClassifier (n_estimators=200, max_depth=4, lr=0.1, subsample=0.8) เรียนจาก probability output ของ LSTM+BiLSTM รวมกัน (stacking) |

**Class distribution ของ target (หลังปรับ threshold เป็น 0.3%):**
Down 19.5% / Neutral 58.1% / Up 22.4% — ข้อมูลไม่สมดุล จึงต้องใช้ class weighting

---

### 4. ผลลัพธ์บน Test Set

| โมเดล | Accuracy |
|---|---|
| LSTM (เดี่ยว) | 31.26% |
| BiLSTM (เดี่ยว) | 32.23% |
| **Ensemble (Stacking)** | **48.49%** |

**Classification Report (Ensemble):**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Down | 0.24 | 0.15 | 0.19 | 234 |
| Neutral | 0.55 | 0.79 | 0.65 | 546 |
| Up | 0.31 | 0.13 | 0.18 | 247 |
| **Accuracy** | | | **0.48** | 1027 |
| Macro avg | 0.37 | 0.36 | 0.34 | 1027 |
| Weighted avg | 0.42 | 0.48 | 0.43 | 1027 |

**ข้อสังเกตสำคัญ:**
- โมเดลเดี่ยว (LSTM/BiLSTM) แม่นยำต่ำกว่าการเดาสุ่มของคลาสส่วนใหญ่ (Neutral 58%) แต่เมื่อ stack ผ่าน Gradient Boosting ความแม่นยำโดยรวมดีขึ้นชัดเจน
- โมเดลยัง**เอนเอียงไปทาย Neutral มากเกินไป** (recall 79%) ขณะที่ทาย Down/Up ได้ recall ต่ำ (~13–15%) — สะท้อนว่าโมเดลยังไม่จับสัญญาณการกลับตัวได้ดีนัก เหมาะกับงานต่อยอด เช่น ปรับ threshold, เพิ่มฟีเจอร์ momentum, หรือใช้ focal loss

---

### 5. Artifacts ที่บันทึกไว้

```
models/
├── improved_lstm.keras
├── improved_bilstm.keras
├── improved_meta.pkl          # GradientBoosting meta-model
├── improved_scaler.pkl        # StandardScaler
├── improved_features.pkl      # รายชื่อ 15 features ที่เลือก
└── improved_config.pkl        # time_steps=48, threshold=0.003

data/
├── train_data.csv / val_data.csv / test_data.csv
├── full_features.csv          # ก่อน feature selection (48 features)
└── raw_data.csv               # ราคาดิบจาก yfinance
```

---

### 6. Top Features (จัดอันดับด้วย Mutual Information)

1. `Gold_ATR_Pct`
2. `Gold_ATR`
3. `Gold_BB_Width`
4. `Gold_MACD`
5. `Gold_MACD_Signal`
6. `Gold_RSI`
7. `Gold_Return_72`, `Gold_Return_12`, `Gold_Return_24`
8. `BOS_Bearish`, `CHoCH_Bullish` *(ฟีเจอร์จาก ICT)*
9. `OB_Net_20` *(Order Block net score)*
10. `Body_Ratio`, `Inside_Bar` *(candlestick features)*
11. `VIX_Return_12`

→ ฟีเจอร์ volatility (ATR, BB Width) มีอิทธิพลสูงสุด รองลงมาคือ momentum (MACD, RSI) และฟีเจอร์ ICT/candlestick ก็ติด Top 15 ด้วย ยืนยันว่าการเพิ่ม price-action pattern ช่วยเสริมสัญญาณได้จริง

---

### 7. ข้อเสนอแนะสำหรับพัฒนาต่อ

- แก้ class imbalance ด้วยเทคนิคอื่นเพิ่มเติม เช่น SMOTE บน sequence หรือ focal loss แทน class_weight เพียงอย่างเดียว
- ทดลองปรับ threshold การ label (0.3%) ให้ตอบโจทย์ trading strategy จริง แล้ววัดผลเป็น backtest (เชื่อมกับ pipeline backtest ที่มีอยู่แล้วใน `Train.ipynb`)
- เปรียบเทียบผลลัพธ์ classification นี้กับ XGBoost Classifier (Conviction) ที่มีอยู่ใน README เดิม เพื่อดูว่าแนวทาง Deep Learning + ICT ให้ Sharpe/Win Rate ดีกว่าหรือไม่เมื่อแปลงกลับเป็นกลยุทธ์เทรด
- พิจารณาย้ายโค้ดจาก notebook เดี่ยวไปเป็นโมดูลใน `src/` (เช่น `src/ict_features.py`, `src/dl_models.py`) ให้สอดคล้องกับโครงสร้างโปรเจกต์เดิม