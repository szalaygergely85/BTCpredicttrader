# BTC Price Prediction + Paper Trading

A local research project that trains a 3-model **Temporal Fusion Transformer ensemble** to forecast short-term price direction on BTC, BTC-scalping (5-minute), and individual US stocks. Predictions feed a CLI paper-trading engine with stop-loss / take-profit, position sizing, and optional short-selling. Everything runs locally on a single NVIDIA GPU. No API keys required.

This README is written so that a new reader (or their AI assistant) can understand **how the system works end-to-end** — especially the machine-learning side — and reproduce it on their own machine.

---

## Current state (honest assessment)

This is a research project, not a strategy you should run with real money. Results below are out-of-sample backtests on the last 20% of historical data (single regime), so they're optimistic about real-world performance.

| Mode | Test return | Sharpe | vs HODL | Status |
|---|---|---|---|---|
| **Swing** (BTC 1h) | +12.3% over 217d | 3.04 | +47.6% ✓ | Solidly profitable in this regime |
| **Scalp** (BTC 5min) | -2.7% over 35d | -16.6 | +3.3% ✓ | Models are short-biased; needs retraining |
| **Shares** (NVDA 1h) | -0.4% over 27d | 0.54 | -16.2% ✗ | Strategy mismatched to smooth-trend stocks |

The test period was mostly a BTC drawdown from ~$108k to ~$73k, which favours a short-leaning strategy. The numbers should be re-validated on a bull-market window before drawing conclusions.

---

## How it works (the ML side)

### The prediction task

Given the last *N* hours of OHLCV data plus engineered features, predict a single binary label:

> **Will the close price be at least X% higher Y candles from now?**

For swing mode: *N=168* (7 days of hourly candles), *Y=4* (4 hours ahead), *X=0.4%*.

The label is computed in `model.py:prepare_dataset()`:

```python
df["target"] = (df["close"].shift(-horizon) > df["close"] * threshold).astype(int)
```

This is a **binary classification problem**, not regression. The model outputs two logits (UP / DOWN); we apply softmax to get `P(UP)`.

### Why this framing

Predicting raw price is hard and the loss signal is dominated by drift. A directional binary label with an explicit threshold (must move >0.4% to count as UP) ignores noise around the current price and forces the model to commit to a real directional call.

### Architecture: Temporal Fusion Transformer (TFT)

We use `pytorch-forecasting`'s `TemporalFusionTransformer`. TFT is a sequence model designed for forecasting that combines:

- **Variable selection networks** — learns per-feature gating weights so irrelevant features are downweighted
- **LSTM encoder** over the past *N* candles
- **Multi-head self-attention** over the encoder outputs (lets the model attend to specific past candles, e.g. a recent breakout)
- **Quantile / classification head** — in our case, 2-class output with CrossEntropy

```python
TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=128,
    attention_head_size=8,
    dropout=0.3,
    hidden_continuous_size=64,
    loss=CrossEntropy(),
    output_size=2,
)
```

Roughly 1.1M parameters per model. Trains in ~10–15 minutes per model on an RTX 4060.

**Why TFT** over a plain LSTM or vanilla transformer: variable selection is the killer feature for tabular time-series with many heterogeneous features (price-based indicators, time-of-day, sentiment, etc.) — it learns which features matter when, rather than treating them as a flat input vector.

### Features (24 total)

All features are computed in `data.py` and fed into the model as `time_varying_unknown_reals`:

| Category | Features | Why |
|---|---|---|
| Momentum | `rsi`, `stoch_rsi`, `macd`, `macd_diff` | Overbought/oversold and trend strength |
| Trend | `ema20`, `ema50`, `ema200`, `ema_ratio` | Multi-scale trend direction |
| Volatility | `bb_upper`, `bb_lower`, `bb_width`, `atr` | Regime detection + stop-loss sizing |
| Volume | `obv`, `volume_delta`, `vwap` | Confirms moves; VWAP is intraday fair value |
| Regime | `adx` | Trending vs ranging |
| Time | `hour`, `day_of_week` | Captures session effects (US/Asia overlap, weekend volatility) |
| Multi-timeframe | `rsi_4h`, `macd_diff_4h`, `ema20_4h`, `ema50_4h` | Higher-TF context joined onto 1h candles |
| External | `fear_greed`, `funding_rate` | Crowd sentiment + futures positioning |

The 4h features are computed on independent 4h candles and merged onto the 1h dataframe with forward-fill, so the model sees "what does the 4h chart look like right now?" as additional context.

### Training: CrossEntropy with class-balanced sample weights

In a bear-market dataset, the UP class is rare (~30–35%). Without weighting, the model collapses to "always predict DOWN" because that's locally optimal. We rebalance:

```python
n_up   = df["target"].sum()
n_down = len(df) - n_up
df["sample_weight"] = df["target"].map({1: n_down / n_up, 0: 1.0})
```

UP samples get weight `n_down/n_up` (≈2–3×), DOWN samples stay at 1.0. The dataloader passes these weights to CrossEntropy.

**Train/val split: 80/20, time-ordered (no shuffling).** For time-series this is non-negotiable — random shuffling leaks future information into training. The split is enforced by `time_idx`:

```python
df[df.time_idx <= cutoff]  # train
df[df.time_idx >  cutoff]  # validation
```

Training runs for 30 epochs, batch size 64, dropout 0.3, gradient clip 0.1.

### The ensemble (3 models, majority vote)

A single TFT is noisy. We train three models with deliberately *different* configurations to encourage diverse failure modes:

| Model | Encoder length | Training data slice |
|---|---|---|
| A | 168h (7 days) | Full dataset |
| B | 168h (7 days) | Most recent 50% only |
| C | 336h (14 days) | Full dataset |

Model B sees only recent data — it overfits to the current regime but reacts faster to regime shifts. Model C sees twice as much history per prediction — slower but more stable. Model A is the baseline.

At inference time:
1. Each model emits a `P(UP)` for the current candle
2. Majority vote on direction (UP if 2+ models say UP)
3. Confidence = average of agreeing-model probabilities

This is in `ensemble.py:predict_ensemble()`.

### Calibration

Raw classifier probabilities are often miscalibrated — a model that says "70% confident" might actually only be right 60% of the time. After training, we bin validation predictions by confidence (0.5–0.6, 0.6–0.7, etc.) and record the actual accuracy in each bin. At inference, the model's raw score is mapped to its bin's empirical accuracy. This is in `model.py:compute_calibration()` and `apply_calibration()`.

### Critical detail: logits vs probabilities

The model uses `CrossEntropy` loss with `output_size=2`. The raw output `raw["prediction"]` is a tensor of shape `[batch, 1, 2]` containing **logits**, not probabilities. To get `P(UP)`:

```python
logits = raw["prediction"][:, 0, :]   # [batch, 2]
probs = torch.softmax(logits, dim=-1)
p_up = probs[:, 1]                     # P(UP)
```

This is the canonical pattern in `model.py:84` and `model.py:223`. The backtest used to skip the softmax (treating logits as probabilities), which silently destroyed signal calibration — see "Recent fixes" below.

---

## Data pipeline

### Crypto (BTC)

Data is fetched from **Binance public endpoints** — no auth, no rate limits for market data:

1. `data.py:get_training_data()` — 3 years of 1h `BTCUSDT` candles (paginated, ~26,000 rows)
2. `data.py:get_scalp_training_data()` — 6 months of 5-min candles (~50,000 rows)
3. 4h candles fetched separately and merged for multi-timeframe features
4. `alternative.me` — daily Fear & Greed Index
5. Binance Futures — funding rates (8h intervals)
6. Binance order book — bid/ask imbalance (live only, not used in training)

All indicators (RSI, MACD, EMA, ATR, Bollinger, etc.) computed with the `ta` library.

### Stocks

`data_shares.py` uses **yfinance** for 2 years of hourly OHLCV (market hours only). Same indicators, plus a "daily-as-4h" proxy (daily candles forward-filled to hourly). No funding rate (stocks don't have one); Fear & Greed kept as a global risk-on/off proxy.

### Sentiment (live only, not a model feature)

`sentiment.py` fetches RSS headlines from CoinDesk, CoinTelegraph, r/Bitcoin, r/CryptoCurrency, BitcoinMagazine, Decrypt — sends them to a local **Ollama** LLM (`llama3.1`) which scores them BULLISH / BEARISH / NEUTRAL (0–100). This is used at live inference time to bias the signal; the trained model does *not* see this feature.

---

## Backtesting methodology

`backtest.py` simulates the strategy on the held-out 20% of data, with proper temporal isolation:

```python
cutoff = int(n * 0.8)
validation = TimeSeriesDataSet(
    df_prep[df_prep.time_idx >= cutoff - ENCODER_LENGTH],
    ...
    min_prediction_idx=cutoff,   # ← only predicts on rows beyond training cutoff
)
```

Models are loaded from checkpoints (which were trained on the first 80%), and only the last 20% is scored. This is genuine out-of-sample.

### What the backtest reports

- **Total return** and **Sharpe ratio** (annualized from daily-sampled equity)
- **Max drawdown**
- **Win rate**, average win/loss, profit factor
- **Trade-type breakdown** (BUY, SELL, OPEN SHORT, STOP_LOSS, TAKE_PROFIT…)
- **3-window regime breakdown** — splits the test period into thirds so you can see whether the edge is consistent or comes from one lucky window
- **Score distribution histogram** — sanity-check what the model is actually predicting

### Honest caveats

1. Threshold tuning (`BUY_CONF`, `SELL_CONF`, stop multipliers) was done by looking at backtest results, which means the swing numbers are partially **in-sample**. Real out-of-sample will be lower.
2. The test window (last 217 days) was almost entirely a BTC downtrend. A short-biased strategy was always going to look good. A bull-market validation is the missing piece.
3. Scalp test = 35 days. Shares (NVDA) test = 27 days. These are too short to draw conclusions.

---

## Trading logic (brief)

Three modes, same TFT ensemble under the hood, different parameters:

| | Swing | Scalp | Shares |
|---|---|---|---|
| Candle | 1h | 5min | 1h (market hours) |
| Encoder | 168 candles | 60 candles | 48 candles |
| Horizon | 4h | 15min | 4h |
| Target threshold | +0.4% | +0.1% | +0.5% |
| BUY confidence | ≥70% | ≥70% | ≥70% |
| SELL confidence | ≥75% | ≥75% | ≥95% (effectively no shorts) |
| Allow shorts | yes (in bear regime) | yes (in bear regime) | no |
| Stop loss | 1.5×ATR | 1.5×ATR | 2.5×ATR (wider) |
| Take profit | 2.5×ATR | 2.5×ATR | none (let winners run) |
| Cooldown | 4 candles | 6 candles | 8 candles |

**Entry filters** (long): must be above EMA200 (`in_bull`) AND either
- mean-reversion: `RSI < 55 AND price ≤ VWAP`, OR
- trend-following: `confidence ≥ 75` (added to capture rallies)

**Short entry**: only when `price < EMA200`. 10% of cash as collateral, ATR-based stops.

**Position sizing** scales with model confidence: 10% / 20% / 30% / 40% of available cash. Trend-following entries get a slightly larger size than dip entries.

The live trader (`main.py` + `trader.py` + `signals.py`) is similar but adds an extra sentiment gate from the Ollama LLM headline scoring.

---

## Setup

### Requirements

- Windows or Linux (tested on Windows 11)
- **NVIDIA GPU with CUDA** (training will fail without one — tested on RTX 4060)
- **Python 3.10–3.13** (3.13 works but needs cu124 PyTorch wheel)
- [Ollama](https://ollama.com/download) — only needed for live trading sentiment, not for backtesting or training

### Install

```bash
git clone <this-repo>
cd BitcoinPricePredict

# 1. PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Project deps
pip install pytorch-forecasting pytorch-lightning lightning \
            pandas numpy ta \
            python-binance yfinance pytz \
            feedparser ollama schedule requests \
            flask
```

### Ollama (for live sentiment only)

```bash
ollama pull llama3.1   # ~4.7 GB, one-time
```

Skip this if you only care about training and backtesting — sentiment falls back to NEUTRAL when Ollama isn't running.

---

## Usage

### First run — trains the ensemble

```bash
# Swing (default): 3 BTC models, ~10–15 min each on RTX 4060
python main.py

# Scalping: separate ensemble on 5-min candles
python main.py -scalping

# Stocks: separate ensemble per ticker
python main.py -shares NVDA
```

First run for any mode:
1. Fetches historical data from Binance / yfinance
2. Computes indicators
3. Trains 3 TFT models (saves `tft_a.ckpt`, `tft_b.ckpt`, `tft_c.ckpt` — or `_scalp_*`, `_nvda_*`)
4. Saves validation accuracy + calibration to `meta.json` / `meta_scalp.json` / `meta_nvda.json`
5. Starts the live prediction loop (15 min for swing, 2 min for scalp, 30 min for shares)

Subsequent runs load the saved models instantly. Models auto-retrain after 7 days.

### Backtesting (the more interesting part for understanding)

```bash
python backtest.py                  # Swing — full 217-day test
python backtest.py -scalping        # Scalp — 35-day test
python backtest.py -shares NVDA     # NVDA — 27-day test
```

This loads the trained checkpoints and replays them on the held-out last 20% of data. Prints score distributions, trade breakdowns, and a 3-window regime breakdown.

### Dashboard

```bash
python dashboard.py
# → http://localhost:5000
```

A Flask web UI that shows live portfolio status, recent predictions, accuracy stats, and lets you start/stop the prediction processes for each mode. Templates are in `templates/index.html`.

### Forcing a retrain

```bash
# Delete the checkpoints + meta for the mode you want to retrain
rm tft_a.ckpt tft_b.ckpt tft_c.ckpt meta.json        # swing
rm tft_scalp_*.ckpt meta_scalp.json                  # scalp
rm tft_nvda_*.ckpt meta_nvda.json                    # NVDA
```

---

## File reference

### Training & inference (the ML core)

| File | Role |
|---|---|
| `model.py` | TFT trainer (`train_single`), single-model predictor (`predict_single`), feature list, dataset prep, calibration |
| `ensemble.py` | Manages the 3-model ensemble (swing): training, loading, majority voting |
| `scalp.py` | Same as ensemble but for the 5-min scalp models |
| `shares.py` | Same as ensemble but per-ticker stock models |
| `data.py` | Binance candle fetching + all crypto indicators + funding rate + order book |
| `data_shares.py` | yfinance candle fetching + market-hours logic for stocks |
| `sentiment.py` | RSS headline fetching + Ollama LLM scoring (live only) |

### Trading & app

| File | Role |
|---|---|
| `main.py` | Entry point. Argparse for mode, schedules ticks, calls predict + execute |
| `signals.py` | Combines TFT output + sentiment + order-book imbalance into BUY/SELL/HOLD |
| `trader.py` | Paper-trading state machine: positions, cash, stops, fees, persistence |
| `tracker.py` | Logs predictions, resolves them after `horizon` candles, prints accuracy stats bucketed by confidence |
| `backtest.py` | Offline simulator over the held-out 20% of data |
| `dashboard.py` | Flask web UI |
| `templates/index.html` | Dashboard HTML |

### Runtime files (gitignored)

| File | Purpose |
|---|---|
| `tft_*.ckpt` | Trained TFT model weights (one set per mode) |
| `meta*.json` | Per-mode metadata: last_trained, val_accuracy, calibration bins |
| `portfolio*.json` | Current cash/position/trades (paper) |
| `predictions*.json` | Logged predictions + resolved correctness (for accuracy tracking) |
| `log_*.txt` | stdout captured by dashboard subprocess |

Delete any of these to reset that piece of state.

---

## Recent fixes worth knowing about

### The softmax bug (fixed)

The backtest used to read `raw["prediction"][:, 0, 1]` directly as if it were a probability. But the model is trained with `CrossEntropy(output_size=2)` — that tensor is **logits ∈ ℝ**, not a probability ∈ [0, 1]. Most logits are negative reals; the legacy code clamped them to `1e-7` (because they're invalid as probabilities) and then ran a temperature-scaling sigmoid on top, producing wildly miscalibrated scores.

After fixing this (`backtest.py:329` now applies `torch.softmax`), the score distribution became bimodal and well-calibrated, and the temperature-scaling step (`TEMPERATURE = 2.0`) became unnecessary and counterproductive. It's now `TEMPERATURE = 1.0` (no scaling).

The live trader was never affected — it always went through `ensemble.py → model.py:predict_single`, which applies softmax correctly. Only the backtest was wrong.

### Per-mode SL/TP

The stop-loss / take-profit multipliers used to be hardcoded (1.5× / 2.5× ATR). Now they're configurable per mode. Stocks use a wider stop (2.5× ATR) and disable take-profit entirely so winners can ride the trend.

### Trend-following long entry

The original long entry required `RSI < 55 AND price ≤ VWAP` — pure mean-reversion. In a strong uptrend RSI stays high and price stays above VWAP, so the bot never bought rallies. Now an additional `confidence ≥ 75` path opens longs on strong model signals regardless of dip conditions.

---

## Known limitations

1. **Single regime backtest.** The 20% test window covers one bear market. Strategy may perform very differently in a sustained bull run.
2. **Scalp models are short-biased.** Training data was dominated by a downtrend; the models almost never produce confident UP predictions. Retraining on longer, balanced history is the real fix — parameter tweaks won't help.
3. **Stock models are weak on smooth uptrends.** A strategy with stops and TPs structurally can't beat buy-and-hold during a one-way rally. Stocks may simply be the wrong asset class for this design.
4. **In-sample threshold tuning.** Confidence thresholds were chosen by looking at backtest results. A proper walk-forward validation would give more honest numbers.
5. **No transaction-cost realism beyond a flat 0.1% fee.** No slippage, no funding payments on shorts, no overnight stock financing.

---

## License & disclaimer

Research / educational use. **Not financial advice. Do not use this with real money.** The paper-trading numbers are optimistic for many reasons listed above, and the live live trader has never been validated against a real account.
