# BTCpredicttrader

A local Bitcoin price prediction and paper trading CLI tool. No API keys required. Everything runs locally on CUDA (NVIDIA GPU).

## What it does

Runs every 15 minutes and prints a trading signal based on two inputs:

1. **TFT Model** — a Temporal Fusion Transformer trained on 1 year of hourly BTC/USDT candles, predicting whether price will rise >0.5% over the next 2 hours
2. **Sentiment** — fetches the 10 latest headlines from CoinDesk and CoinTelegraph RSS feeds, sends them to a local LLM (llama3.1 via Ollama), and gets a BULLISH / BEARISH / NEUTRAL score

Combines both into a signal: `STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`

Also runs a **paper trading** simulation starting with $20 USDT, and tracks **prediction accuracy** broken down by confidence level.

Example output:
```
[18:15] BTC $76,803.95 | TFT: UP 82.3% | Sentiment: BULLISH 71 | → BUY
  Latest headlines analyzed by llama3.1:
    • Bitcoin ETF inflows hit record $1.2B in single day
    • Fed signals dovish stance, risk assets rally
    • MicroStrategy adds 5,000 BTC to treasury
    ...
  Accuracy: 64.2% correct over 28 predictions | Avg confidence: 76.1%
  By confidence — 50-65%: 48% (10)  65-80%: 69% (13)  80%+: 100% (5)
  💰 BOUGHT 0.000026 BTC for $2.00 | Portfolio: $20.14 | P&L: +0.14 (+0.70%)
```

---

## Requirements

- Windows / Linux
- Python 3.10–3.12 (Python 3.13 works but needs cu124 PyTorch index)
- NVIDIA GPU with CUDA (tested on RTX 4060)
- [Ollama](https://ollama.com/download) installed and running

---

## Installation

### 1. Install PyTorch with CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install project dependencies
```bash
pip install pytorch-forecasting pytorch-lightning pandas numpy ta python-binance feedparser ollama schedule requests
```

### 3. Install Ollama and pull the model
Download Ollama from [ollama.com/download](https://ollama.com/download), then:
```bash
ollama pull llama3.1
```
This downloads ~4.7 GB once. After that Ollama runs as a background service.

---

## Usage

```bash
python main.py
```

On first run it will:
1. Fetch 1 year of hourly BTC candles from Binance (no auth required)
2. Fetch 4h candles for multi-timeframe features
3. Fetch the Fear & Greed Index from alternative.me
4. Train the TFT model on your GPU (~10–15 minutes)
5. Start the 15-minute prediction loop

Subsequent runs load the saved model instantly. The model auto-retrains after 7 days.

---

## Project structure

```
main.py          — entry point, 15-minute loop
data.py          — Binance candle fetching + all indicator calculations
model.py         — TFT model: training, loading, prediction
sentiment.py     — RSS headline fetching + Ollama LLM sentiment scoring
signals.py       — combines TFT + sentiment into a trading signal
trader.py        — paper trading engine (buy/sell/portfolio tracking)
tracker.py       — prediction accuracy logger and stats printer
```

### Runtime files (git-ignored)
```
tft_btc.ckpt     — trained model checkpoint
meta.json        — last training date (triggers retraining after 7 days)
portfolio.json   — paper trading state (balance, trades, P&L)
predictions.json — prediction log for accuracy tracking
```

---

## Model details

**Architecture:** Temporal Fusion Transformer (pytorch-forecasting)

**Target:** Will BTC price be >0.5% higher 2 hours from now? (binary 0/1)

**Loss:** QuantileLoss [0.1, 0.5, 0.9] — median quantile used as direction score

**Training data:** 1 year of hourly BTCUSDT candles from Binance public API

**Features (20 total):**

| Category | Features |
|----------|---------|
| Momentum | RSI (14), Stochastic RSI, MACD, MACD diff |
| Trend | EMA20, EMA50 |
| Volatility | Bollinger Bands upper/lower/width, ATR |
| Volume | OBV, Volume delta, VWAP (24h rolling) |
| Time | Hour of day, Day of week |
| 4h timeframe | RSI 4h, MACD diff 4h, EMA20 4h, EMA50 4h |
| Sentiment | Fear & Greed Index (alternative.me) |

**Model size:** ~1.1M parameters | hidden: 128 | attention heads: 8

**Train/val split:** 80/20

---

## Signal logic

| Condition | Signal |
|-----------|--------|
| TFT UP ≥65% AND Sentiment BULLISH | STRONG BUY |
| TFT UP ≥55% AND Sentiment not BEARISH | BUY |
| TFT DOWN ≥65% AND Sentiment BEARISH | STRONG SELL |
| TFT DOWN ≥55% AND Sentiment not BULLISH | SELL |
| Everything else | HOLD |

---

## Paper trading logic

- Starts with **$20.00 USDT** (virtual, no real money)
- **BUY / STRONG BUY** → spend 10% of available USDT
- **SELL / STRONG SELL** → sell all held BTC
- State saved to `portfolio.json`, persists across restarts
- To reset: delete `portfolio.json`

---

## Accuracy tracking

Predictions are logged with their timestamp and price. After 2 hours, the tracker checks whether the actual price movement matched the prediction and marks it correct or wrong.

Stats are printed every tick and broken down by confidence bucket (50–65%, 65–80%, 80%+) so you can see whether high-confidence predictions are actually more reliable.

Reset by deleting `predictions.json`.

---

## Notes

- Sentiment falls back to NEUTRAL 50 if Ollama is not running
- The model requires CUDA — it will error on startup if no GPU is detected
- Binance public API has no rate limits for market data, no account needed
- Fear & Greed Index is fetched from [alternative.me](https://alternative.me/crypto/fear-and-greed-index/) (free, no key)
- To force retraining: delete `tft_btc.ckpt` and `meta.json`
