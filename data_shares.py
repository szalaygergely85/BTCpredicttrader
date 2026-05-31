"""
data_shares.py — yfinance-based data pipeline for stock tickers.
Reuses the same indicator set as data.py so the TFT feature list stays unchanged.
Market hours: NYSE Mon–Fri 9:30–16:00 ET only.
"""
import pandas as pd
import numpy as np
import ta
import requests
import pytz
from datetime import datetime

import yfinance as yf


# ── Market hours ───────────────────────────────────────────────
def is_market_open():
    """True if NYSE is currently open (9:30–16:00 ET, Mon–Fri)."""
    et  = pytz.timezone("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:
        return False
    mo = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    mc = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return mo <= now < mc

def minutes_to_open():
    """Minutes until next market open (0 if already open)."""
    if is_market_open():
        return 0
    et  = pytz.timezone("America/New_York")
    now = datetime.now(et)
    target = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now >= target or now.weekday() >= 5:
        # next trading day
        from datetime import timedelta
        days_ahead = 1
        while (now + timedelta(days=days_ahead)).weekday() >= 5:
            days_ahead += 1
        target = (now + timedelta(days=days_ahead)).replace(
            hour=9, minute=30, second=0, microsecond=0
        )
    delta = target - now
    return max(0, int(delta.total_seconds() / 60))


# ── Data helpers ────────────────────────────────────────────────
def _flatten(df):
    """Flatten yfinance MultiIndex columns and normalise to OHLCV."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"open": "open", "high": "high", "low": "low",
                             "close": "close", "volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _add_indicators(df):
    df["rsi"]          = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd               = ta.trend.MACD(df["close"])
    df["macd"]         = macd.macd()
    df["macd_signal"]  = macd.macd_signal()
    df["macd_diff"]    = macd.macd_diff()
    bb                 = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"]     = bb.bollinger_hband()
    df["bb_lower"]     = bb.bollinger_lband()
    df["bb_width"]     = df["bb_upper"] - df["bb_lower"]
    df["atr"]          = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["ema20"]        = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"]        = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["volume_delta"] = df["volume"].diff()
    df["hour"]         = df["time"].dt.hour
    df["day_of_week"]  = df["time"].dt.dayofweek
    df["obv"]          = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    stoch              = ta.momentum.StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3)
    df["stoch_rsi"]    = stoch.stochrsi()
    df["vwap"]         = (df["close"] * df["volume"]).rolling(24).sum() / df["volume"].rolling(24).sum()
    df["ema200"]       = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    df["adx"]          = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["ema_ratio"]    = df["ema20"] / df["ema50"]
    df["funding_rate"] = 0.0   # not applicable to stocks
    df = df.dropna().reset_index(drop=True)
    return df


def _add_daily_features(df):
    """Use daily candles as the '4h' multi-timeframe proxy for stocks."""
    ticker = df.attrs.get("ticker", "")
    raw = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    daily = _flatten(raw)
    daily["time"] = raw.index.tz_localize(None) if raw.index.tz else raw.index
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.dropna().reset_index(drop=True)

    daily["rsi_4h"]      = ta.momentum.RSIIndicator(daily["close"], window=14).rsi()
    macd_d               = ta.trend.MACD(daily["close"])
    daily["macd_diff_4h"]= macd_d.macd_diff()
    daily["ema20_4h"]    = ta.trend.EMAIndicator(daily["close"], window=20).ema_indicator()
    daily["ema50_4h"]    = ta.trend.EMAIndicator(daily["close"], window=50).ema_indicator()
    daily = daily[["time", "rsi_4h", "macd_diff_4h", "ema20_4h", "ema50_4h"]].dropna()

    # merge by date (hourly rows → closest preceding daily candle)
    df["_date"] = df["time"].dt.normalize()
    daily["_date"] = pd.to_datetime(daily["time"]).dt.normalize()
    df = pd.merge_asof(
        df.sort_values("time"),
        daily.sort_values("_date")[["_date", "rsi_4h", "macd_diff_4h", "ema20_4h", "ema50_4h"]],
        left_on="_date", right_on="_date", direction="backward",
    )
    df = df.drop(columns=["_date"], errors="ignore")
    return df


def _add_fear_greed(df):
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1500&format=json", timeout=10)
        data = resp.json()["data"]
        fg   = pd.DataFrame(data)[["timestamp", "value"]]
        fg["date"]        = pd.to_datetime(fg["timestamp"].astype(int), unit="s").dt.normalize()
        fg["fear_greed"]  = fg["value"].astype(float)
        fg = fg[["date", "fear_greed"]].sort_values("date").reset_index(drop=True)
        df["date"] = df["time"].dt.normalize()
        df = df.merge(fg, on="date", how="left")
        df["fear_greed"] = df["fear_greed"].fillna(50.0)
        df = df.drop(columns=["date"])
    except Exception as e:
        print(f"  Fear & Greed fetch failed ({e}), using 50")
        df["fear_greed"] = 50.0
    return df


def _fetch_hourly(ticker, period):
    raw = yf.download(ticker, period=period, interval="1h", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    df = _flatten(raw)
    # Remove timezone, keep naive UTC-ish timestamps
    df["time"] = raw.index.tz_localize(None) if raw.index.tz else raw.index
    df["time"] = pd.to_datetime(df["time"])
    df.attrs["ticker"] = ticker
    # Drop pre/post-market rows (keep 09:30–16:00 only)
    df = df[df["time"].dt.hour.between(9, 15)].copy()
    df = df[~((df["time"].dt.hour == 9) & (df["time"].dt.minute < 30))].copy()
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def get_shares_training_data(ticker):
    ticker = ticker.upper()
    print(f"  Fetching 2y hourly candles for {ticker} (market hours only)...")
    df = _fetch_hourly(ticker, period="2y")
    print(f"  Fetched {len(df)} raw candles. Adding indicators...")
    df = _add_indicators(df)
    print(f"  After indicators: {len(df)} rows. Adding daily (4h-proxy) features...")
    df.attrs["ticker"] = ticker
    df = _add_daily_features(df)
    print("  Adding Fear & Greed Index...")
    df = _add_fear_greed(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def get_shares_live_data(ticker):
    ticker = ticker.upper()
    df = _fetch_hourly(ticker, period="60d")
    df = _add_indicators(df)
    df.attrs["ticker"] = ticker
    df = _add_daily_features(df)
    df = _add_fear_greed(df)
    df = df.dropna().reset_index(drop=True)
    return df
