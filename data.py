from binance.client import Client
import pandas as pd
import numpy as np
import ta
import time
import requests

client = Client("", "")

CANDLES_PER_REQUEST = 1000

def get_candles(interval, limit):
    all_raw = []
    remaining = limit
    end_time = None

    while remaining > 0:
        batch = min(remaining, CANDLES_PER_REQUEST)
        kwargs = {"symbol": "BTCUSDT", "interval": interval, "limit": batch}
        if end_time:
            kwargs["endTime"] = end_time
        raw = client.get_klines(**kwargs)
        if not raw:
            break
        all_raw = raw + all_raw
        remaining -= len(raw)
        end_time = raw[0][0] - 1
        time.sleep(0.1)

    df = pd.DataFrame(all_raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    df = df[["open_time","open","high","low","close","volume"]].copy()
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.rename(columns={"open_time": "time"})
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df

def fetch_fear_greed():
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=400&format=json", timeout=10)
        data = resp.json()["data"]
        fg = pd.DataFrame(data)[["timestamp", "value"]]
        fg["date"] = pd.to_datetime(fg["timestamp"].astype(int), unit="s").dt.normalize()
        fg["fear_greed"] = fg["value"].astype(float)
        return fg[["date", "fear_greed"]].sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"  Fear & Greed fetch failed ({e}), using neutral 50")
        return None

def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["volume_delta"] = df["volume"].diff()
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    stoch = ta.momentum.StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3)
    df["stoch_rsi"] = stoch.stochrsi()
    df["vwap"] = (df["close"] * df["volume"]).rolling(24).sum() / df["volume"].rolling(24).sum()
    df = df.dropna().reset_index(drop=True)
    return df

def add_4h_features(df_1h, limit=2200):
    df_4h = get_candles(Client.KLINE_INTERVAL_4HOUR, limit)
    df_4h["rsi_4h"] = ta.momentum.RSIIndicator(df_4h["close"], window=14).rsi()
    macd_4h = ta.trend.MACD(df_4h["close"])
    df_4h["macd_diff_4h"] = macd_4h.macd_diff()
    df_4h["ema20_4h"] = ta.trend.EMAIndicator(df_4h["close"], window=20).ema_indicator()
    df_4h["ema50_4h"] = ta.trend.EMAIndicator(df_4h["close"], window=50).ema_indicator()
    df_4h = df_4h[["time","rsi_4h","macd_diff_4h","ema20_4h","ema50_4h"]].dropna()
    df = pd.merge_asof(df_1h.sort_values("time"), df_4h.sort_values("time"), on="time", direction="backward")
    return df

def add_fear_greed(df):
    fg = fetch_fear_greed()
    if fg is None:
        df["fear_greed"] = 50.0
        return df
    df["date"] = df["time"].dt.normalize()
    df = df.merge(fg, on="date", how="left")
    df["fear_greed"] = df["fear_greed"].fillna(50.0)
    df = df.drop(columns=["date"])
    return df

def get_training_data():
    print("  Fetching 1h candles (8760, paginated)...")
    df = get_candles(Client.KLINE_INTERVAL_1HOUR, 8760)
    print(f"  Fetched {len(df)} raw 1h candles. Adding indicators...")
    df = add_indicators(df)
    print("  Fetching 4h candles for multi-timeframe features...")
    df = add_4h_features(df, limit=2200)
    print("  Fetching Fear & Greed Index...")
    df = add_fear_greed(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns")
    return df

def get_live_data():
    df = get_candles(Client.KLINE_INTERVAL_15MINUTE, 500)
    df = add_indicators(df)
    df = add_4h_features(df, limit=100)
    df = add_fear_greed(df)
    df = df.dropna().reset_index(drop=True)
    return df
