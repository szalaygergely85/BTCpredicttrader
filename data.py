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
        resp = requests.get("https://api.alternative.me/fng/?limit=1500&format=json", timeout=10)
        data = resp.json()["data"]
        fg = pd.DataFrame(data)[["timestamp", "value"]]
        fg["date"] = pd.to_datetime(fg["timestamp"].astype(int), unit="s").dt.normalize()
        fg["fear_greed"] = fg["value"].astype(float)
        return fg[["date", "fear_greed"]].sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"  Fear & Greed fetch failed ({e}), using neutral 50")
        return None

def get_order_book_imbalance():
    """Fetch real-time order book imbalance from Binance. Returns value from -1 to +1."""
    try:
        resp = requests.get("https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=20", timeout=5)
        data = resp.json()
        bid_vol = sum(float(b[1]) for b in data["bids"])
        ask_vol = sum(float(a[1]) for a in data["asks"])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)  # -1 to +1
        return round(imbalance, 4)
    except Exception as e:
        print(f"  Order book fetch failed ({e}), using 0.0")
        return 0.0

def get_funding_rates(limit=3000):
    """Fetch historical 8h funding rates from Binance futures"""
    all_data = []
    end_time = None
    remaining = limit
    while remaining > 0:
        batch = min(remaining, 1000)
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit={batch}"
        if end_time:
            url += f"&endTime={end_time}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
        except Exception as e:
            print(f"  Funding rate fetch error: {e}")
            break
        if not data:
            break
        all_data = data + all_data
        remaining -= len(data)
        end_time = data[0]["fundingTime"] - 1
        time.sleep(0.1)
    if not all_data:
        return None
    df = pd.DataFrame(all_data)
    df["time"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["time", "funding_rate"]].sort_values("time").reset_index(drop=True)

def add_funding_rate(df):
    """Merge funding rates into hourly candles by forward-fill."""
    try:
        print("  Fetching funding rates from Binance Futures...")
        funding = get_funding_rates(limit=3000)
        if funding is None or funding.empty:
            print("  Funding rate data unavailable, using 0.0")
            df["funding_rate"] = 0.0
            return df
        df = pd.merge_asof(
            df.sort_values("time"),
            funding.sort_values("time"),
            on="time",
            direction="backward"
        )
        df["funding_rate"] = df["funding_rate"].fillna(0.0)
        print(f"  Funding rates merged ({len(funding)} records).")
        return df
    except Exception as e:
        print(f"  Funding rate merge failed ({e}), using 0.0")
        df["funding_rate"] = 0.0
        return df

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
    # Regime detection features
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["ema_ratio"] = df["ema20"] / df["ema50"]  # >1 = bullish trend, <1 = bearish
    df = df.dropna().reset_index(drop=True)
    return df

def add_4h_features(df_1h, limit=6570):
    print(f"  Fetching 4h candles (limit={limit})...")
    df_4h = get_candles(Client.KLINE_INTERVAL_4HOUR, limit)
    print(f"  Fetched {len(df_4h)} 4h candles. Computing 4h indicators...")
    df_4h["rsi_4h"] = ta.momentum.RSIIndicator(df_4h["close"], window=14).rsi()
    macd_4h = ta.trend.MACD(df_4h["close"])
    df_4h["macd_diff_4h"] = macd_4h.macd_diff()
    df_4h["ema20_4h"] = ta.trend.EMAIndicator(df_4h["close"], window=20).ema_indicator()
    df_4h["ema50_4h"] = ta.trend.EMAIndicator(df_4h["close"], window=50).ema_indicator()
    df_4h = df_4h[["time","rsi_4h","macd_diff_4h","ema20_4h","ema50_4h"]].dropna()
    df = pd.merge_asof(df_1h.sort_values("time"), df_4h.sort_values("time"), on="time", direction="backward")
    return df

def add_fear_greed(df):
    print("  Fetching Fear & Greed Index...")
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
    print("  Fetching 1h candles (26280 = 3 years, paginated)...")
    df = get_candles(Client.KLINE_INTERVAL_1HOUR, 26280)  # 3 years
    print(f"  Fetched {len(df)} raw 1h candles. Adding indicators...")
    df = add_indicators(df)
    print(f"  After indicators: {len(df)} rows.")
    print("  Fetching 4h candles for multi-timeframe features...")
    df = add_4h_features(df, limit=6570)
    print("  Merging Fear & Greed Index...")
    df = add_fear_greed(df)
    print("  Merging funding rates...")
    df = add_funding_rate(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns")
    return df

def get_live_data():
    print("  Fetching live 1h candles...")
    df = get_candles(Client.KLINE_INTERVAL_1HOUR, 600)
    df = add_indicators(df)
    print("  Adding 4h features for live data...")
    df = add_4h_features(df, limit=200)
    df = add_fear_greed(df)
    df = add_funding_rate(df)
    df = df.dropna().reset_index(drop=True)
    return df

def get_scalp_training_data():
    """6 months of 5-min candles for scalping model training."""
    limit = 52560  # ~6 months of 5-min candles (bull + bear coverage)
    print(f"  Fetching 5-min candles ({limit} = ~6 months, paginated)...")
    df = get_candles(Client.KLINE_INTERVAL_5MINUTE, limit)
    print(f"  Fetched {len(df)} raw 5-min candles. Adding indicators...")
    df = add_indicators(df)
    df = add_fear_greed(df)
    df = add_funding_rate(df)
    df_4h = get_candles(Client.KLINE_INTERVAL_4HOUR, 1100)
    df_4h["rsi_4h"] = ta.momentum.RSIIndicator(df_4h["close"], window=14).rsi()
    macd_4h = ta.trend.MACD(df_4h["close"])
    df_4h["macd_diff_4h"] = macd_4h.macd_diff()
    df_4h["ema20_4h"] = ta.trend.EMAIndicator(df_4h["close"], window=20).ema_indicator()
    df_4h["ema50_4h"] = ta.trend.EMAIndicator(df_4h["close"], window=50).ema_indicator()
    df_4h = df_4h[["time","rsi_4h","macd_diff_4h","ema20_4h","ema50_4h"]].dropna()
    df = pd.merge_asof(df.sort_values("time"), df_4h.sort_values("time"), on="time", direction="backward")
    df = df.dropna().reset_index(drop=True)
    print(f"  Final scalp dataset: {len(df)} rows")
    return df

def get_scalp_live_data():
    """Latest 5-min candles for scalping prediction."""
    df = get_candles(Client.KLINE_INTERVAL_5MINUTE, 500)
    df = add_indicators(df)
    df = add_fear_greed(df)
    df = add_funding_rate(df)
    df_4h = get_candles(Client.KLINE_INTERVAL_4HOUR, 100)
    df_4h["rsi_4h"] = ta.momentum.RSIIndicator(df_4h["close"], window=14).rsi()
    macd_4h = ta.trend.MACD(df_4h["close"])
    df_4h["macd_diff_4h"] = macd_4h.macd_diff()
    df_4h["ema20_4h"] = ta.trend.EMAIndicator(df_4h["close"], window=20).ema_indicator()
    df_4h["ema50_4h"] = ta.trend.EMAIndicator(df_4h["close"], window=50).ema_indicator()
    df_4h = df_4h[["time","rsi_4h","macd_diff_4h","ema20_4h","ema50_4h"]].dropna()
    df = pd.merge_asof(df.sort_values("time"), df_4h.sort_values("time"), on="time", direction="backward")
    df = df.dropna().reset_index(drop=True)
    return df
