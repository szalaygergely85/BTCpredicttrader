"""
shares.py — 3-model TFT ensemble for any stock ticker (hourly candles).

Model A: encoder=48h  (~7 trading days), full data
Model B: encoder=48h  (~7 trading days), recent 50%
Model C: encoder=96h  (~14 trading days), full data

Usage: imported by main.py when -shares TICKER is passed.
"""
import json
import os
import warnings
import logging
from datetime import datetime, timedelta

from model import FEATURES, prepare_dataset, make_dataset, train_single, predict_single

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

HORIZON    = 4      # 4 hours ahead
THRESHOLD  = 1.005  # +0.5% move labelled UP  (stocks are less volatile than BTC)

# Per-ticker state stored here at runtime
_models_state = {}   # ticker -> {label -> {training_dataset, calibration}}


def _meta_path(ticker):
    return f"meta_{ticker.lower()}.json"

def _model_configs(ticker):
    t = ticker.lower()
    return [
        {"label": f"{ticker} A", "path": f"tft_{t}_a.ckpt", "encoder_length": 48,  "data_slice": "full"},
        {"label": f"{ticker} B", "path": f"tft_{t}_b.ckpt", "encoder_length": 48,  "data_slice": "recent_50pct"},
        {"label": f"{ticker} C", "path": f"tft_{t}_c.ckpt", "encoder_length": 96,  "data_slice": "full"},
    ]


def _load_meta(ticker):
    path = _meta_path(ticker)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_meta(ticker, meta):
    path = _meta_path(ticker)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)

def _needs_training(ticker):
    meta = _load_meta(ticker)
    for m in _model_configs(ticker):
        if not os.path.exists(m["path"]):
            return True
        last_str = meta.get(m["label"], {}).get("last_trained")
        if not last_str:
            return True
        if datetime.now() - datetime.fromisoformat(last_str) > timedelta(days=7):
            return True
    return False

def _get_slice(df, data_slice):
    if data_slice == "recent_50pct":
        return df.iloc[len(df) // 2:].reset_index(drop=True)
    return df

def _train_all(ticker, df):
    global _models_state
    meta = _load_meta(ticker)
    models = _model_configs(ticker)
    print("=" * 60)
    print(f"TRAINING {ticker} ENSEMBLE (3 TFT Models — hourly candles)")
    print("  WARNING: ~30-45 minutes total.")
    print("=" * 60)

    if ticker not in _models_state:
        _models_state[ticker] = {}

    for m in models:
        label, path, encoder_length, data_slice = m["label"], m["path"], m["encoder_length"], m["data_slice"]
        print(f"\n{'='*60}")
        print(f"  Training {label}  (encoder={encoder_length}h, data={data_slice})")
        print(f"{'='*60}")
        df_slice = _get_slice(df, data_slice)
        tft, training_dataset, val_acc, calibration = train_single(
            df_slice, encoder_length, path, label=label,
            horizon=HORIZON, threshold=THRESHOLD,
        )
        _models_state[ticker][label] = {"training_dataset": training_dataset, "calibration": calibration}
        if label not in meta:
            meta[label] = {}
        meta[label]["last_trained"] = datetime.now().isoformat()
        meta[label]["val_accuracy"] = val_acc
        meta[label]["calibration"]  = calibration
        meta[label]["encoder_length"] = encoder_length
        _save_meta(ticker, meta)
        print(f"  [{label}] Done. Val accuracy: {val_acc}%")

    print("\n" + "=" * 60)
    print(f"  All 3 {ticker} models trained and saved.")
    print("=" * 60)

def _load_all(ticker, df):
    global _models_state
    meta = _load_meta(ticker)
    if ticker not in _models_state:
        _models_state[ticker] = {}

    for m in _model_configs(ticker):
        label, path, encoder_length, data_slice = m["label"], m["path"], m["encoder_length"], m["data_slice"]
        df_slice = _get_slice(df, data_slice)
        df_prep  = prepare_dataset(df_slice, horizon=HORIZON, threshold=THRESHOLD)
        cutoff   = int(len(df_prep) * 0.8)
        training_dataset = make_dataset(df_prep, cutoff, encoder_length=encoder_length)
        model_meta = meta.get(label, {})
        _models_state[ticker][label] = {
            "training_dataset": training_dataset,
            "calibration": model_meta.get("calibration"),
        }
        val_acc = model_meta.get("val_accuracy", "n/a")
        print(f"  [{label}] Loaded. Val accuracy: {val_acc}%")


def load_or_train(ticker, df=None):
    """Main entry point: train if missing/stale, otherwise load."""
    ticker = ticker.upper()
    if _needs_training(ticker):
        if df is None:
            from data_shares import get_shares_training_data
            print(f"[{ticker}] Fetching 2-year training data...")
            df = get_shares_training_data(ticker)
        _train_all(ticker, df)
    else:
        print(f"[{ticker}] Loading existing ensemble...")
        if df is None:
            from data_shares import get_shares_training_data
            print(f"[{ticker}] Reconstructing dataset schemas...")
            df = get_shares_training_data(ticker)
        _load_all(ticker, df)

    meta = _load_meta(ticker)
    print(f"\n  {ticker} ensemble stats:")
    for m in _model_configs(ticker):
        m_meta  = meta.get(m["label"], {})
        val_acc = m_meta.get("val_accuracy", "n/a")
        trained = m_meta.get("last_trained", "unknown")[:10]
        print(f"    {m['label']}: val_accuracy={val_acc}%  trained={trained}")


def predict(ticker, df):
    """
    Majority vote across 3 models for the given ticker.
    Returns dict with direction, confidence, votes.
    """
    ticker = ticker.upper()
    state_map = _models_state.get(ticker, {})
    results = []

    for m in _model_configs(ticker):
        label, path = m["label"], m["path"]
        state = state_map.get(label)
        if state is None:
            continue
        try:
            result = predict_single(df, state["training_dataset"], path,
                                    calibration=state["calibration"])
            result["label"] = label
            results.append(result)
        except Exception as e:
            print(f"  [{label}] Prediction error: {e}")

    if not results:
        return {"direction": "HOLD", "confidence": 50.0, "votes": {}}

    up_votes   = [r for r in results if r["direction"] == "UP"]
    down_votes = [r for r in results if r["direction"] == "DOWN"]

    if len(up_votes) > len(down_votes):
        direction, agreeing = "UP", up_votes
    elif len(down_votes) > len(up_votes):
        direction, agreeing = "DOWN", down_votes
    else:
        avg_score = sum(r["raw_score"] for r in results) / len(results)
        direction = "UP" if avg_score >= 0.5 else "DOWN"
        agreeing  = results

    confidence = sum(r["confidence"] for r in agreeing) / len(agreeing)
    votes      = {r["label"]: r["direction"] for r in results}

    return {
        "direction":  direction,
        "confidence": round(confidence, 1),
        "votes":      votes,
    }
