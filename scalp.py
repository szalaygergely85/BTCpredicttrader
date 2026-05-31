"""
scalp.py — 3-model TFT ensemble for scalping mode (5-min candles).

Model SA: encoder=60  (5h),  trained on full 6-month dataset
Model SB: encoder=60  (5h),  trained on most recent 50%
Model SC: encoder=120 (10h), trained on full 6-month dataset

Mirrors the swing ensemble structure but tuned for 5-min candles.
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

META_PATH = "meta_scalp.json"
HORIZON = 3        # 3 × 5min = 15min ahead
THRESHOLD = 1.001  # >0.1% move labelled UP — balanced across bull+bear

MODELS = [
    {"label": "Scalp A", "path": "tft_scalp_a.ckpt", "encoder_length": 60,  "data_slice": "full"},
    {"label": "Scalp B", "path": "tft_scalp_b.ckpt", "encoder_length": 60,  "data_slice": "recent_50pct"},
    {"label": "Scalp C", "path": "tft_scalp_c.ckpt", "encoder_length": 120, "data_slice": "full"},
]

_models_state = {}  # label -> {training_dataset, calibration}


def _load_meta():
    if not os.path.exists(META_PATH):
        return {}
    try:
        with open(META_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_meta(meta):
    tmp = META_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, META_PATH)

def _needs_training():
    meta = _load_meta()
    for m in MODELS:
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

def _train_all(df):
    global _models_state
    meta = _load_meta()
    print("=" * 60)
    print("TRAINING SCALP ENSEMBLE (3 TFT Models — 5-min candles)")
    print("  WARNING: ~45-60 minutes total.")
    print("=" * 60)

    for m in MODELS:
        label, path, encoder_length, data_slice = m["label"], m["path"], m["encoder_length"], m["data_slice"]
        print(f"\n{'='*60}")
        print(f"  Training {label}  (encoder={encoder_length}, data={data_slice})")
        print(f"{'='*60}")
        df_slice = _get_slice(df, data_slice)
        tft, training_dataset, val_acc, calibration = train_single(
            df_slice, encoder_length, path, label=label,
            horizon=HORIZON, threshold=THRESHOLD,
        )
        _models_state[label] = {"training_dataset": training_dataset, "calibration": calibration}
        if label not in meta:
            meta[label] = {}
        meta[label]["last_trained"] = datetime.now().isoformat()
        meta[label]["val_accuracy"] = val_acc
        meta[label]["calibration"] = calibration
        meta[label]["encoder_length"] = encoder_length
        _save_meta(meta)
        print(f"  [{label}] Done. Val accuracy: {val_acc}%")

    print("\n" + "=" * 60)
    print("  All 3 scalp models trained and saved.")
    print("=" * 60)

def _load_all(df):
    global _models_state
    meta = _load_meta()
    for m in MODELS:
        label, path, encoder_length, data_slice = m["label"], m["path"], m["encoder_length"], m["data_slice"]
        df_slice = _get_slice(df, data_slice)
        df_prep = prepare_dataset(df_slice, horizon=HORIZON, threshold=THRESHOLD)
        cutoff = int(len(df_prep) * 0.8)
        training_dataset = make_dataset(df_prep, cutoff, encoder_length=encoder_length)
        model_meta = meta.get(label, {})
        _models_state[label] = {
            "training_dataset": training_dataset,
            "calibration": model_meta.get("calibration"),
        }
        val_acc = model_meta.get("val_accuracy", "n/a")
        print(f"  [{label}] Loaded. Val accuracy: {val_acc}%")

def load_or_train(df=None):
    """Main entry point: train if missing/stale, otherwise load."""
    if _needs_training():
        if df is None:
            from data import get_scalp_training_data
            print("[Scalp] Fetching 6-month scalp training data...")
            df = get_scalp_training_data()
        _train_all(df)
    else:
        print("[Scalp] Loading existing scalp ensemble...")
        if df is None:
            from data import get_scalp_training_data
            print("[Scalp] Reconstructing dataset schemas...")
            df = get_scalp_training_data()
        _load_all(df)

    meta = _load_meta()
    print("\n  Scalp ensemble stats:")
    for m in MODELS:
        m_meta = meta.get(m["label"], {})
        val_acc = m_meta.get("val_accuracy", "n/a")
        trained = m_meta.get("last_trained", "unknown")
        print(f"    {m['label']}: val_accuracy={val_acc}%  trained={trained[:10]}")

def predict(df):
    """
    Majority vote across 3 scalp models.
    Returns dict with direction, confidence, votes.
    """
    global _models_state
    results = []
    for m in MODELS:
        label, path = m["label"], m["path"]
        state = _models_state.get(label)
        if state is None:
            continue
        try:
            result = predict_single(df, state["training_dataset"], path, calibration=state["calibration"])
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
        agreeing = results

    confidence = sum(r["confidence"] for r in agreeing) / len(agreeing)
    votes = {r["label"]: r["direction"] for r in results}

    return {
        "direction": direction,
        "confidence": round(confidence, 1),
        "votes": votes,
    }
