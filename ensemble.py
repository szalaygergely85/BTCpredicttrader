"""
ensemble.py — Trains and manages 3 TFT models with different encoder lengths.

Model A: encoder=168h (7 days),  trained on full dataset     → tft_a.ckpt
Model B: encoder=168h (7 days),  trained on most recent 50%  → tft_b.ckpt
Model C: encoder=336h (14 days), trained on full dataset     → tft_c.ckpt

Voting: majority vote for direction, average confidence of agreeing models.
meta.json stores last_trained date and validation stats per model.
"""
import json
import os
import warnings
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from data import get_training_data
from model import (
    FEATURES, prepare_dataset, make_dataset,
    train_single, predict_single, apply_calibration
)

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

META_PATH = "meta.json"
OLD_MODEL_PATH = "tft_btc.ckpt"

MODELS = [
    {
        "label": "Model A",
        "path": "tft_a.ckpt",
        "encoder_length": 168,
        "data_slice": "full",
    },
    {
        "label": "Model B",
        "path": "tft_b.ckpt",
        "encoder_length": 168,
        "data_slice": "recent_50pct",
    },
    {
        "label": "Model C",
        "path": "tft_c.ckpt",
        "encoder_length": 336,
        "data_slice": "full",
    },
]

# Cached state
_models_state = {}  # label -> {training_dataset, calibration}

def load_meta():
    if not os.path.exists(META_PATH):
        return {}
    try:
        with open(META_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def save_meta(meta):
    tmp = META_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, META_PATH)

def needs_training(meta):
    """Check if any model is missing or older than 7 days."""
    for m in MODELS:
        if not os.path.exists(m["path"]):
            return True
        model_meta = meta.get(m["label"], {})
        last_trained_str = model_meta.get("last_trained")
        if not last_trained_str:
            return True
        last_trained = datetime.fromisoformat(last_trained_str)
        if datetime.now() - last_trained > timedelta(days=7):
            return True
    return False

def _get_data_slice(df, data_slice):
    """Return the appropriate slice of the dataframe."""
    if data_slice == "full":
        return df
    elif data_slice == "recent_50pct":
        cutoff_idx = len(df) // 2
        return df.iloc[cutoff_idx:].reset_index(drop=True)
    return df

def train_all(df=None):
    """Train all 3 models. Fetches data if not provided."""
    global _models_state

    if df is None:
        print("=" * 60)
        print("Fetching training data (3 years)...")
        df = get_training_data()

    print("=" * 60)
    print("TRAINING ENSEMBLE (3 TFT Models)")
    print("  WARNING: Training 3 models will take approximately 30-45 minutes.")
    print("=" * 60)

    meta = load_meta()

    for m_cfg in MODELS:
        label = m_cfg["label"]
        path = m_cfg["path"]
        encoder_length = m_cfg["encoder_length"]
        data_slice = m_cfg["data_slice"]

        print(f"\n{'='*60}")
        print(f"  Training {label}  (encoder={encoder_length}h, data={data_slice})")
        print(f"{'='*60}")

        df_slice = _get_data_slice(df, data_slice)
        tft, training_dataset, val_accuracy, calibration = train_single(
            df_slice, encoder_length, path, label=label,
            horizon=4, threshold=1.004,
        )

        _models_state[label] = {
            "training_dataset": training_dataset,
            "calibration": calibration,
        }

        if label not in meta:
            meta[label] = {}
        meta[label]["last_trained"] = datetime.now().isoformat()
        meta[label]["val_accuracy"] = val_accuracy
        meta[label]["calibration"] = calibration
        meta[label]["encoder_length"] = encoder_length
        meta[label]["data_slice"] = data_slice
        save_meta(meta)
        print(f"  [{label}] Done. Val accuracy: {val_accuracy}%")

    # Remove old single-model checkpoint if present
    if os.path.exists(OLD_MODEL_PATH):
        os.remove(OLD_MODEL_PATH)
        print(f"\n  Removed old {OLD_MODEL_PATH} (replaced by tft_a/b/c.ckpt)")

    print("\n" + "=" * 60)
    print("  All 3 models trained and saved.")
    print("=" * 60)

def load_all(df=None):
    """
    Load all 3 models from checkpoints (re-building training datasets).
    Fetches training data if not provided.
    """
    global _models_state

    if df is None:
        print("  Loading training data to rebuild dataset schemas...")
        df = get_training_data()

    meta = load_meta()

    for m_cfg in MODELS:
        label = m_cfg["label"]
        path = m_cfg["path"]
        encoder_length = m_cfg["encoder_length"]
        data_slice = m_cfg["data_slice"]

        df_slice = _get_data_slice(df, data_slice)
        df_prep = prepare_dataset(df_slice, horizon=4, threshold=1.004)
        training_cutoff = int(len(df_prep) * 0.8)
        training_dataset = make_dataset(df_prep, training_cutoff, encoder_length=encoder_length)

        model_meta = meta.get(label, {})
        calibration = model_meta.get("calibration", None)
        val_accuracy = model_meta.get("val_accuracy", None)

        _models_state[label] = {
            "training_dataset": training_dataset,
            "calibration": calibration,
        }

        if val_accuracy is not None:
            print(f"  [{label}] Loaded. Val accuracy: {val_accuracy}%")
        else:
            print(f"  [{label}] Loaded.")

def load_or_train_all():
    """Main entry point: train if needed, otherwise load existing models."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This tool requires an NVIDIA GPU.")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    meta = load_meta()

    if needs_training(meta):
        print("One or more models missing or outdated. Starting training...")
        train_all()
    else:
        print("Loading existing ensemble models...")
        load_all()

    # Print validation stats
    meta = load_meta()
    print("\n  Ensemble model stats:")
    for m_cfg in MODELS:
        label = m_cfg["label"]
        model_meta = meta.get(label, {})
        val_acc = model_meta.get("val_accuracy", "n/a")
        last_trained = model_meta.get("last_trained", "unknown")
        print(f"    {label}: val_accuracy={val_acc}%  trained={last_trained[:10]}")

def predict_ensemble(df):
    """
    Run predictions on all 3 models and combine via majority vote.
    Returns dict with direction, confidence, votes.
    """
    global _models_state

    results = []
    for m_cfg in MODELS:
        label = m_cfg["label"]
        path = m_cfg["path"]
        state = _models_state.get(label)
        if state is None:
            print(f"  [{label}] Not loaded, skipping.")
            continue
        try:
            result = predict_single(
                df,
                state["training_dataset"],
                path,
                calibration=state["calibration"],
            )
            result["label"] = label
            results.append(result)
        except Exception as e:
            print(f"  [{label}] Prediction error: {e}")

    if not results:
        return {"direction": "HOLD", "confidence": 50.0, "votes": {}}

    # Majority vote
    up_votes = [r for r in results if r["direction"] == "UP"]
    down_votes = [r for r in results if r["direction"] == "DOWN"]

    if len(up_votes) > len(down_votes):
        direction = "UP"
        agreeing = up_votes
    elif len(down_votes) > len(up_votes):
        direction = "DOWN"
        agreeing = down_votes
    else:
        # Tie: go with the average score
        avg_score = sum(r["raw_score"] for r in results) / len(results)
        direction = "UP" if avg_score >= 0.5 else "DOWN"
        agreeing = results

    avg_confidence = sum(r["confidence"] for r in agreeing) / len(agreeing)

    votes = {r["label"]: r["direction"] for r in results}
    confs = {r["label"]: r["confidence"] for r in results}

    return {
        "direction": direction,
        "confidence": round(avg_confidence, 1),
        "votes": votes,
        "confidences": confs,
    }
