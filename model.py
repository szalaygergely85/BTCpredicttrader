import json
import os
import warnings
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from data import get_training_data

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

torch.set_float32_matmul_precision("medium")

MODEL_PATH = "tft_btc.ckpt"
META_PATH = "meta.json"

FEATURES = [
    # original
    "rsi", "macd", "macd_diff", "bb_upper", "bb_lower", "bb_width",
    "atr", "ema20", "ema50", "volume_delta", "hour", "day_of_week",
    # new 1h indicators
    "obv", "stoch_rsi", "vwap",
    # 4h multi-timeframe
    "rsi_4h", "macd_diff_4h", "ema20_4h", "ema50_4h",
    # sentiment
    "fear_greed",
]

def should_retrain():
    if not os.path.exists(META_PATH) or not os.path.exists(MODEL_PATH):
        return True
    with open(META_PATH) as f:
        meta = json.load(f)
    last = datetime.fromisoformat(meta["last_trained"])
    return datetime.now() - last > timedelta(days=7)

def save_meta():
    with open(META_PATH, "w") as f:
        json.dump({"last_trained": datetime.now().isoformat()}, f)

def prepare_dataset(df):
    df = df.copy()
    # improvement 1: predict >0.5% gain over next 2 candles — filters noise
    df["target"] = (df["close"].shift(-2) > df["close"] * 1.005).astype(float)
    df = df.dropna().reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df["group"] = "BTC"
    for col in FEATURES:
        df[col] = df[col].astype(float)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def make_dataset(df, cutoff):
    return TimeSeriesDataSet(
        df[df.time_idx <= cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=168,
        max_prediction_length=1,
        time_varying_unknown_reals=FEATURES,
        target_normalizer=None,
    )

def train():
    print("=" * 60)
    print("TRAINING TFT MODEL  (all 5 improvements active)")
    df = get_training_data()
    print(f"  Preparing dataset...")
    df = prepare_dataset(df)
    up_pct = df["target"].mean() * 100
    print(f"  Rows: {len(df)}  |  UP (>0.5% in 2h): {up_pct:.1f}%")

    # improvement 2: 80/20 validation split
    training_cutoff = int(len(df) * 0.8)
    print(f"  Train: {training_cutoff} rows  |  Val: {len(df) - training_cutoff} rows")

    training = make_dataset(df, training_cutoff)
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    # improvement 3+4+5: more features (in FEATURES list) + larger model
    print("  Building larger TFT model...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.003,
        hidden_size=128,           # was 64
        attention_head_size=8,     # was 4
        dropout=0.1,
        hidden_continuous_size=64, # was 32
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        output_size=3,
    )
    total_params = sum(p.numel() for p in tft.parameters())
    print(f"  Parameters: {total_params:,}  |  Features: {len(FEATURES)}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Epochs: 40  |  Batch: 64  |  hidden: 128  |  heads: 8")
    print("  Starting training...")
    print("=" * 60)

    trainer = Trainer(
        max_epochs=40,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
        gradient_clip_val=0.1,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(MODEL_PATH)
    save_meta()

    print("=" * 60)
    print("  Model trained and saved to tft_btc.ckpt")
    print("=" * 60)
    return tft, training

_model = None
_training_dataset = None

def load_or_train():
    global _model, _training_dataset
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This tool requires an NVIDIA GPU.")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    if should_retrain():
        _model, _training_dataset = train()
    else:
        print("Loading existing model...")
        df = get_training_data()
        df = prepare_dataset(df)
        training_cutoff = int(len(df) * 0.8)
        _training_dataset = make_dataset(df, training_cutoff)
        _model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
        _model.eval()
        print("  Model loaded and ready.")

def predict(df):
    global _model, _training_dataset
    df = prepare_dataset(df)
    df["time_idx"] = range(len(df))
    df["group"] = "BTC"
    dataset = TimeSeriesDataSet.from_dataset(_training_dataset, df, predict=True, stop_randomization=True)
    loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    with torch.no_grad():
        raw = _model.predict(loader, mode="raw")
    # index 1 = median quantile (0.5)
    score = float(raw["prediction"][0, 0, 1])
    score = max(0.0, min(1.0, score))
    if score >= 0.5:
        return {"direction": "UP", "confidence": round(score * 100, 1)}
    else:
        return {"direction": "DOWN", "confidence": round((1.0 - score) * 100, 1)}
