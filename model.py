"""
model.py — Single-model trainer used by ensemble.py
Exports: train_single(), predict_single()
"""
import json
import os
import warnings
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

torch.set_float32_matmul_precision("medium")

FEATURES = [
    # base 1h indicators
    "rsi", "macd", "macd_diff", "bb_upper", "bb_lower", "bb_width",
    "atr", "ema20", "ema50", "volume_delta", "hour", "day_of_week",
    # additional 1h indicators
    "obv", "stoch_rsi", "vwap",
    # 4h multi-timeframe
    "rsi_4h", "macd_diff_4h", "ema20_4h", "ema50_4h",
    # sentiment
    "fear_greed",
    # regime detection
    "ema200", "adx", "ema_ratio",
    # funding rate
    "funding_rate",
]

def prepare_dataset(df):
    """Prepare and label the dataframe for TFT training."""
    df = df.copy()
    # Improvement 1: predict >0.8% gain over next 4 candles (4h horizon)
    df["target"] = (df["close"].shift(-4) > df["close"] * 1.008).astype(float)
    df = df.dropna().reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df["group"] = "BTC"
    for col in FEATURES:
        df[col] = df[col].astype(float)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def make_dataset(df, cutoff, encoder_length=168):
    """Create a TimeSeriesDataSet for training or validation."""
    return TimeSeriesDataSet(
        df[df.time_idx <= cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=encoder_length,
        max_prediction_length=1,
        time_varying_unknown_reals=FEATURES,
        target_normalizer=None,
    )

def compute_calibration(model, val_loader):
    """
    Compute calibration buckets on the validation set.
    Returns a dict: bucket_label -> actual_accuracy
    """
    model.eval()
    all_preds = []
    all_actuals = []
    with torch.no_grad():
        for batch in val_loader:
            x, (y, weight) = batch
            raw = model(x)
            scores = raw["prediction"][:, 0, 1].cpu().numpy()
            targets = y[:, 0].cpu().numpy()
            all_preds.extend(scores.tolist())
            all_actuals.extend(targets.tolist())

    buckets = {
        "0.5-0.6": {"preds": [], "actuals": []},
        "0.6-0.7": {"preds": [], "actuals": []},
        "0.7-0.8": {"preds": [], "actuals": []},
        "0.8-0.9": {"preds": [], "actuals": []},
        "0.9-1.0": {"preds": [], "actuals": []},
    }
    for pred, actual in zip(all_preds, all_actuals):
        pred_clamped = max(0.0, min(1.0, pred))
        if pred_clamped >= 0.9:
            key = "0.9-1.0"
        elif pred_clamped >= 0.8:
            key = "0.8-0.9"
        elif pred_clamped >= 0.7:
            key = "0.7-0.8"
        elif pred_clamped >= 0.6:
            key = "0.6-0.7"
        elif pred_clamped >= 0.5:
            key = "0.5-0.6"
        else:
            continue
        buckets[key]["preds"].append(pred_clamped)
        buckets[key]["actuals"].append(actual)

    calibration = {}
    print("  Calibration (predicted bucket -> actual accuracy):")
    for key, vals in buckets.items():
        if vals["actuals"]:
            acc = sum(vals["actuals"]) / len(vals["actuals"])
            calibration[key] = round(acc, 4)
            print(f"    {key}: {acc*100:.1f}% ({len(vals['actuals'])} samples)")
        else:
            calibration[key] = None
            print(f"    {key}: no samples")
    return calibration

def apply_calibration(raw_score, calibration):
    """Map a raw model score through the calibration table."""
    if calibration is None:
        return raw_score
    score = max(0.0, min(1.0, raw_score))
    if score >= 0.9:
        key = "0.9-1.0"
    elif score >= 0.8:
        key = "0.8-0.9"
    elif score >= 0.7:
        key = "0.7-0.8"
    elif score >= 0.6:
        key = "0.6-0.7"
    elif score >= 0.5:
        key = "0.5-0.6"
    else:
        return score
    cal_val = calibration.get(key)
    if cal_val is None:
        return score
    return cal_val

def train_single(df, encoder_length, model_path, label="Model"):
    """
    Train a single TFT model and save to model_path.
    Returns (model, training_dataset, val_accuracy, calibration).
    """
    print(f"  [{label}] Preparing dataset (encoder_length={encoder_length})...")
    df = prepare_dataset(df)
    up_pct = df["target"].mean() * 100
    print(f"  [{label}] Rows: {len(df)}  |  UP (>0.8% in 4h): {up_pct:.1f}%")

    # 80/20 walk-forward split
    training_cutoff = int(len(df) * 0.8)
    print(f"  [{label}] Train: {training_cutoff} rows  |  Val: {len(df) - training_cutoff} rows")

    training = make_dataset(df, training_cutoff, encoder_length=encoder_length)
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    print(f"  [{label}] Building TFT model (features={len(FEATURES)})...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.003,
        hidden_size=128,
        attention_head_size=8,
        dropout=0.1,
        hidden_continuous_size=64,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        output_size=3,
    )
    total_params = sum(p.numel() for p in tft.parameters())
    print(f"  [{label}] Parameters: {total_params:,}  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  [{label}] Epochs: 40  |  Batch: 64  |  hidden: 128  |  heads: 8")
    print(f"  [{label}] Starting training...")

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
    trainer.save_checkpoint(model_path)
    print(f"  [{label}] Saved to {model_path}")

    # Compute validation accuracy
    tft.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            x, (y, weight) = batch
            raw = tft(x)
            scores = raw["prediction"][:, 0, 1].cpu().numpy()
            targets = y[:, 0].cpu().numpy()
            preds = (scores >= 0.5).astype(float)
            correct += (preds == targets).sum()
            total += len(targets)
    val_accuracy = round(correct / total * 100, 2) if total > 0 else 0.0
    print(f"  [{label}] Validation accuracy: {val_accuracy}%")

    # Compute calibration
    calibration = compute_calibration(tft, val_loader)

    return tft, training, val_accuracy, calibration

def predict_single(df, training_dataset, model_path, calibration=None):
    """
    Run prediction using saved model at model_path.
    Returns dict with direction, confidence, raw_score.
    """
    df = prepare_dataset(df)
    df["time_idx"] = range(len(df))
    df["group"] = "BTC"
    dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, df, predict=True, stop_randomization=True
    )
    loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    model.eval()

    with torch.no_grad():
        raw = model.predict(loader, mode="raw")
    score = float(raw["prediction"][0, 0, 1])
    score = max(0.0, min(1.0, score))

    # Apply calibration if available
    calibrated_score = apply_calibration(score, calibration)

    if calibrated_score >= 0.5:
        return {
            "direction": "UP",
            "confidence": round(calibrated_score * 100, 1),
            "raw_score": round(score, 4),
        }
    else:
        return {
            "direction": "DOWN",
            "confidence": round((1.0 - calibrated_score) * 100, 1),
            "raw_score": round(score, 4),
        }
