"""
model.py — Single-model trainer used by ensemble.py
Exports: train_single(), predict_single()
"""
import warnings
import logging
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import CrossEntropy
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

def prepare_dataset(df, horizon=4, threshold=1.008):
    """Prepare and label the dataframe for TFT training.
    horizon: candles ahead to predict
    threshold: price move required to be labelled UP
    Target is integer (0 or 1) for CrossEntropy classification.
    sample_weight rebalances UP/DOWN classes so the model sees equal importance.
    """
    df = df.copy()
    # Integer target required for CrossEntropy loss
    df["target"] = (df["close"].shift(-horizon) > df["close"] * threshold).astype(int)
    df = df.dropna().reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df["group"] = "BTC"
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(float)
        else:
            df[col] = 0.0
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Class-balanced sample weights: UP samples get weight n_down/n_up,
    # DOWN samples get weight 1.0.  Fixes the model's tendency to always
    # predict DOWN when the training set is >70% DOWN (bear-market data).
    n_up   = max(int(df["target"].sum()), 1)
    n_down = max(len(df) - n_up, 1)
    df["sample_weight"] = df["target"].map({1: n_down / n_up, 0: 1.0}).astype(float)

    return df

def make_dataset(df, cutoff, encoder_length=168):
    """Create a TimeSeriesDataSet for training or validation."""
    weight_col = "sample_weight" if "sample_weight" in df.columns else None
    return TimeSeriesDataSet(
        df[df.time_idx <= cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=encoder_length,
        max_prediction_length=1,
        time_varying_unknown_reals=FEATURES,
        target_normalizer=None,
        weight=weight_col,
        allow_missing_timesteps=True,
    )

def _get_prob(raw):
    """Extract UP probability from CrossEntropy model output."""
    logits = raw["prediction"][:, 0, :]   # shape [batch, 2]
    probs = F.softmax(logits, dim=-1)
    return probs[:, 1].cpu().numpy()       # probability of class 1 (UP)

def compute_calibration(model, val_loader):
    """Compute calibration buckets on the validation set."""
    model.eval()
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, (y, weight) = batch
            raw = model(x)
            scores = _get_prob(raw)
            targets = y[:, 0].cpu().numpy()
            all_preds.extend(scores.tolist())
            all_actuals.extend(targets.tolist())

    buckets = {k: {"preds": [], "actuals": []} for k in
               ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]}

    for pred, actual in zip(all_preds, all_actuals):
        p = max(0.0, min(1.0, pred))
        if p >= 0.9:   key = "0.9-1.0"
        elif p >= 0.8: key = "0.8-0.9"
        elif p >= 0.7: key = "0.7-0.8"
        elif p >= 0.6: key = "0.6-0.7"
        elif p >= 0.5: key = "0.5-0.6"
        else: continue
        buckets[key]["preds"].append(p)
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
    if score >= 0.9:   key = "0.9-1.0"
    elif score >= 0.8: key = "0.8-0.9"
    elif score >= 0.7: key = "0.7-0.8"
    elif score >= 0.6: key = "0.6-0.7"
    elif score >= 0.5: key = "0.5-0.6"
    else: return score
    cal_val = calibration.get(key)
    return cal_val if cal_val is not None else score

def train_single(df, encoder_length, model_path, label="Model", horizon=4, threshold=1.008):
    """
    Train a single TFT model with CrossEntropy (binary classification).
    Returns (model, training_dataset, val_accuracy, calibration).
    """
    print(f"  [{label}] Preparing dataset (encoder={encoder_length}, horizon={horizon}, threshold={threshold})...")
    df = prepare_dataset(df, horizon=horizon, threshold=threshold)
    up_pct = df["target"].mean() * 100
    print(f"  [{label}] Rows: {len(df)}  |  UP: {up_pct:.1f}%  |  DOWN: {100-up_pct:.1f}%")

    training_cutoff = int(len(df) * 0.8)
    print(f"  [{label}] Train: {training_cutoff} rows  |  Val: {len(df) - training_cutoff} rows")

    training = make_dataset(df, training_cutoff, encoder_length=encoder_length)
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader   = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    print(f"  [{label}] Building TFT model (CrossEntropy, 2 classes, features={len(FEATURES)})...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=128,
        attention_head_size=8,
        dropout=0.3,
        hidden_continuous_size=64,
        loss=CrossEntropy(),
        output_size=2,
    )
    total_params = sum(p.numel() for p in tft.parameters())
    print(f"  [{label}] Parameters: {total_params:,}  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  [{label}] Epochs: 30  |  Batch: 64  |  hidden: 128  |  dropout: 0.3  |  Loss: CrossEntropy")
    print(f"  [{label}] Starting training...")

    trainer = Trainer(
        max_epochs=30,
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

    # Validation accuracy
    tft.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            x, (y, weight) = batch
            raw = tft(x)
            scores = _get_prob(raw)
            targets = y[:, 0].cpu().numpy()
            preds = (scores >= 0.5).astype(int)
            correct += (preds == targets.astype(int)).sum()
            total += len(targets)
    val_accuracy = round(correct / total * 100, 2) if total > 0 else 0.0
    print(f"  [{label}] Validation accuracy: {val_accuracy}%")

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
    dataset = TimeSeriesDataSet.from_dataset(training_dataset, df, predict=True, stop_randomization=True)
    loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    model.eval()

    with torch.no_grad():
        raw = model.predict(loader, mode="raw")

    logits = raw["prediction"][0, 0, :]          # shape [2]
    probs = F.softmax(logits, dim=-1)
    score = float(probs[1].item())                # P(UP)
    score = max(0.0, min(1.0, score))

    calibrated = apply_calibration(score, calibration)

    if calibrated >= 0.5:
        return {"direction": "UP",   "confidence": round(calibrated * 100, 1), "raw_score": round(score, 4)}
    else:
        return {"direction": "DOWN", "confidence": round((1.0 - calibrated) * 100, 1), "raw_score": round(score, 4)}
