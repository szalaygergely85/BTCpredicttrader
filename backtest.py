"""
backtest.py — Out-of-sample backtester on the last 20% of historical data.

Usage:
    python backtest.py                  # BTC swing (1h, 3-model ensemble)
    python backtest.py -scalping        # BTC scalp (5-min, 3-model ensemble)
    python backtest.py -shares NVDA     # Stock ticker (1h, 3-model ensemble)
"""
import argparse
import math
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

parser = argparse.ArgumentParser()
parser.add_argument("-scalping", action="store_true", help="Backtest BTC scalping ensemble")
parser.add_argument("-shares",   type=str, default=None, metavar="TICKER", help="Stock ticker, e.g. -shares NVDA")
args = parser.parse_args()
SCALPING = args.scalping
TICKER   = args.shares.upper() if args.shares else None
SHARES   = TICKER is not None

FEE_RATE      = 0.001
STARTING_USDT = 100.0
TEMPERATURE   = 1.0   # 1.0 = no scaling (model probs are well-calibrated after softmax fix)

if SHARES:
    from data_shares import get_shares_training_data as _get_data
    def get_data(): return _get_data(TICKER)
    t = TICKER.lower()
    ENSEMBLE_MODELS = [
        {"label": f"{TICKER} A", "path": f"tft_{t}_a.ckpt"},
        {"label": f"{TICKER} B", "path": f"tft_{t}_b.ckpt"},
        {"label": f"{TICKER} C", "path": f"tft_{t}_c.ckpt"},
    ]
    ENCODER_LENGTH   = 48
    HORIZON          = 4
    THRESHOLD        = 1.005
    BUY_CONF         = 70
    SELL_CONF        = 95    # no shorts for stocks — high bar rarely triggers
    RSI_THRESHOLD    = 55
    COOLDOWN_CANDLES = 8     # stocks: ≥8 market hours (~1 day) between entries
    MODE             = f"Shares — {TICKER} (1h, 3-model ensemble)"
    CANDLE_MINUTES   = 60
    ALLOW_SHORTS     = False
    SL_MULT          = 2.5   # wider stop for stocks (gaps, normal volatility)
    TP_MULT          = 2.5   # unused when ALLOW_TP=False
    ALLOW_TP         = False # let winners ride; only exit on SL or SELL signal

elif SCALPING:
    from data import get_scalp_training_data as get_data
    ENSEMBLE_MODELS = [
        {"label": "Scalp A", "path": "tft_scalp_a.ckpt"},
        {"label": "Scalp B", "path": "tft_scalp_b.ckpt"},
        {"label": "Scalp C", "path": "tft_scalp_c.ckpt"},
    ]
    ENCODER_LENGTH   = 60
    HORIZON          = 3
    THRESHOLD        = 1.001
    BUY_CONF         = 70
    SELL_CONF        = 75
    RSI_THRESHOLD    = 50
    COOLDOWN_CANDLES = 6
    MODE             = "Scalp (5-min, 3-model ensemble)"
    CANDLE_MINUTES   = 5
    ALLOW_SHORTS     = True
    SL_MULT          = 1.5
    TP_MULT          = 2.5
    ALLOW_TP         = True
else:
    from data import get_training_data as get_data
    ENSEMBLE_MODELS = [
        {"label": "Model A", "path": "tft_a.ckpt"},
        {"label": "Model B", "path": "tft_b.ckpt"},
        {"label": "Model C", "path": "tft_c.ckpt"},
    ]
    ENCODER_LENGTH   = 168
    HORIZON          = 4
    THRESHOLD        = 1.004
    BUY_CONF         = 70
    SELL_CONF        = 75
    RSI_THRESHOLD    = 55
    COOLDOWN_CANDLES = 4
    MODE             = "Swing (1h, 3-model ensemble)"
    CANDLE_MINUTES   = 60
    ALLOW_SHORTS     = True
    SL_MULT          = 1.5
    TP_MULT          = 2.5
    ALLOW_TP         = True

from model import prepare_dataset, make_dataset, FEATURES


def _temp_scale(p, T):
    """Apply temperature scaling to a probability. T>1 → less extreme."""
    p = max(1e-7, min(1 - 1e-7, p))
    logit = math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(-logit / T))


def _ensemble_vote(scores_per_model, T=1.0):
    """Majority vote across model scores with optional temperature scaling."""
    scaled = [_temp_scale(s, T) for s in scores_per_model]
    up_scores   = [s for s in scaled if s >= 0.5]
    down_scores = [s for s in scaled if s < 0.5]
    if len(up_scores) > len(down_scores):
        return "UP",   sum(up_scores)   / len(up_scores)   * 100
    elif len(down_scores) > len(up_scores):
        return "DOWN", (1 - sum(down_scores) / len(down_scores)) * 100
    else:
        avg = sum(scaled) / len(scaled)
        if avg >= 0.5:
            return "UP",   avg * 100
        else:
            return "DOWN", (1 - avg) * 100


def _sim(all_scores, test_df, buy_conf, sell_conf, rsi_threshold, cooldown,
         allow_shorts=True, sl_mult=1.5, tp_mult=2.5, allow_tp=True):
    """Run one simulation window. Returns (final_usdt, trades, equity_curve).

    sl_mult / tp_mult: ATR multipliers for stop-loss / take-profit.
    allow_tp: if False, longs only exit on stop-loss or model SELL signal (lets winners run).
    """
    usdt = STARTING_USDT
    btc = 0.0
    long_entry_price = None
    long_entry_atr   = None
    short = None
    trades = []
    equity_curve = []
    last_trade_candle = -cooldown - 1  # allow trade on candle 0

    def portfolio_value(price):
        v = usdt + btc * price
        if short:
            pnl = (short["entry_price"] - price) * short["btc"]
            v += short["collateral"] + pnl
        return v

    for candle_idx, (score, row) in enumerate(zip(all_scores, test_df.itertuples())):
        price  = float(row.close)
        atr    = float(row.atr)
        ema200 = float(row.ema200)
        rsi    = float(row.rsi)
        vwap   = float(row.vwap)
        in_bull = price > ema200

        equity_curve.append(portfolio_value(price))

        # --- Stop loss / take profit ---
        if btc > 0 and long_entry_price:
            sl = (long_entry_price - sl_mult * long_entry_atr) if long_entry_atr else long_entry_price * (1 - 0.013 * sl_mult)
            tp = (long_entry_price + tp_mult * long_entry_atr) if long_entry_atr else long_entry_price * (1 + 0.012 * tp_mult)
            hit_sl = price <= sl
            hit_tp = allow_tp and price >= tp
            if hit_sl or hit_tp:
                t_type = "STOP_LOSS LONG" if hit_sl else "TAKE_PROFIT LONG"
                gross = btc * price
                fee   = gross * FEE_RATE
                pnl   = gross - fee - (btc * long_entry_price)
                usdt += gross - fee
                trades.append({"type": t_type, "price": price, "pnl": pnl, "time": str(row.time)})
                btc = 0.0; long_entry_price = None; long_entry_atr = None

        if short:
            sl = (short["entry_price"] + sl_mult * short["entry_atr"]) if short.get("entry_atr") else short["entry_price"] * (1 + 0.013 * sl_mult)
            tp = (short["entry_price"] - tp_mult * short["entry_atr"]) if short.get("entry_atr") else short["entry_price"] * (1 - 0.012 * tp_mult)
            if price >= sl or price <= tp:
                t_type = "STOP_LOSS SHORT" if price >= sl else "TAKE_PROFIT SHORT"
                short_pnl = (short["entry_price"] - price) * short["btc"]
                fee       = abs(short["btc"] * price) * FEE_RATE
                returned  = max(short["collateral"] + short_pnl - fee, 0)
                usdt += returned
                trades.append({"type": t_type, "price": price, "pnl": short_pnl, "time": str(row.time)})
                short = None

        # --- Signal ---
        direction, confidence = _ensemble_vote(score, T=TEMPERATURE)

        if direction == "UP" and confidence >= buy_conf:
            signal = "BUY"
        elif direction == "DOWN" and confidence >= sell_conf:
            signal = "SELL"
        else:
            signal = "HOLD"

        allow_new_short = (not in_bull) and allow_shorts

        # Close short on BUY (no cooldown needed for closes)
        if signal == "BUY" and short:
            short_pnl = (short["entry_price"] - price) * short["btc"]
            fee       = abs(short["btc"] * price) * FEE_RATE
            returned  = max(short["collateral"] + short_pnl - fee, 0)
            usdt += returned
            trades.append({"type": "CLOSE SHORT", "price": price, "pnl": short_pnl, "time": str(row.time)})
            short = None

        in_cooldown = (candle_idx - last_trade_candle) < cooldown

        # Buy long
        if signal == "BUY" and in_bull and usdt > 1.0 and not in_cooldown:
            # Mean-reversion dip entry OR trend-following entry when model is confident
            dip_entry   = rsi < rsi_threshold and price <= vwap
            trend_entry = confidence >= 75
            good_price = dip_entry or trend_entry
            if good_price:
                # Size up slightly on trend-following entries to capture uptrend moves
                if confidence >= 80:
                    buy_pct = 0.40 if trend_entry and not dip_entry else 0.30
                elif confidence >= 70:
                    buy_pct = 0.25 if trend_entry and not dip_entry else 0.20
                else:
                    buy_pct = 0.10
                spend = usdt * buy_pct
                fee   = spend * FEE_RATE
                bought = (spend - fee) / price
                usdt  -= spend
                if btc > 0 and long_entry_price:
                    long_entry_price = (btc * long_entry_price + bought * price) / (btc + bought)
                else:
                    long_entry_price = price
                long_entry_atr = atr
                btc += bought
                trades.append({"type": "BUY", "price": price, "pnl": 0, "time": str(row.time)})
                last_trade_candle = candle_idx

        # Close long on SELL
        if signal == "SELL" and btc > 0:
            gross = btc * price
            fee   = gross * FEE_RATE
            pnl   = gross - fee - (btc * long_entry_price)
            usdt += gross - fee
            trades.append({"type": "SELL", "price": price, "pnl": pnl, "time": str(row.time)})
            btc = 0.0; long_entry_price = None; long_entry_atr = None

        # Open short
        if signal == "SELL" and allow_new_short and not short and usdt > 1.0 and not in_cooldown:
            collateral = usdt * 0.10
            fee        = collateral * FEE_RATE
            net_col    = collateral - fee
            short_btc  = net_col / price
            usdt -= collateral
            short = {"entry_price": price, "btc": short_btc, "collateral": net_col, "entry_atr": atr}
            trades.append({"type": "OPEN SHORT", "price": price, "pnl": 0, "time": str(row.time)})
            last_trade_candle = candle_idx

    # Close open positions at end
    price = float(test_df["close"].iloc[-1])
    if btc > 0 and long_entry_price:
        gross = btc * price
        fee   = gross * FEE_RATE
        pnl   = gross - fee - (btc * long_entry_price)
        usdt += gross - fee
        trades.append({"type": "CLOSE LONG (END)", "price": price, "pnl": pnl, "time": "end"})
        btc = 0.0
    if short:
        short_pnl = (short["entry_price"] - price) * short["btc"]
        fee       = abs(short["btc"] * price) * FEE_RATE
        returned  = max(short["collateral"] + short_pnl - fee, 0)
        usdt += returned
        trades.append({"type": "CLOSE SHORT (END)", "price": price, "pnl": short_pnl, "time": "end"})

    return usdt, trades, equity_curve


def _stats(final_usdt, trades, equity_curve):
    entry_types = {"BUY", "OPEN SHORT"}
    closed = [t for t in trades if t["type"] not in entry_types]
    wins   = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    win_rate     = len(wins) / len(closed) * 100 if closed else 0
    avg_win      = np.mean([t["pnl"] for t in wins])   if wins   else 0
    avg_loss     = np.mean([t["pnl"] for t in losses]) if losses else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_return = (final_usdt - STARTING_USDT) / STARTING_USDT * 100

    peak = STARTING_USDT; max_dd = 0.0
    for v in equity_curve:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd: max_dd = dd

    candles_per_day = 1440 / CANDLE_MINUTES
    daily_values  = equity_curve[::int(candles_per_day)] if len(equity_curve) > candles_per_day else equity_curve
    daily_returns = np.diff(daily_values) / np.array(daily_values[:-1]) if len(daily_values) > 1 else np.array([0])
    sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)) if np.std(daily_returns) > 0 else 0

    type_counts = {}
    for t in trades:
        type_counts[t["type"]] = type_counts.get(t["type"], 0) + 1

    return {
        "final": final_usdt, "total_return": total_return,
        "max_dd": max_dd, "sharpe": sharpe,
        "closed": len(closed), "win_rate": win_rate,
        "wins": len(wins), "losses": len(losses),
        "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "type_counts": type_counts,
    }


def run_backtest():
    print(f"=== Backtester: {MODE} ===")
    print(f"    Temperature: {TEMPERATURE}  |  Cooldown: {COOLDOWN_CANDLES} candles  |  SELL_CONF: {SELL_CONF}%\n")

    print("Loading historical data...")
    df = get_data()
    df_prep = prepare_dataset(df, horizon=HORIZON, threshold=THRESHOLD)
    n = len(df_prep)
    cutoff = int(n * 0.8)
    test_size = n - cutoff
    hours = test_size * CANDLE_MINUTES / 60
    print(f"Total rows: {n} | Train: {cutoff} | Test: {test_size} ({hours:.0f}h = {hours/24:.0f} days)")

    print(f"\nBuilding validation dataset (encoder={ENCODER_LENGTH})...")
    validation = TimeSeriesDataSet(
        df_prep[df_prep.time_idx >= cutoff - ENCODER_LENGTH],
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=1,
        time_varying_unknown_reals=FEATURES,
        target_normalizer=None,
        min_prediction_idx=cutoff,
    )
    val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

    scores_by_model = {}
    for m_cfg in ENSEMBLE_MODELS:
        label, path = m_cfg["label"], m_cfg["path"]
        try:
            print(f"Loading {label} from {path}...")
            model = TemporalFusionTransformer.load_from_checkpoint(path)
            model.eval()
            scores = []
            with torch.no_grad():
                for batch in val_loader:
                    x, (y, weight) = batch
                    raw = model(x)
                    # raw["prediction"] is logits from CrossEntropy loss — apply softmax to get P(UP)
                    probs = torch.softmax(raw["prediction"][:, 0, :], dim=-1)
                    scores.extend(probs[:, 1].cpu().numpy().tolist())
            scores_by_model[label] = scores
            print(f"  {label}: {len(scores)} predictions")
        except Exception as e:
            print(f"  {label}: failed to load ({e}), skipping.")

    if not scores_by_model:
        print("No models loaded — aborting.")
        return

    n_preds = min(len(s) for s in scores_by_model.values())
    for label in scores_by_model:
        scores_by_model[label] = scores_by_model[label][:n_preds]

    all_scores = []
    for i in range(n_preds):
        all_scores.append([scores_by_model[lbl][i] for lbl in scores_by_model])

    test_df = df_prep.iloc[cutoff:].reset_index(drop=True)
    n_preds = min(len(all_scores), len(test_df))
    all_scores = all_scores[:n_preds]
    test_df    = test_df.iloc[:n_preds]

    # --- Score distribution (sanity check) ---
    flat_scores = [s for bundle in all_scores for s in bundle]
    scaled_scores = [_temp_scale(s, TEMPERATURE) for s in flat_scores]
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    raw_hist   = [sum(1 for s in flat_scores   if bins[i] <= s < bins[i+1]) for i in range(len(bins)-1)]
    scaled_hist= [sum(1 for s in scaled_scores if bins[i] <= s < bins[i+1]) for i in range(len(bins)-1)]
    print(f"\nScore distribution (before / after T={TEMPERATURE}):")
    for i, (lo, hi) in enumerate(zip(bins, bins[1:])):
        print(f"  {lo:.1f}-{hi:.1f}: {raw_hist[i]:5d} raw  →  {scaled_hist[i]:5d} scaled")

    # --- Buy-and-hold benchmark ---
    hodl_start = float(test_df["close"].iloc[0])
    hodl_end   = float(test_df["close"].iloc[-1])
    hodl_return = (hodl_end - hodl_start) / hodl_start * 100
    hodl_fee    = 2 * FEE_RATE * 100  # buy in + sell out
    hodl_net    = hodl_return - hodl_fee

    # --- Full simulation ---
    print(f"\nRunning full backtest simulation ({n_preds} candles)...")
    final_usdt, trades, equity_curve = _sim(
        all_scores, test_df,
        buy_conf=BUY_CONF, sell_conf=SELL_CONF,
        rsi_threshold=RSI_THRESHOLD, cooldown=COOLDOWN_CANDLES,
        allow_shorts=ALLOW_SHORTS,
        sl_mult=SL_MULT, tp_mult=TP_MULT, allow_tp=ALLOW_TP,
    )
    s = _stats(final_usdt, trades, equity_curve)

    # --- 3-window breakdown ---
    w = n_preds // 3
    windows = [
        (0,       w,       "Window 1 (oldest)"),
        (w,       2*w,     "Window 2 (middle)"),
        (2*w,     n_preds, "Window 3 (recent)"),
    ]
    window_results = []
    for start, end, wlabel in windows:
        wdf     = test_df.iloc[start:end].reset_index(drop=True)
        wscores = all_scores[start:end]
        if len(wdf) == 0:
            continue
        wfinal, wtrades, wequity = _sim(
            wscores, wdf,
            buy_conf=BUY_CONF, sell_conf=SELL_CONF,
            rsi_threshold=RSI_THRESHOLD, cooldown=COOLDOWN_CANDLES,
            allow_shorts=ALLOW_SHORTS,
            sl_mult=SL_MULT, tp_mult=TP_MULT, allow_tp=ALLOW_TP,
        )
        t_start = str(wdf["time"].iloc[0])[:10]
        t_end   = str(wdf["time"].iloc[-1])[:10]
        hodl_w  = (float(wdf["close"].iloc[-1]) - float(wdf["close"].iloc[0])) / float(wdf["close"].iloc[0]) * 100
        bot_w   = (wfinal - STARTING_USDT) / STARTING_USDT * 100
        n_trades= len([t for t in wtrades if t["type"] not in {"BUY", "OPEN SHORT"}])
        window_results.append((wlabel, t_start, t_end, bot_w, hodl_w, n_trades))

    # --- Print ---
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  BACKTEST RESULTS — {MODE}")
    print(sep)
    t_start = str(test_df["time"].iloc[0])[:16]
    t_end   = str(test_df["time"].iloc[-1])[:16]
    print(f"  Period:          {t_start} → {t_end}")
    print(f"  Days tested:     {hours/24:.0f}")
    print(f"  Temperature:     {TEMPERATURE}  |  Cooldown: {COOLDOWN_CANDLES} candles")
    print()
    print(f"  Starting:        ${STARTING_USDT:.2f}")
    print(f"  Final (bot):     ${s['final']:.2f}")
    print(f"  Total return:    {s['total_return']:+.2f}%")
    print(f"  Buy-and-hold:    {hodl_net:+.2f}%  (raw {hodl_return:+.2f}% - fees)")
    print(f"  vs HODL:         {s['total_return'] - hodl_net:+.2f}%  {'✓ BEATS HODL' if s['total_return'] > hodl_net else '✗ lags HODL'}")
    print(f"  Max drawdown:    -{s['max_dd']:.2f}%")
    print(f"  Sharpe ratio:    {s['sharpe']:.2f}")
    print()
    print(f"  Closed trades:   {s['closed']}")
    print(f"  Win rate:        {s['win_rate']:.1f}%  ({s['wins']}W / {s['losses']}L)")
    print(f"  Avg win:         ${s['avg_win']:+.4f}")
    print(f"  Avg loss:        ${s['avg_loss']:+.4f}")
    print(f"  Profit factor:   {s['profit_factor']:.2f}")
    print()
    print("  Trade breakdown:")
    for t_type, count in sorted(s["type_counts"].items()):
        print(f"    {t_type:32s}: {count}")

    print()
    print("  3-Window Regime Breakdown:")
    print(f"  {'Window':20s} {'Period':23s} {'Bot':>8s}  {'HODL':>8s}  {'Trades':>7s}  {'vs HODL':>8s}")
    print(f"  {'-'*20} {'-'*23} {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}")
    for wlabel, ts, te, bot_w, hodl_w, n_t in window_results:
        diff = bot_w - hodl_w
        flag = "✓" if diff > 0 else "✗"
        print(f"  {wlabel:20s} {ts}→{te}  {bot_w:>+7.2f}%  {hodl_w:>+7.2f}%  {n_t:>7d}  {diff:>+7.2f}% {flag}")

    print(sep)

    print("\n  Last 10 closed trades:")
    entry_types = {"BUY", "OPEN SHORT"}
    closed = [t for t in trades if t["type"] not in entry_types]
    print(f"  {'Time':18s} {'Type':28s} {'Price':>10s} {'P&L':>10s}")
    print(f"  {'-'*18} {'-'*28} {'-'*10} {'-'*10}")
    for t in closed[-10:]:
        print(f"  {t['time']:18s} {t['type']:28s} ${t['price']:>9,.0f} {t['pnl']:>+10.4f}")


if __name__ == "__main__":
    run_backtest()
