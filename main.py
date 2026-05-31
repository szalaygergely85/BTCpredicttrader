import argparse
import schedule
import time
from datetime import datetime

parser = argparse.ArgumentParser(description="BTC/Stock Prediction + Paper Trading")
parser.add_argument("-scalping", action="store_true", help="BTC scalp mode (5-min candles, 2-min ticks)")
parser.add_argument("-shares",   type=str,  default=None, metavar="TICKER",
                    help="Stock ticker to trade, e.g. -shares NVDA")
args = parser.parse_args()

SCALPING = args.scalping
TICKER   = args.shares.upper() if args.shares else None
SHARES   = TICKER is not None

# Mode is mutually exclusive
if SCALPING and SHARES:
    raise SystemExit("Cannot combine -scalping and -shares. Pick one.")

if SHARES:
    from data_shares import get_shares_live_data, is_market_open, minutes_to_open
    import shares as model_manager
    import trader
    import tracker
    trader.PORTFOLIO_PATH   = f"portfolio_{TICKER.lower()}.json"
    trader.COOLDOWN_MINUTES = 60   # min 60 min between new entries for stocks
    tracker.TRACKER_PATH    = f"predictions_{TICKER.lower()}.json"

elif SCALPING:
    from data import get_scalp_live_data as get_live_data, get_order_book_imbalance
    import scalp as model_manager
    import trader
    import tracker
    trader.PORTFOLIO_PATH   = "portfolio_scalp.json"
    trader.COOLDOWN_MINUTES = 10   # scalp: min 10 min between new entries
    tracker.TRACKER_PATH    = "predictions_scalp.json"

else:
    from data import get_live_data, get_order_book_imbalance
    from ensemble import load_or_train_all, predict_ensemble
    import trader
    import tracker
    trader.COOLDOWN_MINUTES = 60   # swing: min 60 min between new entries

from signals import get_signal

if not SHARES:
    from sentiment import get_sentiment


def _parse_stop_note(note):
    entry_price, pct_str = None, ""
    for part in note.split(","):
        part = part.strip()
        if part.startswith("entry"):
            try:
                entry_price = float(part.replace("entry $", "").replace(",", ""))
            except Exception:
                pass
        if part.endswith("%"):
            pct_str = part
    return entry_price, pct_str


def run_shares():
    """Single tick for stock trading mode."""
    if not is_market_open():
        mins = minutes_to_open()
        now  = datetime.now().strftime("%H:%M")
        print(f"[{TICKER}] [{now}] Market closed — {mins} min to open. Waiting...")
        return

    try:
        df = get_shares_live_data(TICKER)
        current_price = float(df["close"].iloc[-1])
        ema200 = float(df["ema200"].iloc[-1])
        atr    = float(df["atr"].iloc[-1])
        in_bull_regime = current_price > ema200

        # --- Stop loss / take profit ---
        stop_type, position_type = trader.check_stops(current_price)
        if stop_type is not None:
            stop_trade, _ = trader.execute_stop(stop_type, position_type, current_price)
            if stop_trade:
                entry_price, pct_str = _parse_stop_note(stop_trade.get("note", ""))
                entry_str = f"${entry_price:,.2f}" if entry_price else "unknown"
                label = "STOP LOSS" if stop_type == "STOP_LOSS" else "TAKE PROFIT"
                print(f"  {label} triggered — closed {position_type} at ${current_price:,.2f} (entry {entry_str}, {pct_str})")

        # --- Prediction ---
        tft_result = model_manager.predict(TICKER, df)
        votes_str  = ""
        if tft_result.get("votes"):
            votes_str = " | Votes: " + " / ".join(
                f"{k.split()[-1]}:{v}" for k, v in tft_result["votes"].items()
            )

        # Stocks: use basic signal without order book imbalance
        # Sentiment proxy: simplified — skip BTC-specific fear/greed weighting
        direction  = tft_result["direction"]
        confidence = tft_result["confidence"]
        if direction == "UP" and confidence >= 65:
            signal = "STRONG BUY" if confidence >= 75 else "BUY"
        elif direction == "DOWN" and confidence >= 90:
            signal = "STRONG SELL" if confidence >= 95 else "SELL"
        else:
            signal = "HOLD"

        now = datetime.now().strftime("%H:%M")
        regime_str = "BULL" if in_bull_regime else "BEAR"
        print(
            f"[{TICKER}] [{now}] ${current_price:,.2f} | "
            f"TFT: {direction} {confidence}%{votes_str} | "
            f"Regime: {regime_str} | ATR: ${atr:,.2f} | -> {signal}"
        )

        tracker.log_prediction(direction, confidence, current_price)
        tracker.print_stats(current_price)

        # --- Sizing: stocks = long only (no shorts, no margin) ---
        rsi           = float(df["rsi"].iloc[-1])
        price_vs_vwap = current_price <= float(df["vwap"].iloc[-1])
        buy_pct       = None

        if signal in ("BUY", "STRONG BUY"):
            good_price      = rsi < 55 and price_vs_vwap
            good_prediction = confidence >= 65

            if good_price and good_prediction and in_bull_regime:
                buy_pct = 0.30 if confidence >= 80 else (0.20 if confidence >= 70 else 0.10)
                print(f"  Good entry: RSI={rsi:.1f} | Conf={confidence}% → buying {int(buy_pct*100)}%")
            else:
                reasons = []
                if not in_bull_regime:    reasons.append("bear regime (price < EMA200)")
                if rsi >= 55:             reasons.append(f"RSI={rsi:.1f} overbought")
                if not price_vs_vwap:     reasons.append("price above VWAP")
                if confidence < 65:       reasons.append(f"confidence {confidence}% too low")
                print(f"  Skipping buy: {', '.join(reasons)}")

        # Pyramiding
        s_pre = trader.status(current_price)
        if s_pre["long_entry_price"] and s_pre["btc"] > 0 and signal in ("BUY", "STRONG BUY") and in_bull_regime:
            upnl_pct = (current_price - s_pre["long_entry_price"]) / s_pre["long_entry_price"] * 100
            if upnl_pct >= 1.5:
                if buy_pct is None:
                    buy_pct = 0.05
                    print(f"  Pyramiding: position up {upnl_pct:.1f}%, adding 5%")
                else:
                    buy_pct = min(buy_pct + 0.05, 0.35)
                    print(f"  Pyramiding: position up {upnl_pct:.1f}%, boosting to {int(buy_pct*100)}%")

        # Stocks: no shorts (simplest, avoids margin/PDT rules)
        trade, portfolio = trader.execute(
            signal, current_price,
            buy_pct=buy_pct,
            allow_new_short=False,
            atr=atr,
        )

        if trade:
            icons = {"BUY": "LONG  ", "SELL": "SELL  ",
                     "STOP_LOSS LONG": "STOP LOSS  ", "TAKE_PROFIT LONG": "TAKE PROFIT"}
            icon    = icons.get(trade["action"], "      ")
            note    = f" ({trade['note']})" if trade.get("note") else ""
            fee_str = f" fee=${trade.get('fee', 0):.4f}" if trade.get("fee") else ""
            print(
                f"  {icon} {trade['btc']:.4f} shares @ ${trade['price']:,.2f}"
                f" | ${trade['usdt']:,.4f}{note}{fee_str}"
                f" | Portfolio: ${trade['portfolio_value']:,.2f}"
                f" | P&L: {'+' if trade['pnl'] >= 0 else ''}{trade['pnl']:,.4f}"
            )

        s = trader.status(current_price)
        share_value = s["btc"] * current_price
        print(
            f"  Portfolio: ${s['total_value']:,.4f}"
            f" | P&L: {'+' if s['pnl'] >= 0 else ''}{s['pnl']:,.4f} ({'+' if s['pnl_pct'] >= 0 else ''}{s['pnl_pct']:.2f}%)"
        )
        print(f"  Cash: ${s['usdt']:,.4f} | Shares: {s['btc']:.4f} (worth ${share_value:,.4f})")
        if s.get("long_entry_price") and s["btc"] > 0:
            entry = s["long_entry_price"]
            pct   = (current_price - entry) / entry * 100
            entry_atr = s.get("long_entry_atr")
            sl = (entry - 1.5 * entry_atr) if entry_atr else entry * 0.98
            tp = (entry + 2.5 * entry_atr) if entry_atr else entry * 1.03
            print(f"  LONG entry: ${entry:,.2f} | uPnL: {'+' if pct >= 0 else ''}{pct:.2f}% | SL: ${sl:,.2f}  TP: ${tp:,.2f}")

    except Exception as e:
        print(f"[{TICKER}] [ERROR] {e}")
        import traceback
        traceback.print_exc()


def run():
    """Single tick for BTC modes (swing or scalp)."""
    try:
        df = get_live_data()
        current_price  = df["close"].iloc[-1]
        ema200         = float(df["ema200"].iloc[-1])
        atr            = float(df["atr"].iloc[-1])
        in_bull_regime = current_price > ema200

        # --- Stop loss / take profit ---
        stop_type, position_type = trader.check_stops(current_price)
        if stop_type is not None:
            stop_trade, _ = trader.execute_stop(stop_type, position_type, current_price)
            if stop_trade:
                entry_price, pct_str = _parse_stop_note(stop_trade.get("note", ""))
                entry_str = f"${entry_price:,.2f}" if entry_price else "unknown"
                label = "STOP LOSS" if stop_type == "STOP_LOSS" else "TAKE PROFIT"
                print(f"  {label} triggered — closed {position_type} at ${current_price:,.2f} (entry {entry_str}, {pct_str})")

        # --- Prediction ---
        if SCALPING:
            tft_result = model_manager.predict(df)
        else:
            tft_result = predict_ensemble(df)

        votes_str = ""
        if tft_result.get("votes"):
            votes_str = " | Votes: " + " / ".join(
                f"{k.split()[-1]}:{v}" for k, v in tft_result["votes"].items()
            )

        sentiment_result = get_sentiment()
        imbalance        = get_order_book_imbalance()
        signal           = get_signal(tft_result, sentiment_result, order_book_imbalance=imbalance)
        now              = datetime.now().strftime("%H:%M")
        regime_str       = "BULL" if in_bull_regime else "BEAR"
        mode_tag         = "[SCALP] " if SCALPING else ""

        print(
            f"{mode_tag}[{now}] BTC ${current_price:,.2f} | "
            f"TFT: {tft_result['direction']} {tft_result['confidence']}%{votes_str} | "
            f"Sentiment: {sentiment_result['sentiment']} {sentiment_result['score']} | "
            f"OBI: {imbalance:+.3f} | Regime: {regime_str} | -> {signal}"
        )

        tracker.log_prediction(tft_result["direction"], tft_result["confidence"], current_price)
        tracker.print_stats(current_price)

        rsi           = float(df["rsi"].iloc[-1])
        price_vs_vwap = float(df["close"].iloc[-1]) <= float(df["vwap"].iloc[-1])
        conf          = tft_result["confidence"]
        rsi_threshold = 50 if SCALPING else 55
        buy_conf_threshold  = 60
        sell_conf_threshold = 85

        buy_pct         = None
        allow_new_short = not in_bull_regime

        if signal in ("BUY", "STRONG BUY"):
            good_price      = rsi < rsi_threshold and price_vs_vwap
            good_prediction = conf >= buy_conf_threshold
            if good_price and good_prediction and in_bull_regime:
                buy_pct = 0.30 if conf >= 80 else (0.20 if conf >= 70 else 0.10)
                print(f"  Good entry: RSI={rsi:.1f} | Conf={conf}% → buying {int(buy_pct*100)}% | ATR=${atr:,.0f}")
            else:
                reasons = []
                if not in_bull_regime: reasons.append("bear regime (price < EMA200)")
                if rsi >= rsi_threshold: reasons.append(f"RSI={rsi:.1f} overbought")
                if not price_vs_vwap: reasons.append("price above VWAP")
                if conf < buy_conf_threshold: reasons.append(f"confidence {conf}% too low")
                print(f"  Skipping buy: {', '.join(reasons)}")

        s_pre = trader.status(current_price)
        if s_pre["long_entry_price"] and s_pre["btc"] > 0 and signal in ("BUY", "STRONG BUY") and in_bull_regime:
            entry    = s_pre["long_entry_price"]
            upnl_pct = (current_price - entry) / entry * 100
            if upnl_pct >= 1.5:
                if buy_pct is None:
                    buy_pct = 0.05
                    print(f"  Pyramiding: position up {upnl_pct:.1f}%, adding 5%")
                else:
                    buy_pct = min(buy_pct + 0.05, 0.35)
                    print(f"  Pyramiding: position up {upnl_pct:.1f}%, boosting to {int(buy_pct*100)}%")

        trade, portfolio = trader.execute(
            signal, current_price,
            buy_pct=buy_pct,
            allow_new_short=allow_new_short,
            atr=atr,
        )

        if trade:
            icons = {
                "BUY": "LONG  ", "SELL": "SELL  ",
                "OPEN SHORT": "SHORT ", "CLOSE SHORT": "COVER ",
                "STOP_LOSS LONG": "STOP LOSS LONG   ", "TAKE_PROFIT LONG": "TAKE PROFIT LONG ",
                "STOP_LOSS SHORT": "STOP LOSS SHORT  ", "TAKE_PROFIT SHORT": "TAKE PROFIT SHORT",
            }
            icon    = icons.get(trade["action"], "     ")
            note    = f" ({trade['note']})" if trade.get("note") else ""
            fee_str = f" fee=${trade.get('fee', 0):.4f}" if trade.get("fee") else ""
            print(
                f"  {icon} {trade['btc']:.6f} BTC @ ${trade['price']:,.2f}"
                f" | ${trade['usdt']:,.4f}{note}{fee_str}"
                f" | Portfolio: ${trade['portfolio_value']:,.2f}"
                f" | P&L: {'+' if trade['pnl'] >= 0 else ''}{trade['pnl']:,.4f}"
            )

        s = trader.status(current_price)
        btc_value = s["btc"] * current_price
        lines = [
            f"  Portfolio: ${s['total_value']:,.4f}"
            f" | P&L: {'+' if s['pnl'] >= 0 else ''}{s['pnl']:,.4f} ({'+' if s['pnl_pct'] >= 0 else ''}{s['pnl_pct']:.2f}%)"
        ]
        lines.append(f"  USDT: ${s['usdt']:,.4f} | BTC: {s['btc']:.6f} (worth ${btc_value:,.4f})")
        if s.get("long_entry_price") and s["btc"] > 0:
            entry     = s["long_entry_price"]
            entry_atr = s.get("long_entry_atr")
            pct       = (current_price - entry) / entry * 100
            sl = (entry - 1.5 * entry_atr) if entry_atr else entry * 0.98
            tp = (entry + 2.5 * entry_atr) if entry_atr else entry * 1.03
            lines.append(
                f"  LONG entry: ${entry:,.2f} | Current: ${current_price:,.2f}"
                f" | uPnL: {'+' if pct >= 0 else ''}{pct:.2f}% | SL: ${sl:,.2f}  TP: ${tp:,.2f}"
            )
        if s["short"]:
            sh = s["short"]
            lines.append(
                f"  SHORT: {sh['btc']:.6f} BTC @ ${sh['entry_price']:,.2f}"
                f" | uPnL: {'+' if sh['pnl'] >= 0 else ''}{sh['pnl']:,.4f}"
            )
        for line in lines:
            print(line)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if SHARES:
        print(f"Starting Stock Prediction + Paper Trading — {TICKER}")
        print(f"Starting balance: $100.00  |  Long-only  |  Market hours only")
        print("-" * 60)
        model_manager.load_or_train(TICKER)
        run_shares()
        schedule.every(30).minutes.do(run_shares)

    elif SCALPING:
        print("Starting BTC Prediction + Paper Trading (Scalping Mode — 5-min candles)")
        print("Starting balance: $100.00 USDT")
        print("-" * 60)
        model_manager.load_or_train()
        run()
        schedule.every(2).minutes.do(run)

    else:
        print("Starting BTC Prediction + Paper Trading (Ensemble Edition)")
        print("Starting balance: $100.00 USDT")
        print("-" * 60)
        load_or_train_all()
        run()
        schedule.every(15).minutes.do(run)

    while True:
        schedule.run_pending()
        time.sleep(1)
