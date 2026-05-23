import schedule
import time
from datetime import datetime
from data import get_live_data, get_order_book_imbalance
from ensemble import load_or_train_all, predict_ensemble
from sentiment import get_sentiment
from signals import get_signal
import trader
import tracker

def run():
    try:
        df = get_live_data()
        current_price = df["close"].iloc[-1]

        # --- Stop loss / take profit check (BEFORE signal logic) ---
        stop_type, position_type = trader.check_stops(current_price)
        if stop_type is not None:
            stop_trade, _ = trader.execute_stop(stop_type, position_type, current_price)
            if stop_trade:
                entry_price = None
                # Parse entry from note e.g. "entry $77,000.00, -2.33%"
                note = stop_trade.get("note", "")
                pct_str = ""
                for part in note.split(","):
                    part = part.strip()
                    if part.startswith("entry"):
                        try:
                            entry_price = float(part.replace("entry $", "").replace(",", ""))
                        except Exception:
                            pass
                    if part.endswith("%"):
                        pct_str = part

                if stop_type == "STOP_LOSS":
                    icon = "STOP LOSS"
                    arrow = "STOP_LOSS"
                else:
                    icon = "TAKE PROFIT"
                    arrow = "TAKE_PROFIT"

                entry_str = f"${entry_price:,.2f}" if entry_price else "unknown"
                if stop_type == "STOP_LOSS":
                    print(
                        f"  STOP LOSS triggered — closed {position_type} at "
                        f"${current_price:,.2f} (entry {entry_str}, {pct_str})"
                    )
                else:
                    print(
                        f"  TAKE PROFIT triggered — closed {position_type} at "
                        f"${current_price:,.2f} (entry {entry_str}, {pct_str})"
                    )

        # --- Normal signal logic ---
        tft_result = predict_ensemble(df)
        sentiment_result = get_sentiment()
        imbalance = get_order_book_imbalance()
        signal = get_signal(tft_result, sentiment_result, order_book_imbalance=imbalance)
        now = datetime.now().strftime("%H:%M")

        # Show ensemble votes
        votes_str = ""
        if tft_result.get("votes"):
            votes_str = " | Votes: " + " / ".join(
                f"{k.split()[-1]}:{v}" for k, v in tft_result["votes"].items()
            )

        print(
            f"[{now}] BTC ${current_price:,.2f} | "
            f"TFT: {tft_result['direction']} {tft_result['confidence']}%{votes_str} | "
            f"Sentiment: {sentiment_result['sentiment']} {sentiment_result['score']} | "
            f"OBI: {imbalance:+.3f} | "
            f"-> {signal}"
        )

        # log prediction and print accuracy stats
        tracker.log_prediction(tft_result["direction"], tft_result["confidence"], current_price)
        tracker.print_stats(current_price)

        trade, portfolio = trader.execute(signal, current_price)

        if trade:
            icons = {
                "BUY": "LONG  ",
                "SELL": "SELL  ",
                "OPEN SHORT": "SHORT ",
                "CLOSE SHORT": "COVER ",
                "STOP_LOSS LONG": "STOP LOSS LONG   ",
                "TAKE_PROFIT LONG": "TAKE PROFIT LONG ",
                "STOP_LOSS SHORT": "STOP LOSS SHORT  ",
                "TAKE_PROFIT SHORT": "TAKE PROFIT SHORT",
            }
            icon = icons.get(trade["action"], "     ")
            note = f" ({trade['note']})" if trade.get("note") else ""
            print(
                f"  {icon} {trade['btc']:.6f} BTC @ ${trade['price']:,.2f}"
                f" | ${trade['usdt']:,.4f}{note}"
                f" | Portfolio: ${trade['portfolio_value']:,.2f}"
                f" | P&L: {'+' if trade['pnl'] >= 0 else ''}{trade['pnl']:,.4f}"
            )

        s = trader.status(current_price)
        btc_value = s['btc'] * current_price
        lines = [
            f"  Portfolio: ${s['total_value']:,.4f}"
            f" | P&L: {'+' if s['pnl'] >= 0 else ''}{s['pnl']:,.4f} ({'+' if s['pnl_pct'] >= 0 else ''}{s['pnl_pct']:.2f}%)"
        ]
        lines.append(
            f"  USDT: ${s['usdt']:,.4f}"
            f" | BTC: {s['btc']:.6f} (worth ${btc_value:,.4f})"
        )
        if s.get("long_entry_price") and s["btc"] > 0:
            entry = s["long_entry_price"]
            pct = (current_price - entry) / entry * 100
            lines.append(
                f"  LONG entry: ${entry:,.2f}"
                f" | Current: ${current_price:,.2f}"
                f" | uPnL: {'+' if pct >= 0 else ''}{pct:.2f}%"
                f" | SL: ${entry*0.98:,.2f}  TP: ${entry*1.03:,.2f}"
            )
        if s["short"]:
            sh = s["short"]
            lines.append(
                f"  SHORT: {sh['btc']:.6f} BTC @ ${sh['entry_price']:,.2f}"
                f" | Now worth ${sh['btc'] * current_price:,.4f}"
                f" | uPnL: {'+' if sh['pnl'] >= 0 else ''}{sh['pnl']:,.4f}"
            )
        for line in lines:
            print(line)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting BTC Prediction + Paper Trading (Ensemble Edition)")
    print(f"Starting balance: $2.00 USDT")
    print("-" * 60)
    load_or_train_all()
    run()
    schedule.every(15).minutes.do(run)
    while True:
        schedule.run_pending()
        time.sleep(1)
