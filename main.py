import schedule
import time
from datetime import datetime
from data import get_live_data
from model import load_or_train, predict
from sentiment import get_sentiment
from signals import get_signal
import trader
import tracker

def run():
    try:
        df = get_live_data()
        current_price = df["close"].iloc[-1]
        tft_result = predict(df)
        sentiment_result = get_sentiment()
        signal = get_signal(tft_result, sentiment_result)
        now = datetime.now().strftime("%H:%M")

        print(
            f"[{now}] BTC ${current_price:,.2f} | "
            f"TFT: {tft_result['direction']} {tft_result['confidence']}% | "
            f"Sentiment: {sentiment_result['sentiment']} {sentiment_result['score']} | "
            f"→ {signal}"
        )

        # log prediction and print accuracy stats
        tracker.log_prediction(tft_result["direction"], tft_result["confidence"], current_price)
        tracker.print_stats(current_price)

        trade, portfolio = trader.execute(signal, current_price)

        if trade:
            print(
                f"  {'💰 BOUGHT' if trade['action'] == 'BUY' else '💸 SOLD  '} "
                f"{trade['btc']:.6f} BTC for ${trade['usdt']:,.2f} "
                f"| Portfolio: ${trade['portfolio_value']:,.2f} "
                f"| P&L: {'+'if trade['pnl'] >= 0 else ''}{trade['pnl']:,.2f} ({'+' if trade['pnl'] >= 0 else ''}{(trade['pnl'] / 20 * 100):.2f}%)"
            )
        else:
            s = trader.status(current_price)
            print(
                f"  [HOLD] Portfolio: ${s['total_value']:,.2f} "
                f"(USDT: ${s['usdt']:,.2f} | BTC: {s['btc']:.6f}) "
                f"| P&L: {'+'if s['pnl'] >= 0 else ''}{s['pnl']:,.2f} ({'+' if s['pnl_pct'] >= 0 else ''}{s['pnl_pct']:.2f}%)"
            )

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    print("Starting BTC Prediction + Paper Trading (demo)")
    print(f"Starting balance: $20.00 USDT")
    print("-" * 60)
    load_or_train()
    run()
    schedule.every(15).minutes.do(run)
    while True:
        schedule.run_pending()
        time.sleep(1)
