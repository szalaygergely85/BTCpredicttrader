def get_signal(tft_result, sentiment_result, order_book_imbalance=0.0):
    """
    Compute trading signal from TFT prediction, sentiment, and order book imbalance.

    Order book imbalance filter:
      - imbalance > +0.3 and signal is SELL  → downgrade to HOLD (buyers dominating)
      - imbalance < -0.3 and signal is BUY   → downgrade to HOLD (sellers dominating)
    """
    direction = tft_result["direction"]
    confidence = tft_result["confidence"]
    sentiment = sentiment_result["sentiment"]

    if direction == "UP" and confidence >= 65 and sentiment == "BULLISH":
        signal = "STRONG BUY"
    elif direction == "UP" and confidence >= 55 and sentiment != "BEARISH":
        signal = "BUY"
    elif direction == "DOWN" and confidence >= 65 and sentiment == "BEARISH":
        signal = "STRONG SELL"
    elif direction == "DOWN" and confidence >= 55 and sentiment != "BULLISH":
        signal = "SELL"
    else:
        signal = "HOLD"

    # Order book imbalance filter
    if order_book_imbalance > 0.3 and signal in ("SELL", "STRONG SELL"):
        print(f"  Order book imbalance +{order_book_imbalance:.3f} (buyers dominating) → downgrading {signal} to HOLD")
        signal = "HOLD"
    elif order_book_imbalance < -0.3 and signal in ("BUY", "STRONG BUY"):
        print(f"  Order book imbalance {order_book_imbalance:.3f} (sellers dominating) → downgrading {signal} to HOLD")
        signal = "HOLD"

    return signal
