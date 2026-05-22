def get_signal(tft_result, sentiment_result):
    direction = tft_result["direction"]
    confidence = tft_result["confidence"]
    sentiment = sentiment_result["sentiment"]

    if direction == "UP" and confidence >= 65 and sentiment == "BULLISH":
        return "STRONG BUY"
    elif direction == "UP" and confidence >= 55 and sentiment != "BEARISH":
        return "BUY"
    elif direction == "DOWN" and confidence >= 65 and sentiment == "BEARISH":
        return "STRONG SELL"
    elif direction == "DOWN" and confidence >= 55 and sentiment != "BULLISH":
        return "SELL"
    else:
        return "HOLD"
