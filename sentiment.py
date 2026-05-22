import feedparser
import ollama
import json

FEEDS = [
    "https://feeds.feedburner.com/CoinDesk",
    "https://cointelegraph.com/rss",
]

def fetch_headlines():
    headlines = []
    for url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            headlines.append(entry.title)
    return headlines[:10]

def get_sentiment():
    try:
        headlines = fetch_headlines()
        if not headlines:
            return {"sentiment": "NEUTRAL", "score": 50}
        print("  Latest headlines analyzed by llama3.1:")
        for h in headlines:
            print(f"    • {h}")
        headlines_text = "\n".join(f"- {h}" for h in headlines)
        prompt = (
            "You are a crypto analyst. Given these BTC news headlines, "
            "respond with ONLY a JSON object like this: "
            '{\"sentiment\": \"BULLISH\", \"score\": 72}. '
            "sentiment must be BULLISH, BEARISH, or NEUTRAL. "
            "score is 0-100. No explanation, no markdown, just the JSON. "
            f"Headlines:\n{headlines_text}"
        )
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"].strip()
        content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        sentiment = result.get("sentiment", "NEUTRAL").upper()
        score = int(result.get("score", 50))
        if sentiment not in ("BULLISH", "BEARISH", "NEUTRAL"):
            sentiment = "NEUTRAL"
        score = max(0, min(100, score))
        return {"sentiment": sentiment, "score": score}
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"sentiment": "NEUTRAL", "score": 50}
