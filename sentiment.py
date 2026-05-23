import feedparser
import ollama
import json

FEEDS = [
    ("CoinDesk",        "https://feeds.feedburner.com/CoinDesk"),
    ("CoinTelegraph",   "https://cointelegraph.com/rss"),
    ("Reddit/Bitcoin",  "https://www.reddit.com/r/Bitcoin/.rss"),
    ("Reddit/Crypto",   "https://www.reddit.com/r/CryptoCurrency/.rss"),
    ("BitcoinMagazine", "https://bitcoinmagazine.com/.rss"),
    ("Decrypt",         "https://decrypt.co/feed"),
]

def fetch_headlines():
    headlines = []
    for source, url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                headlines.append((source, entry.title))
        except Exception as e:
            print(f"  Feed error ({source}): {e}")
    return headlines[:18]

def get_sentiment():
    try:
        headline_pairs = fetch_headlines()
        if not headline_pairs:
            return {"sentiment": "NEUTRAL", "score": 50}
        print("  Latest headlines analyzed by llama3.1:")
        for source, title in headline_pairs:
            print(f"    [{source}] {title}")
        headlines_text = "\n".join(f"- {title}" for _, title in headline_pairs)
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
