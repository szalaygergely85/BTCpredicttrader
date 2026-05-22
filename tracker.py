import json
import os
from datetime import datetime

TRACKER_PATH = "predictions.json"
HORIZON_TICKS = 8        # 8 x 15min = 2 hours
MIN_SAMPLES = 5          # don't show stats until we have enough data

def load():
    if not os.path.exists(TRACKER_PATH):
        return []
    try:
        with open(TRACKER_PATH) as f:
            return json.load(f)
    except Exception:
        print("  [tracker] predictions.json corrupted, resetting.")
        return []

def save(data):
    tmp = TRACKER_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, TRACKER_PATH)

def log_prediction(direction, confidence, price):
    data = load()
    data.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "direction": direction,
        "confidence": confidence,
        "price_at_prediction": price,
        "resolved": False,
        "correct": None,
        "price_at_resolution": None,
        "tick": len(data),
    })
    save(data)

def resolve_predictions(current_price):
    data = load()
    current_tick = max((d["tick"] for d in data), default=0)
    changed = False

    for entry in data:
        if entry["resolved"]:
            continue
        if current_tick - entry["tick"] >= HORIZON_TICKS:
            actual_up = current_price > entry["price_at_prediction"] * 1.005
            predicted_up = entry["direction"] == "UP"
            entry["correct"] = (actual_up == predicted_up)
            entry["price_at_resolution"] = round(current_price, 2)
            entry["resolved"] = True
            changed = True

    if changed:
        save(data)
    return data

def get_stats(current_price):
    data = resolve_predictions(current_price)
    resolved = [d for d in data if d["resolved"]]

    if len(resolved) < MIN_SAMPLES:
        remaining = MIN_SAMPLES - len(resolved)
        return {"status": "warming_up", "remaining": remaining}

    total = len(resolved)
    correct = sum(1 for d in resolved if d["correct"])
    accuracy = correct / total * 100
    avg_conf = sum(d["confidence"] for d in resolved) / total

    # break down by confidence bucket
    buckets = {"50-65": [], "65-80": [], "80+": []}
    for d in resolved:
        c = d["confidence"]
        if c >= 80:
            buckets["80+"].append(d["correct"])
        elif c >= 65:
            buckets["65-80"].append(d["correct"])
        else:
            buckets["50-65"].append(d["correct"])

    def bucket_str(items):
        if not items:
            return "n/a"
        return f"{sum(items)/len(items)*100:.0f}% ({len(items)})"

    return {
        "status": "ok",
        "total": total,
        "accuracy": round(accuracy, 1),
        "avg_confidence": round(avg_conf, 1),
        "by_confidence": {
            "50-65%": bucket_str(buckets["50-65"]),
            "65-80%": bucket_str(buckets["65-80"]),
            "80%+":   bucket_str(buckets["80+"]),
        },
        "last_resolved": resolved[-1]["time"] if resolved else None,
    }

def print_stats(current_price):
    stats = get_stats(current_price)
    if stats["status"] == "warming_up":
        print(f"  Accuracy: warming up ({stats['remaining']} more predictions needed)")
        return
    b = stats["by_confidence"]
    print(
        f"  Accuracy: {stats['accuracy']}% correct over {stats['total']} predictions"
        f" | Avg confidence: {stats['avg_confidence']}%"
    )
    print(
        f"  By confidence — "
        f"50-65%: {b['50-65%']}  "
        f"65-80%: {b['65-80%']}  "
        f"80%+: {b['80%+']}"
    )
