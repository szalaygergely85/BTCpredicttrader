"""
dashboard.py — Web dashboard for BTC + Stock Prediction + Paper Trading
Usage: python dashboard.py  →  http://localhost:5000
"""
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
import requests
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

STARTING_USDT  = 100.0
CWD            = os.path.dirname(os.path.abspath(__file__))

_processes     = {}          # mode_key -> subprocess.Popen
_process_lock  = threading.Lock()

# ── Helpers ────────────────────────────────────────────────────

def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def _btc_price():
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=3)
        return float(r.json()["price"])
    except Exception:
        return None

def _stock_price(ticker):
    try:
        import yfinance as yf
        return float(yf.Ticker(ticker).fast_info["lastPrice"])
    except Exception:
        return None

def _is_market_open():
    try:
        import pytz
        et  = __import__("pytz").timezone("America/New_York")
        now = datetime.now(et)
        if now.weekday() >= 5:
            return False
        mo = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        mc = now.replace(hour=16, minute=0,  second=0, microsecond=0)
        return mo <= now < mc
    except Exception:
        return None

def _is_running(key):
    with _process_lock:
        proc = _processes.get(key)
        if proc is None:
            return False
        return proc.poll() is None

def _portfolio_stats(portfolio, price):
    usdt     = portfolio.get("usdt", STARTING_USDT)
    qty      = portfolio.get("btc", 0.0)           # shares use same "btc" field
    short    = portfolio.get("short")
    starting = portfolio.get("starting_usdt", STARTING_USDT)

    qty_value = qty * price if price else 0.0
    short_pnl = short_collateral = 0.0
    if short and price:
        short_pnl        = (short["entry_price"] - price) * short["btc"]
        short_collateral = short.get("collateral", 0.0)

    total   = usdt + qty_value + short_collateral + short_pnl
    pnl     = total - starting
    pnl_pct = pnl / starting * 100

    long_entry = portfolio.get("long_entry_price")
    long_upnl  = None
    if long_entry and qty > 0 and price:
        long_upnl = (price - long_entry) / long_entry * 100

    return {
        "usdt": round(usdt, 4), "qty": round(qty, 6),
        "qty_value": round(qty_value, 4),
        "total": round(total, 4),
        "pnl": round(pnl, 4), "pnl_pct": round(pnl_pct, 2),
        "long_entry": long_entry,
        "long_upnl": round(long_upnl, 2) if long_upnl is not None else None,
        "short": short, "short_pnl": round(short_pnl, 4) if short else None,
    }

def _accuracy_stats(predictions):
    resolved = [p for p in predictions if p.get("resolved") and p.get("correct") is not None]
    if not resolved:
        return {"total": 0, "correct": 0, "accuracy": 0.0}
    correct = sum(1 for p in resolved if p["correct"])
    return {"total": len(resolved), "correct": correct,
            "accuracy": round(correct / len(resolved) * 100, 1)}

def _equity_curve(portfolio):
    trades  = portfolio.get("trades", [])
    starting = portfolio.get("starting_usdt", STARTING_USDT)
    points  = [{"time": "Start", "value": starting}]
    for t in trades:
        pv = t.get("portfolio_value")
        if pv is not None:
            points.append({"time": t.get("time", ""), "value": round(pv, 2)})
    return points

def _get_share_tickers():
    """Discover tickers from portfolio_*.json files + running processes."""
    tickers = set()
    # From portfolio files
    for f in os.listdir(CWD):
        m = re.match(r"^portfolio_(.+)\.json$", f)
        if m:
            slug = m.group(1)
            if slug not in ("scalp",):      # exclude known non-stock portfolios
                tickers.add(slug.upper())
    # From running processes
    with _process_lock:
        for key in _processes:
            if key.startswith("shares_") and _processes[key] and _processes[key].poll() is None:
                tickers.add(key[7:].upper())
    return sorted(tickers)

def _start_process(key, cmd, log_path):
    with _process_lock:
        log_file = open(log_path, "a", buffering=1)
        log_file.write(f"\n\n{'='*60}\n  Started {key} — {datetime.now()}\n{'='*60}\n")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=CWD)
        _processes[key] = proc
    return proc

def _stop_process(key):
    with _process_lock:
        proc = _processes.get(key)
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        _processes[key] = None
    return True

# ── Routes ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    btc_price = _btc_price()

    def _mode_block(portfolio_path, predictions_path, key, price):
        portfolio   = _read_json(portfolio_path, {"usdt": STARTING_USDT, "btc": 0, "trades": []})
        predictions = _read_json(predictions_path, [])
        return {
            "running":         _is_running(key),
            "stats":           _portfolio_stats(portfolio, price or 0),
            "accuracy":        _accuracy_stats(predictions),
            "equity_curve":    _equity_curve(portfolio),
            "trades":          list(reversed(portfolio.get("trades", [])))[:20],
            "last_prediction": predictions[-1] if predictions else None,
        }

    # BTC modes
    swing = _mode_block("portfolio.json",       "predictions.json",       "swing", btc_price)
    scalp = _mode_block("portfolio_scalp.json", "predictions_scalp.json", "scalp", btc_price)

    # Stock tickers
    shares_data = {}
    for ticker in _get_share_tickers():
        t    = ticker.lower()
        price = _stock_price(ticker) if _is_market_open() else None
        # Fall back to last known price from portfolio
        if price is None:
            port = _read_json(f"portfolio_{t}.json", {})
            trades = port.get("trades", [])
            price = float(trades[-1]["price"]) if trades else 0.0
        block = _mode_block(
            f"portfolio_{t}.json",
            f"predictions_{t}.json",
            f"shares_{ticker}",
            price,
        )
        block["ticker"]      = ticker
        block["price"]       = round(price, 2) if price else None
        block["market_open"] = _is_market_open()
        shares_data[ticker]  = block

    return jsonify({
        "btc_price":   btc_price,
        "now":         datetime.now().strftime("%H:%M:%S"),
        "market_open": _is_market_open(),
        "swing":       swing,
        "scalp":       scalp,
        "shares":      shares_data,
    })


@app.route("/api/start/<path:mode>", methods=["POST"])
def api_start(mode):
    if _is_running(mode):
        return jsonify({"ok": False, "error": f"{mode} already running"})

    if mode == "swing":
        cmd      = [sys.executable, "main.py"]
        log_path = "log_swing.txt"
    elif mode == "scalp":
        cmd      = [sys.executable, "main.py", "-scalping"]
        log_path = "log_scalp.txt"
    elif mode.startswith("shares_"):
        ticker   = mode[7:].upper()
        if not re.match(r"^[A-Z]{1,5}$", ticker):
            return jsonify({"ok": False, "error": f"Invalid ticker: {ticker}"}), 400
        cmd      = [sys.executable, "main.py", "-shares", ticker]
        log_path = f"log_{ticker.lower()}.txt"
    else:
        return jsonify({"ok": False, "error": "unknown mode"}), 400

    proc = _start_process(mode, cmd, log_path)
    return jsonify({"ok": True, "pid": proc.pid})


@app.route("/api/stop/<path:mode>", methods=["POST"])
def api_stop(mode):
    if not _stop_process(mode):
        return jsonify({"ok": False, "error": f"{mode} not running"})
    return jsonify({"ok": True})


@app.route("/api/log/<path:mode>")
def api_log(mode):
    if mode == "swing":
        log_path = "log_swing.txt"
    elif mode == "scalp":
        log_path = "log_scalp.txt"
    elif mode.startswith("shares_"):
        ticker   = mode[7:].lower()
        log_path = f"log_{ticker}.txt"
    else:
        return jsonify({"ok": False, "lines": []}), 400

    n = int(request.args.get("n", 150))
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
        return jsonify({"ok": True, "lines": [l.rstrip() for l in lines[-n:]], "total": len(lines)})
    except FileNotFoundError:
        return jsonify({"ok": True, "lines": ["No log yet — start the mode to begin logging."], "total": 0})


@app.route("/api/shares/remove/<ticker>", methods=["POST"])
def api_shares_remove(ticker):
    """Stop process and delete portfolio/predictions files for a ticker."""
    ticker = ticker.upper()
    t      = ticker.lower()
    _stop_process(f"shares_{ticker}")
    removed = []
    for f in [f"portfolio_{t}.json", f"predictions_{t}.json", f"log_{t}.txt",
              f"meta_{t}.json", f"tft_{t}_a.ckpt", f"tft_{t}_b.ckpt", f"tft_{t}_c.ckpt"]:
        path = os.path.join(CWD, f)
        if os.path.exists(path):
            os.remove(path)
            removed.append(f)
    return jsonify({"ok": True, "removed": removed})


if __name__ == "__main__":
    print("Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
