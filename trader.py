import json
import os
from datetime import datetime

PORTFOLIO_PATH = "portfolio.json"
STARTING_USDT = 2.0

def load_portfolio():
    if not os.path.exists(PORTFOLIO_PATH):
        portfolio = {
            "usdt": STARTING_USDT,
            "btc": 0.0,
            "short": None,
            "trades": [],
            "starting_usdt": STARTING_USDT,
        }
        save_portfolio(portfolio)
        return portfolio
    try:
        with open(PORTFOLIO_PATH) as f:
            return json.load(f)
    except Exception:
        print("  [trader] portfolio.json corrupted, resetting.")
        portfolio = {
            "usdt": STARTING_USDT,
            "btc": 0.0,
            "short": None,
            "trades": [],
            "starting_usdt": STARTING_USDT,
        }
        save_portfolio(portfolio)
        return portfolio

def save_portfolio(portfolio):
    tmp = PORTFOLIO_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(portfolio, f, indent=2)
    os.replace(tmp, PORTFOLIO_PATH)

def _record_trade(portfolio, action, price, btc, usdt, note=""):
    total_value = _total_value(portfolio, price)
    pnl = total_value - portfolio["starting_usdt"]
    portfolio["trades"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "action": action,
        "price": round(price, 2),
        "btc": round(btc, 6),
        "usdt": round(usdt, 2),
        "note": note,
        "portfolio_value": round(total_value, 2),
        "pnl": round(pnl, 2),
    })

def _total_value(portfolio, price):
    value = portfolio["usdt"] + portfolio["btc"] * price
    if portfolio.get("short"):
        short = portfolio["short"]
        # unrealized short P&L: we profit if price fell below entry
        short_pnl = (short["entry_price"] - price) * short["btc"]
        value += short["collateral"] + short_pnl
    return value

def execute(signal, price):
    portfolio = load_portfolio()
    trade = None

    # --- CLOSE SHORT if BUY signal ---
    if signal in ("BUY", "STRONG BUY") and portfolio.get("short"):
        short = portfolio["short"]
        short_pnl = (short["entry_price"] - price) * short["btc"]
        returned = short["collateral"] + short_pnl
        portfolio["usdt"] += max(returned, 0)  # can't go below 0
        result = "profit" if short_pnl >= 0 else "loss"
        _record_trade(portfolio, "CLOSE SHORT", price, short["btc"], abs(short_pnl),
                      note=f"{result} ${short_pnl:+.4f}")
        trade = portfolio["trades"][-1]
        portfolio["short"] = None

    # --- BUY LONG ---
    if signal in ("BUY", "STRONG BUY") and portfolio["usdt"] > 1.0:
        pct = 0.50 if portfolio["usdt"] < 1000 else 0.10
        spend = portfolio["usdt"] * pct
        bought = spend / price
        portfolio["usdt"] -= spend
        portfolio["btc"] += bought
        _record_trade(portfolio, "BUY", price, bought, spend)
        trade = portfolio["trades"][-1]

    # --- CLOSE LONG if SELL signal ---
    if signal in ("SELL", "STRONG SELL") and portfolio["btc"] > 0.0:
        received = portfolio["btc"] * price
        portfolio["usdt"] += received
        _record_trade(portfolio, "SELL", price, portfolio["btc"], received)
        trade = portfolio["trades"][-1]
        portfolio["btc"] = 0.0

    # --- OPEN SHORT ---
    if signal in ("SELL", "STRONG SELL") and not portfolio.get("short") and portfolio["usdt"] > 1.0:
        pct = 0.50 if portfolio["usdt"] < 1000 else 0.10
        collateral = portfolio["usdt"] * pct
        short_btc = collateral / price
        portfolio["usdt"] -= collateral
        portfolio["short"] = {
            "entry_price": price,
            "btc": short_btc,
            "collateral": collateral,
        }
        _record_trade(portfolio, "OPEN SHORT", price, short_btc, collateral,
                      note=f"entry ${price:,.2f}")
        trade = portfolio["trades"][-1]

    save_portfolio(portfolio)
    return trade, portfolio

def status(price):
    portfolio = load_portfolio()
    total_value = _total_value(portfolio, price)
    pnl = total_value - portfolio["starting_usdt"]
    pnl_pct = (pnl / portfolio["starting_usdt"]) * 100

    short_info = None
    if portfolio.get("short"):
        short = portfolio["short"]
        short_pnl = (short["entry_price"] - price) * short["btc"]
        short_info = {
            "entry_price": short["entry_price"],
            "btc": round(short["btc"], 6),
            "pnl": round(short_pnl, 4),
        }

    return {
        "usdt": round(portfolio["usdt"], 2),
        "btc": round(portfolio["btc"], 6),
        "short": short_info,
        "total_value": round(total_value, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
    }
