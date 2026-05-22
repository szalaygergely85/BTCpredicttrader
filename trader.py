import json
import os
from datetime import datetime

PORTFOLIO_PATH = "portfolio.json"
STARTING_USDT = 20.0

def load_portfolio():
    if not os.path.exists(PORTFOLIO_PATH):
        portfolio = {
            "usdt": STARTING_USDT,
            "btc": 0.0,
            "trades": [],
            "starting_usdt": STARTING_USDT,
        }
        save_portfolio(portfolio)
        return portfolio
    with open(PORTFOLIO_PATH) as f:
        return json.load(f)

def save_portfolio(portfolio):
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(portfolio, f, indent=2)

def execute(signal, price):
    portfolio = load_portfolio()
    usdt = portfolio["usdt"]
    btc = portfolio["btc"]
    action = None
    amount_usdt = 0.0
    amount_btc = 0.0

    if signal in ("BUY", "STRONG BUY") and usdt > 1.0:
        spend = usdt * 0.10  # buy with 10% of available USDT
        bought = spend / price
        portfolio["usdt"] -= spend
        portfolio["btc"] += bought
        action = "BUY"
        amount_usdt = spend
        amount_btc = bought

    elif signal in ("SELL", "STRONG SELL") and btc > 0.0:
        received = btc * price
        portfolio["usdt"] += received
        portfolio["btc"] = 0.0
        action = "SELL"
        amount_usdt = received
        amount_btc = btc

    if action:
        total_value = portfolio["usdt"] + portfolio["btc"] * price
        pnl = total_value - portfolio["starting_usdt"]
        trade = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "action": action,
            "price": round(price, 2),
            "btc": round(amount_btc, 6),
            "usdt": round(amount_usdt, 2),
            "portfolio_value": round(total_value, 2),
            "pnl": round(pnl, 2),
        }
        portfolio["trades"].append(trade)
        save_portfolio(portfolio)
        return trade, portfolio

    return None, portfolio

def status(price):
    portfolio = load_portfolio()
    total_value = portfolio["usdt"] + portfolio["btc"] * price
    pnl = total_value - portfolio["starting_usdt"]
    pnl_pct = (pnl / portfolio["starting_usdt"]) * 100
    return {
        "usdt": round(portfolio["usdt"], 2),
        "btc": round(portfolio["btc"], 6),
        "total_value": round(total_value, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
    }
