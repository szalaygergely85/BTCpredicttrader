import json
import os
from datetime import datetime, timedelta

PORTFOLIO_PATH = "portfolio.json"
STARTING_USDT  = 100.0
FEE_RATE       = 0.001  # 0.1% Binance taker fee

# Minimum minutes between opening new entries (BUY / OPEN SHORT).
# Prevents chasing rapid repeat signals and reduces fee drag.
# Swing: 60 min (1 candle).  Scalp: set in main.py via COOLDOWN_MINUTES.
COOLDOWN_MINUTES = 60

def load_portfolio():
    if not os.path.exists(PORTFOLIO_PATH):
        portfolio = {
            "usdt": STARTING_USDT,
            "btc": 0.0,
            "short": None,
            "long_entry_price": None,
            "long_entry_atr": None,
            "last_entry_time": None,
            "trades": [],
            "starting_usdt": STARTING_USDT,
        }
        save_portfolio(portfolio)
        return portfolio
    try:
        with open(PORTFOLIO_PATH) as f:
            data = json.load(f)
        if "long_entry_price" not in data:
            data["long_entry_price"] = None
        if "long_entry_atr" not in data:
            data["long_entry_atr"] = None
        if "last_entry_time" not in data:
            data["last_entry_time"] = None
        return data
    except Exception:
        print("  [trader] portfolio.json corrupted, resetting.")
        portfolio = {
            "usdt": STARTING_USDT,
            "btc": 0.0,
            "short": None,
            "long_entry_price": None,
            "long_entry_atr": None,
            "last_entry_time": None,
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

def _record_trade(portfolio, action, price, btc, usdt, note="", fee=0.0):
    total_value = _total_value(portfolio, price)
    pnl = total_value - portfolio["starting_usdt"]
    portfolio["trades"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "action": action,
        "price": round(price, 2),
        "btc": round(btc, 6),
        "usdt": round(usdt, 2),
        "fee": round(fee, 4),
        "note": note,
        "portfolio_value": round(total_value, 2),
        "pnl": round(pnl, 2),
    })

def _total_value(portfolio, price):
    value = portfolio["usdt"] + portfolio["btc"] * price
    if portfolio.get("short"):
        short = portfolio["short"]
        short_pnl = (short["entry_price"] - price) * short["btc"]
        value += short["collateral"] + short_pnl
    return value

def check_stops(current_price):
    """
    Uses ATR-based levels when available, falls back to fixed %.
    Long:  SL = entry - 1.5×ATR (or -2%),  TP = entry + 2.5×ATR (or +3%)
    Short: SL = entry + 1.5×ATR (or +2%),  TP = entry - 2.5×ATR (or -3%)
    """
    portfolio = load_portfolio()

    if portfolio["btc"] > 0.0 and portfolio.get("long_entry_price"):
        entry = portfolio["long_entry_price"]
        atr = portfolio.get("long_entry_atr")
        if atr and atr > 0:
            sl_price = entry - 1.5 * atr
            tp_price = entry + 2.5 * atr
        else:
            sl_price = entry * 0.98
            tp_price = entry * 1.03
        if current_price <= sl_price:
            return "STOP_LOSS", "long"
        if current_price >= tp_price:
            return "TAKE_PROFIT", "long"

    if portfolio.get("short"):
        short = portfolio["short"]
        entry = short["entry_price"]
        atr = short.get("entry_atr")
        if atr and atr > 0:
            sl_price = entry + 1.5 * atr
            tp_price = entry - 2.5 * atr
        else:
            sl_price = entry * 1.02
            tp_price = entry * 0.97
        if current_price >= sl_price:
            return "STOP_LOSS", "short"
        if current_price <= tp_price:
            return "TAKE_PROFIT", "short"

    return None, None

def execute_stop(stop_type, position_type, current_price):
    """Close a position due to stop loss or take profit trigger."""
    portfolio = load_portfolio()

    if position_type == "long" and portfolio["btc"] > 0.0:
        entry = portfolio.get("long_entry_price", current_price)
        pct_change = (current_price - entry) / entry * 100
        gross = portfolio["btc"] * current_price
        fee = gross * FEE_RATE
        received = gross - fee
        btc_sold = portfolio["btc"]
        portfolio["usdt"] += received
        note = f"entry ${entry:,.2f}, {pct_change:+.2f}%"
        action = f"{stop_type} LONG"
        _record_trade(portfolio, action, current_price, btc_sold, received, note=note, fee=fee)
        trade = portfolio["trades"][-1]
        portfolio["btc"] = 0.0
        portfolio["long_entry_price"] = None
        portfolio["long_entry_atr"] = None
        save_portfolio(portfolio)
        return trade, portfolio

    if position_type == "short" and portfolio.get("short"):
        short = portfolio["short"]
        entry = short["entry_price"]
        pct_change = (current_price - entry) / entry * 100
        short_pnl = (entry - current_price) * short["btc"]
        fee = abs(short["btc"] * current_price) * FEE_RATE
        returned = max(short["collateral"] + short_pnl - fee, 0)
        portfolio["usdt"] += returned
        note = f"entry ${entry:,.2f}, {pct_change:+.2f}%"
        action = f"{stop_type} SHORT"
        _record_trade(portfolio, action, current_price, short["btc"], abs(short_pnl), note=note, fee=fee)
        trade = portfolio["trades"][-1]
        portfolio["short"] = None
        save_portfolio(portfolio)
        return trade, portfolio

    return None, portfolio

def _in_cooldown(portfolio):
    """True if we opened a new entry position within COOLDOWN_MINUTES."""
    last = portfolio.get("last_entry_time")
    if not last:
        return False
    try:
        elapsed = (datetime.now() - datetime.fromisoformat(last)).total_seconds() / 60
        return elapsed < COOLDOWN_MINUTES
    except Exception:
        return False


def execute(signal, price, buy_pct=None, allow_new_short=True, atr=None):
    portfolio = load_portfolio()
    trade = None

    # --- CLOSE SHORT if BUY signal (closes are never gated by cooldown) ---
    if signal in ("BUY", "STRONG BUY") and portfolio.get("short"):
        short = portfolio["short"]
        short_pnl = (short["entry_price"] - price) * short["btc"]
        fee = abs(short["btc"] * price) * FEE_RATE
        returned = max(short["collateral"] + short_pnl - fee, 0)
        portfolio["usdt"] += returned
        result = "profit" if short_pnl >= 0 else "loss"
        _record_trade(portfolio, "CLOSE SHORT", price, short["btc"], abs(short_pnl),
                      note=f"{result} ${short_pnl:+.4f}", fee=fee)
        trade = portfolio["trades"][-1]
        portfolio["short"] = None

    # --- BUY LONG ---
    if signal in ("BUY", "STRONG BUY") and buy_pct and portfolio["usdt"] > 1.0 and not _in_cooldown(portfolio):
        spend = portfolio["usdt"] * buy_pct
        fee = spend * FEE_RATE
        bought = (spend - fee) / price
        portfolio["usdt"] -= spend
        # Average entry price when pyramiding into an existing position
        if portfolio["btc"] > 0 and portfolio["long_entry_price"]:
            old_btc = portfolio["btc"]
            old_entry = portfolio["long_entry_price"]
            portfolio["long_entry_price"] = (old_btc * old_entry + bought * price) / (old_btc + bought)
        else:
            portfolio["long_entry_price"] = price
        if atr:
            portfolio["long_entry_atr"] = atr
        portfolio["btc"] += bought
        portfolio["last_entry_time"] = datetime.now().isoformat()
        _record_trade(portfolio, "BUY", price, bought, spend,
                      note=f"{int(buy_pct*100)}% of portfolio", fee=fee)
        trade = portfolio["trades"][-1]

    # --- CLOSE LONG if SELL signal ---
    if signal in ("SELL", "STRONG SELL") and portfolio["btc"] > 0.0:
        gross = portfolio["btc"] * price
        fee = gross * FEE_RATE
        received = gross - fee
        portfolio["usdt"] += received
        _record_trade(portfolio, "SELL", price, portfolio["btc"], received, fee=fee)
        trade = portfolio["trades"][-1]
        portfolio["btc"] = 0.0
        portfolio["long_entry_price"] = None
        portfolio["long_entry_atr"] = None

    # --- OPEN SHORT (only allowed in bear regime, respects cooldown) ---
    if signal in ("SELL", "STRONG SELL") and allow_new_short and not portfolio.get("short") and portfolio["usdt"] > 1.0 and not _in_cooldown(portfolio):
        pct = 0.10
        collateral = portfolio["usdt"] * pct
        fee = collateral * FEE_RATE
        net_collateral = collateral - fee
        short_btc = net_collateral / price
        portfolio["usdt"] -= collateral
        portfolio["short"] = {
            "entry_price": price,
            "btc": short_btc,
            "collateral": net_collateral,
            "entry_atr": atr,
        }
        portfolio["last_entry_time"] = datetime.now().isoformat()
        _record_trade(portfolio, "OPEN SHORT", price, short_btc, collateral,
                      note=f"entry ${price:,.2f}", fee=fee)
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
        "long_entry_price": portfolio.get("long_entry_price"),
        "long_entry_atr": portfolio.get("long_entry_atr"),
        "short": short_info,
        "total_value": round(total_value, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
    }
