# utils.py
import requests
import os
from datetime import datetime, timezone

def log_trade(bot_name, symbol, side, qty, entry_price, pnl=None, exit_price=None, 
              status="open", entry_time=None, exit_time=None, **kwargs):
    """
    Send trade data to the Base44 dashboard API.
    Called from the bot after an order fills.
    """
    # Validate required fields
    if entry_price is None or float(entry_price) <= 0:
        return

    # If you have a Base44 API key – push to the dashboard
    api_key = os.getenv("DASHBOARD_API_KEY")
    if not api_key:
        # No key – just log locally
        print(f"[LOG_TRADE] {bot_name} | {symbol} | {side} | qty={qty} @ {entry_price}")
        return

    base_url = os.getenv("BASE44_API_URL", "https://api.base44.com/api/apps/YOUR_APP_ID/entities")
    headers = {"api-key": api_key, "Content-Type": "application/json"}

    # Build payload with all fields
    payload = {
        "bot_name": bot_name,
        "symbol": symbol,
        "side": side,
        "qty": float(qty),
        "entry_price": float(entry_price),
        "entry_time": entry_time or datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    if pnl is not None:
        payload["pnl"] = float(pnl)
    if exit_price is not None:
        payload["exit_price"] = float(exit_price)
        payload["exit_time"] = exit_time or datetime.now(timezone.utc).isoformat()

    try:
        # First, try to find an existing Trade record to update (if we have an ID)
        # For simplicity, we always POST a new trade. Adjust as needed.
        requests.post(f"{base_url}/Trade/", json=payload, headers=headers, timeout=5)
    except Exception as e:
        print(f"Base44 log_trade error: {e}")


def push_heartbeat(bot_name, equity, buying_power, daily_pnl, total_pnl,
                   open_positions, trades_today, status="running", error_msg=None, **kwargs):
    """
    Send heartbeat (account status) to the Base44 dashboard.
    Called every loop iteration.
    """
    api_key = os.getenv("DASHBOARD_API_KEY")
    if not api_key:
        print(f"[HEARTBEAT] {bot_name} | equity=${equity:.2f} | positions={open_positions}")
        return

    base_url = os.getenv("BASE44_API_URL", "https://api.base44.com/api/apps/YOUR_APP_ID/entities")
    headers = {"api-key": api_key, "Content-Type": "application/json"}

    payload = {
        "bot_name": bot_name,
        "status": status,
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        "account_equity": float(equity),
        "buying_power": float(buying_power),
        "daily_pnl": float(daily_pnl),
        "total_pnl": float(total_pnl),
        "open_positions_count": int(open_positions),
        "trades_today": int(trades_today),
    }
    if error_msg:
        payload["error_message"] = error_msg

    try:
        # Find existing BotStatus record for this bot
        resp = requests.get(f"{base_url}/BotStatus/", headers=headers,
                            params={"bot_name": bot_name}, timeout=5)
        records = resp.json() if resp.ok else []
        if records:
            # Update existing
            record_id = records[0]['id']
            requests.put(f"{base_url}/BotStatus/{record_id}/", json=payload, headers=headers, timeout=5)
        else:
            # Create new
            requests.post(f"{base_url}/BotStatus/", json=payload, headers=headers, timeout=5)
    except Exception as e:
        print(f"Base44 heartbeat error: {e}")
