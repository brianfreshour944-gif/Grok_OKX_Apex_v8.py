#!/usr/bin/env python3
# main_bot.py — Entry point. Contains only the main trading loop.
# All logic lives in the imported modules below.

import asyncio
import os
import time

import numpy as np
import psycopg2
import torch

from alpaca.trading.enums import OrderSide

from config import (
    logger, BOT_NAME, SEQUENCE_LEN, MODEL_PATH,
    MAX_OPEN_POSITIONS, MAX_DRAWDOWN_STOP, MAX_HOLD_HOURS,
    BASE_RISK_PERCENT, MIN_POSITION_USD, MIN_ORDER_USD,
    MIN_HOLD_HOURS_BEFORE_SIGNAL,
    get_regime_params, fmt_price, trading_client, SYMBOLS,
)
from database import report_equity
from data_feeds import scan_stable_assets, get_clean_ohlcv_dataframe
from regime import compute_regime_and_trend, calculate_adjusted_risk
from portfolio import (
    get_all_positions, get_buying_power,
    sell_largest_position, sync_existing_positions, normalize_symbol,
)
from orders import place_order
from feature_engineering import add_features, FEATURE_COLS
from ml_predictor import GrokGQA_Transformer
import joblib

# ── Global state ───────────────────────────────────────────────────────────────
cooldown_until: dict = {symbol: 0.0 for symbol in SYMBOLS}
entry_time:     dict = {}
start_equity         = None


# ── ML predictor (inline, tightly coupled to main loop state) ─────────────────
class SafeMLPredictor:
    def __init__(self, model_path, seq_len=32):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.model   = GrokGQA_Transformer(
            input_dim=len(FEATURE_COLS), seq_len=seq_len,
            embed_dim=128, num_layers=8, num_q_heads=16, num_kv_heads=4, dropout=0.1,
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        scaler_path  = os.path.join(os.path.dirname(model_path), "feature_scaler.pkl")
        self.scaler  = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    def predict(self, df) -> float:
        try:
            df_feat = add_features(df.copy())
            data    = df_feat.tail(self.seq_len).values.astype(np.float32)
            if len(data) < self.seq_len:
                return 0.5
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = np.clip(data, -1e6, 1e6)
            if self.scaler:
                data = self.scaler.transform(data).astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.tensor(data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return float(self.model(x).item())
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return 0.5


predictor = SafeMLPredictor(MODEL_PATH, SEQUENCE_LEN)


# ── Main trading loop ──────────────────────────────────────────────────────────
async def run_trading_mode():
    global start_equity

    sync_existing_positions(entry_time)
    logger.info("🚀 Grok Apex Ironclad Bot v9 - Cutting Edge with DOGE Started")

    # Sync starting_equity to DB if configured
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.info("DATABASE_URL not set — skipping starting_equity sync")
    else:
        try:
            with psycopg2.connect(db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "ALTER TABLE bot_status ADD COLUMN IF NOT EXISTS starting_equity NUMERIC"
                    )
                    cur.execute("""
                        INSERT INTO bot_status
                            (bot_name, starting_equity, live_equity, live_equity_updated_at, last_update)
                        VALUES (%s, %s, %s, NOW(), NOW())
                        ON CONFLICT (bot_name) DO UPDATE
                        SET starting_equity          = EXCLUDED.starting_equity,
                            live_equity              = EXCLUDED.live_equity,
                            live_equity_updated_at   = NOW(),
                            last_update              = NOW()
                    """, (BOT_NAME, float(start_equity or 0), float(start_equity or 0)))
                    conn.commit()
                    logger.info(f"✅ Synced starting_equity to ${start_equity:.2f} in database")
        except Exception as e:
            logger.warning(f"⚠️ Could not sync starting_equity: {e}")

    while True:
        try:
            account      = trading_client.get_account()
            equity       = float(account.equity)
            if start_equity is None:
                start_equity = equity

            report_equity(BOT_NAME, equity)

            drawdown = (equity - start_equity) / start_equity * 100
            if drawdown < MAX_DRAWDOWN_STOP:
                logger.error("🚨 MAX DRAWDOWN HIT - Stopping trading")
                break

            current_positions   = get_all_positions()
            open_count          = len(current_positions)
            total_value         = sum(p["market_value"] for p in current_positions.values())
            buying_power        = get_buying_power()
            max_portfolio_value = equity * BASE_RISK_PERCENT * MAX_OPEN_POSITIONS

            logger.info(
                f"📊 Cycle | Positions: {open_count}/{MAX_OPEN_POSITIONS} | "
                f"Value: ${total_value:.2f} | BP: ${buying_power:.2f} | "
                f"Drawdown: {drawdown:.2f}% | Portfolio cap: ${max_portfolio_value:.2f}"
            )

            buys_allowed            = True
            running_portfolio_value = total_value

            if total_value >= max_portfolio_value:
                logger.warning(f"📉 Portfolio ${total_value:.2f} >= cap ${max_portfolio_value:.2f}")
                sell_largest_position()
                buys_allowed = False

            if open_count >= MAX_OPEN_POSITIONS:
                logger.info(f"🛑 Max positions reached ({open_count}/{MAX_OPEN_POSITIONS}). Holding off on buys.")
                buys_allowed = False

            active_symbols = await scan_stable_assets(limit_scope=18)
            for sym in active_symbols:
                cooldown_until.setdefault(sym, 0.0)

            for symbol in active_symbols:
                now        = time.time()
                alpaca_sym = normalize_symbol(symbol)

                if now < cooldown_until.get(symbol, 0):
                    continue

                df = await get_clean_ohlcv_dataframe(symbol)
                if df is None:
                    continue

                regime, trend, atr_pct = compute_regime_and_trend(df)
                regime_params          = get_regime_params(regime)
                signal                 = predictor.predict(df)
                price                  = df["close"].iloc[-1]

                if price <= 0:
                    continue

                pos_data     = current_positions.get(alpaca_sym)
                has_position = (
                    pos_data is not None
                    and pos_data["qty"] > 0
                    and pos_data["market_value"] >= MIN_POSITION_USD
                )
                qty_held  = pos_data["qty"]       if has_position else 0.0
                avg_entry = pos_data["avg_entry"]  if has_position else 0.0

                # ── EXIT ───────────────────────────────────────────────────────
                if has_position:
                    pnl_pct    = (price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
                    held_hours = (now - entry_time.get(symbol, now)) / 3600
                    exit_reason = None

                    if pnl_pct >= regime_params["profit_target_pct"]:
                        exit_reason = f"🎯 Profit target ({pnl_pct*100:.2f}%) [{regime}]"
                    elif pnl_pct <= -regime_params["stop_loss_pct"]:
                        exit_reason = f"🛑 Stop loss ({pnl_pct*100:.2f}%) [{regime}]"
                    elif held_hours >= MAX_HOLD_HOURS:
                        exit_reason = f"⏰ Max hold time ({held_hours:.1f}h)"
                    elif held_hours >= MIN_HOLD_HOURS_BEFORE_SIGNAL and signal < regime_params["sell_signal"]:
                        exit_reason = f"📉 Signal weak ({signal:.3f}) [{regime}]"

                    if exit_reason:
                        logger.info(f"{exit_reason} — SELL {symbol} @ {fmt_price(price)} | Regime: {regime}")
                        success = await place_order(symbol, OrderSide.SELL, qty_held, price)
                        if success:
                            cooldown_until[symbol] = now + 1800
                            entry_time.pop(symbol, None)
                    else:
                        logger.info(
                            f"📌 Holding {symbol} | Entry: ${fmt_price(avg_entry)} | "
                            f"Now: ${fmt_price(price)} | PnL: {pnl_pct*100:+.2f}% | "
                            f"Held: {held_hours:.1f}h | Signal: {signal:.3f}"
                        )
                    await asyncio.sleep(2)
                    continue

                # ── ENTRY ──────────────────────────────────────────────────────
                if signal > regime_params["buy_signal"]:
                    if not buys_allowed:
                        logger.info(f"🚫 BUY suppressed for {symbol} (cap/position limit)")
                        await asyncio.sleep(2)
                        continue

                    adjusted_risk = calculate_adjusted_risk(equity, atr_pct)
                    qty           = adjusted_risk / price
                    trade_value   = qty * price

                    if trade_value < MIN_ORDER_USD:
                        logger.info(
                            f"🚫 BUY skipped {symbol}: ${trade_value:.2f} below "
                            f"${MIN_ORDER_USD:.2f} minimum (ATR%: {atr_pct:.2f}%)"
                        )
                        await asyncio.sleep(2)
                        continue

                    headroom = max_portfolio_value - running_portfolio_value
                    if headroom < trade_value:
                        logger.warning(f"🚫 BUY blocked {symbol}: only ${headroom:.2f} headroom (need ${trade_value:.2f})")
                        buys_allowed = False
                        await asyncio.sleep(2)
                        continue

                    if buying_power < trade_value:
                        logger.warning(f"🚫 BUY blocked {symbol}: buying power ${buying_power:.2f} < trade ${trade_value:.2f}")
                        buys_allowed = False
                        await asyncio.sleep(2)
                        continue

                    logger.info(
                        f"🟢 BUY {symbol} @ {fmt_price(price)} | Regime: {regime} | "
                        f"Signal: {signal:.3f} | Positions: {open_count}/{MAX_OPEN_POSITIONS}"
                    )
                    success = await place_order(symbol, OrderSide.BUY, qty, price)
                    if success:
                        cooldown_until[symbol]   = now + 900
                        entry_time[symbol]        = now
                        running_portfolio_value  += trade_value
                        open_count               += 1
                        if open_count >= MAX_OPEN_POSITIONS:
                            logger.info("🔒 Max positions reached — no more buys this cycle.")
                            buys_allowed = False

                await asyncio.sleep(2)

            await asyncio.sleep(40)

        except Exception as e:
            logger.error(f"Critical loop error: {e}")
            await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(run_trading_mode())
