#!/usr/bin/env python3
# main_bot.py — Entry point. Contains only the main trading loop.
# All logic lives in the imported modules below.

import asyncio
import os
import sys
import time

import numpy as np
import psycopg2
import torch

from alpaca.trading.enums import OrderSide

from config import (
    logger, BOT_NAME, SEQUENCE_LEN, MODEL_PATH,
    MAX_OPEN_POSITIONS, MAX_DRAWDOWN_STOP, MAX_HOLD_HOURS,
    BASE_RISK_PERCENT, MIN_POSITION_USD, MIN_ORDER_USD, MAX_SINGLE_TRADE_USD,
    MIN_HOLD_HOURS_BEFORE_SIGNAL, BUY_SIGNAL,
    COOLDOWN_SECONDS_BUY, COOLDOWN_SECONDS_SELL, SLEEP_PER_LOOP,
    get_regime_params, fmt_price, trading_client, SYMBOLS,
)
from database import report_equity, init_db, save_bot_state, load_bot_state
from data_feeds import get_clean_ohlcv_dataframe, get_orderbook_with_retry
from regime import compute_regime_and_trend, calculate_adjusted_risk
from portfolio import (
    get_all_positions, get_buying_power,
    sell_largest_position, sync_existing_positions, normalize_symbol,
    swap_weakest_position, cancel_stale_orders, write_heartbeat,
    calculate_kelly_multiplier
)
from orders import place_order
from feature_engineering import add_features, FEATURE_COLS
from notifications import send_discord_alert
from ml_predictor import SafeMLPredictor
import joblib

# --- Global state ---
cooldown_until: dict = {}
entry_time:     dict = {}
latest_signals: dict = {}
highest_prices: dict = {}
start_equity         = None


# NOTE: previously this file had its own duplicate SafeMLPredictor class,
# separate from the one in ml_predictor.py. Consolidated to one source of
# truth so the live bot and diagnostic/backtest tooling can't silently
# drift apart.

predictor = SafeMLPredictor(model_path=MODEL_PATH, seq_len=SEQUENCE_LEN)



# ── Main trading loop ──────────────────────────────────────────────────────────
async def run_trading_mode():
    global start_equity, cooldown_until, entry_time, latest_signals, highest_prices

    init_db()  # create DB tables once at startup (no-op if DATABASE_URL not set)
    cooldown_until, entry_time, latest_signals, highest_prices = load_bot_state()
    sync_existing_positions(entry_time)

    logger.info("🚀 Grok Apex Ironclad Bot v9 - Cutting Edge Started")

    # ── Startup cleanup: close any positions in symbols we no longer trade ──────
    try:
        all_pos = trading_client.get_all_positions()
        active_alpaca_syms = {normalize_symbol(s) for s in SYMBOLS}
        for p in all_pos:
            market_val = float(p.market_value)
            if p.symbol not in active_alpaca_syms:
                if market_val < 1.0:
                    logger.info(f"🧹 Ignoring unsellable dust position {p.symbol} (${market_val:.4f})")
                    continue
                try:
                    trading_client.close_position(p.symbol)
                    logger.info(f"🧹 Startup cleanup: closed stale position {p.symbol} (${market_val:.2f})")
                except Exception as close_err:
                    logger.warning(f"⚠️ Could not close stale position {p.symbol}: {close_err}")
    except Exception as e:
        logger.warning(f"⚠️ Startup cleanup failed: {e}")

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
                    logger.info(f"✅ Synced starting_equity to ${start_equity or 0:.2f} in database")
        except Exception as e:
            logger.warning(f"⚠️ Could not sync starting_equity: {e}")

    while True:
        try:
            write_heartbeat()
            cancel_stale_orders(timeout_minutes=3)
            
            account      = trading_client.get_account()
            equity       = float(account.equity)
            # Guard: never pin start_equity to 0.0. If equity is ever
            # reported as 0 (bad API response, zero-balance account, etc.)
            # a naive `if start_equity is None: start_equity = equity` would
            # permanently lock start_equity at 0.0, causing
            # (equity - start_equity) / start_equity to raise
            # ZeroDivisionError EVERY cycle forever — an unrecoverable
            # crash-loop that looks like a transient error in the logs.
            # Instead, skip the drawdown check entirely on cycles where we
            # don't yet have a valid (>0) baseline equity to compare against.
            if start_equity is None and equity > 0:
                start_equity = equity

            report_equity(BOT_NAME, equity)

            drawdown = None
            if not start_equity:
                logger.warning(
                    f"⚠️ No valid starting equity yet (equity=${equity:.2f}) — "
                    f"skipping drawdown check this cycle."
                )
            else:
                drawdown = (equity - start_equity) / start_equity * 100
                if drawdown < MAX_DRAWDOWN_STOP:
                    logger.error("🚨 MAX DRAWDOWN HIT - Stopping trading")
                    break

            current_positions   = get_all_positions()
            # Exclude true dust (< MIN_POSITION_USD) from open_count so they
            # don't block new entries when they can't be sold anyway.
            open_count          = sum(1 for p in current_positions.values() if p["market_value"] >= MIN_POSITION_USD)
            total_value         = sum(p["market_value"] for p in current_positions.values())
            buying_power        = get_buying_power()
            max_portfolio_value = equity * BASE_RISK_PERCENT * MAX_OPEN_POSITIONS

            drawdown_str = f"{drawdown:.2f}%" if drawdown is not None else "N/A"
            logger.info(
                f"📊 Cycle | Positions: {open_count}/{MAX_OPEN_POSITIONS} | "
                f"Value: ${total_value:.2f} | BP: ${buying_power:.2f} | "
                f"Drawdown: {drawdown_str} | Portfolio cap: ${max_portfolio_value:.2f}"
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

            # Use SYMBOLS directly — no need to scan 18 assets when we only trade 3
            now = time.time()
            symbols_to_process = SYMBOLS

            # PARALLEL FETCH: Fetch all OHLCV dataframes concurrently
            fetch_tasks = [get_clean_ohlcv_dataframe(sym) for sym in symbols_to_process]
            dfs = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            for symbol, df in zip(symbols_to_process, dfs):
                if df is None or isinstance(df, Exception):
                    if isinstance(df, Exception):
                        logger.error(f"Failed to fetch data for {symbol}: {df}")
                    continue

                alpaca_sym = normalize_symbol(symbol)

                regime, trend, atr_pct = compute_regime_and_trend(df)
                regime_params = get_regime_params(regime)
                position_size_multiplier = 1.0  # regime_flag removed (was Windows-only path)

                # ── DIAGNOSTIC ML PREDICTION BLOCK ────────────────────────────
                try:
                    signal = predictor.predict(df)
                    latest_signals[symbol] = signal
                    # FORCE LOG AT INFO LEVEL - this will appear 100%
                    logger.info(
                        f"🔬 TEST | Asset: {symbol} | ML Signal: {signal:.4f} | "
                        f"Threshold: {regime_params['buy_signal']:.4f} (regime={regime}) | Trend: {trend}"
                    )
                except Exception as ml_error:
                    logger.error(f"💥 ML CRASH on {symbol}: {repr(ml_error)}")
                    continue  # skip this asset and move to the next

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
                    
                    highest_seen = highest_prices.get(symbol, avg_entry)
                    if price > highest_seen:
                        highest_prices[symbol] = price
                        highest_seen = price

                    # Time-Decay Stop Loss Logic
                    dynamic_sl_pct = regime_params["stop_loss_pct"]
                    if held_hours >= 2.0:
                        dynamic_sl_pct = regime_params["stop_loss_pct"] * 0.5  # Halve the stop loss
                    elif held_hours >= 1.0:
                        dynamic_sl_pct = regime_params["stop_loss_pct"] * 0.75 # Tighten by 25%

                    if highest_seen > avg_entry * (1 + regime_params["profit_target_pct"]):
                        # Trailing Mode Active (1% trailing stop from peak)
                        trailing_stop_price = highest_seen * 0.99
                        if price <= trailing_stop_price:
                            exit_reason = f"📉 Trailing Stop triggered (Peak: ${fmt_price(highest_seen)}, PnL: {pnl_pct*100:.2f}%) [{regime}]"
                    elif pnl_pct <= -dynamic_sl_pct:
                        exit_reason = f"🛑 Time-Decay Stop loss ({pnl_pct*100:.2f}% <= -{dynamic_sl_pct*100:.2f}%) [{regime}]"
                    elif held_hours >= MAX_HOLD_HOURS:
                        exit_reason = f"⏰ Max hold time ({held_hours:.1f}h)"
                    elif held_hours >= MIN_HOLD_HOURS_BEFORE_SIGNAL and signal < regime_params["sell_signal"]:
                        exit_reason = f"📉 Signal weak ({signal:.3f}) [{regime}]"

                    if exit_reason:
                        logger.info(f"{exit_reason} — SELL {symbol} @ {fmt_price(price)} | Regime: {regime}")
                        success = await place_order(symbol, OrderSide.SELL, qty_held, price)
                        if success:
                            asyncio.create_task(send_discord_alert(
                                title=f"🔴 SELL {symbol}",
                                description=f"**Price:** ${price:.4f}\n**Reason:** {exit_reason}\n**Regime:** {regime}",
                                color=0xFF0000
                            ))
                            cooldown_until[symbol] = now + COOLDOWN_SECONDS_SELL
                            # We deliberately DO NOT pop entry_time or highest_prices here.
                            # If the order fails or gets canceled, we want to retain the hold-time and peak price.
                    else:
                        logger.info(
                            f"📌 Holding {symbol} | Entry: ${fmt_price(avg_entry)} | "
                            f"Now: ${fmt_price(price)} | Peak: ${fmt_price(highest_seen)} | "
                            f"PnL: {pnl_pct*100:+.2f}% | "
                            f"Held: {held_hours:.1f}h | Signal: {signal:.3f}"
                        )
                    await asyncio.sleep(2)
                    continue

                # ── ENTRY ──────────────────────────────────────────────────────
                # Skip entries if this symbol is on cooldown (e.g. recently sold/bought)
                if now < cooldown_until.get(symbol, 0.0):
                    await asyncio.sleep(2)
                    continue

                # Single source of truth: BUY_SIGNAL (config). EMA50 trend filter
                # (trend == "up") gates entries so we only buy with the wind at
                # our back. trend is computed by compute_regime_and_trend() via
                # close vs EMA-50.
                if trend == "up" and signal > regime_params["buy_signal"]:
                    # ── Whale Filter (Level 2 Execution Gate) ──
                    # If orderbook fetch fails, skip the trade rather than
                    # proceeding blindly without depth data.
                    _whale_veto = False
                    try:
                        book = await get_orderbook_with_retry(symbol)
                        # alpaca-py uses .s for size; fallback to .size for
                        # forward-compatibility if the attribute name changes.
                        def _quote_size(q):
                            return getattr(q, 's', None) or getattr(q, 'size', 0) or 0
                        total_bid_size = sum(_quote_size(b) for b in book.bids)
                        total_ask_size = sum(_quote_size(a) for a in book.asks)
                        imbalance = total_bid_size / total_ask_size if total_ask_size > 0 else 1.0
                        
                        if imbalance < 0.65:
                            logger.info(f"🚫 BUY skipped {symbol}: Extreme selling pressure ({imbalance:.2f})")
                            _whale_veto = True
                    except Exception as e:
                        logger.warning(f"⚠️ Orderbook fetch failed for {symbol} — skipping buy to be safe: {e}")
                        _whale_veto = True

                    # ── Sentinel Hard Veto & Volatility Protection Filter ──
                    if atr_pct > 6.0:
                        logger.info(f"🛑 BUY skipped {symbol} by Sentinel Veto: Extreme market volatility (ATR={atr_pct:.2f}%)")
                        await asyncio.sleep(2)
                        continue
                    elif regime == "wild" and signal < 0.70:
                        logger.info(f"🛑 BUY skipped {symbol} by Sentinel Veto: Low conviction in wild market regime (signal={signal:.3f})")
                        await asyncio.sleep(2)
                        continue

                    if _whale_veto:
                        await asyncio.sleep(2)
                        continue


                    if not buys_allowed:
                        sold_notional = swap_weakest_position(symbol, signal, latest_signals)
                        if sold_notional > 0:
                            logger.info(f"✅ Swapped weak position to make room for {symbol}. Proceeding with buy.")
                            buys_allowed = True
                            open_count = max(0, open_count - 1)
                            # The swap just freed up dollar-value headroom too —
                            # without this, running_portfolio_value stays stale
                            # (still counting the just-sold position's value),
                            # which could wrongly re-block the very buy the
                            # swap was meant to enable via the headroom check
                            # further down.
                            running_portfolio_value = max(0.0, running_portfolio_value - sold_notional)
                        else:
                            logger.info(f"🚫 BUY suppressed for {symbol} (cap/position limit and no weak swap found)")
                            await asyncio.sleep(2)
                            continue

                    # Dynamic Position Sizing using Kelly Criterion
                    kelly_mult = calculate_kelly_multiplier(
                        signal_prob=signal,
                        profit_target_pct=regime_params["profit_target_pct"],
                        stop_loss_pct=regime_params["stop_loss_pct"]
                    )
                    
                    adjusted_risk = calculate_adjusted_risk(equity, atr_pct) * position_size_multiplier * kelly_mult
                    qty           = adjusted_risk / price
                    trade_value   = min(qty * price, MAX_SINGLE_TRADE_USD)
                    qty           = trade_value / price

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
                        asyncio.create_task(send_discord_alert(
                            title=f"🟢 BUY {symbol}",
                            description=f"**Price:** ${price:.4f}\n**Signal:** {signal:.4f}\n**Size:** ${trade_value:.2f}\n**Regime:** {regime}",
                            color=0x00FF00
                        ))
                        cooldown_until[symbol]   = now + COOLDOWN_SECONDS_BUY
                        entry_time[symbol]        = now
                        highest_prices[symbol]    = price
                        running_portfolio_value  += trade_value
                        open_count               += 1
                        if open_count >= MAX_OPEN_POSITIONS:
                            logger.info("🔒 Max positions reached — no more buys this cycle.")
                            buys_allowed = False
            
            # Save state at the end of each cycle
            save_bot_state(cooldown_until, entry_time, latest_signals, highest_prices)

            # ── --once test flag: exit after first full cycle ──────────────────
            if "--once" in sys.argv:
                logger.info("🏁 --once flag passed. Cycle complete. Exiting.")
                return

            # Sleep once per entire cycle, not once per asset
            await asyncio.sleep(SLEEP_PER_LOOP)

        except Exception as e:
            logger.error(f"Critical loop error: {e}")
            await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(run_trading_mode())
