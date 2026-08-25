#!/usr/bin/env python3
# main_bot.py — Entry point. Contains only the main trading loop.
# All logic lives in the imported modules below.

import asyncio
import os
import sys
import time

# Ensure UTF-8 encoding for stdout/stderr to prevent UnicodeEncodeError on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import psycopg2

from alpaca.trading.enums import OrderSide

from config import (
    logger, BOT_NAME, SEQUENCE_LEN, MODEL_PATH,
    MAX_OPEN_POSITIONS, MAX_DRAWDOWN_STOP, MAX_HOLD_HOURS,
    BASE_RISK_PERCENT, MIN_POSITION_USD, MIN_ORDER_USD, MAX_SINGLE_TRADE_USD,
    MIN_HOLD_HOURS_BEFORE_SIGNAL,
    COOLDOWN_SECONDS_BUY, COOLDOWN_SECONDS_SELL, SLEEP_PER_LOOP,
    TRAILING_STOP_ATR_MULTIPLIER, MIN_TRAILING_STOP_PCT, MAX_TRAILING_STOP_PCT,
    DYNAMIC_UNIVERSE_CANDIDATES, UNIVERSE_SIZE, UNIVERSE_REFRESH_SECONDS,
    get_regime_params, fmt_price, trading_client,
)
from database import report_equity, init_db, save_bot_state, load_bot_state
from data_feeds import get_clean_ohlcv_dataframe, get_orderbook_with_retry, scan_stable_assets
from regime import compute_regime_and_trend, calculate_adjusted_risk
from portfolio import (
    get_all_positions_async, get_buying_power_async,
    sync_existing_positions, normalize_symbol, denormalize_symbol,
    swap_weakest_position, sell_largest_position, cancel_stale_orders_async, write_heartbeat,
    calculate_kelly_multiplier, has_pending_exit, mark_pending_exit,
)
from orders import place_order
from exit_logic import evaluate_exit
from notifications import send_discord_alert
from ml_predictor import SafeMLPredictor
from money import mul, div, qty as money_qty
from experience_capture import log_entry_experience, log_exit_outcome, log_shadow_prediction
from shadow_model import get_shadow_gbt

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

    await asyncio.to_thread(init_db)  # create DB tables once at startup (no-op if DATABASE_URL not set)
    cooldown_until, entry_time, latest_signals, highest_prices = await asyncio.to_thread(load_bot_state)
    sync_existing_positions(entry_time, highest_prices)
    
    # ── Startup: cancel any stale unfilled orders left over from a crash ──────────
    # If the bot placed a limit order then crashed before recording the fill,
    # those orders are still open on the exchange. This prevents them from
    # locking buying power or causing duplicate fills on restart.
    try:
        await cancel_stale_orders_async(timeout_minutes=3)
    except Exception as e:
        logger.warning(f"⚠️  Startup cancel_stale_orders failed (non-fatal): {e}")
    
    logger.info("🚀 Grok Apex Ironclad Bot v9 - Cutting Edge Started")

    # ── Startup cleanup: close any positions in symbols we no longer trade ──────
    # Checked against the full DYNAMIC_UNIVERSE_CANDIDATES pool, not just
    # whichever symbols happen to be in this hour's active rotation --
    # otherwise a position bought while its symbol was in the top-N would get
    # force-liquidated on every restart the moment it rotates out.
    try:
        all_pos = await asyncio.to_thread(trading_client.get_all_positions)
        active_alpaca_syms = {normalize_symbol(s) for s in DYNAMIC_UNIVERSE_CANDIDATES}
        for p in all_pos:
            market_val = float(p.market_value)
            if p.symbol not in active_alpaca_syms:
                if market_val < 1.0:
                    logger.info(f"🧹 Ignoring unsellable dust position {p.symbol} (${market_val:.4f})")
                    continue
                try:
                    await asyncio.to_thread(trading_client.close_position, p.symbol)
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
            # Fetch current equity once to ensure start_equity is valid before DB sync
            try:
                account = await asyncio.to_thread(trading_client.get_account)
                initial_equity = float(account.equity)
                if initial_equity > 0:
                    start_equity = initial_equity
                else:
                    logger.warning(f"⚠️ Initial equity from broker is <=0 (${initial_equity:.2f}), using last known start_equity from state")
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch initial equity for DB sync: {e}")

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

    # Dynamic universe state. last_universe_scan starts at 0.0 so the very
    # first cycle always scans immediately rather than waiting a full
    # UNIVERSE_REFRESH_SECONDS with an empty universe.
    active_universe    = []
    last_universe_scan = 0.0

    while True:
        try:
            write_heartbeat()
            await cancel_stale_orders_async(timeout_minutes=3)
            
            account      = await asyncio.to_thread(trading_client.get_account)
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

            await asyncio.to_thread(report_equity, BOT_NAME, equity)

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

            current_positions   = await get_all_positions_async()
            # Exclude true dust (< MIN_POSITION_USD) from open_count so they
            # don't block new entries when they can't be sold anyway.
            open_count          = sum(1 for p in current_positions.values() if p["market_value"] >= MIN_POSITION_USD)
            total_value         = sum(p["market_value"] for p in current_positions.values())
            buying_power        = await get_buying_power_async()
            max_portfolio_value = equity * BASE_RISK_PERCENT * MAX_OPEN_POSITIONS
            
            # running_portfolio_value tracks TOTAL portfolio exposure (position value only),
            # used for the headroom check against max_portfolio_value.
            # Note: max_portfolio_value is based on total equity, while position exposure
            # is only the long positions. The difference represents unused cash/buying power.
            # This comparison is intentional: we limit position EXPOSURE, not total equity.
            # However, running_portfolio_value must include cash proceeds from sells
            # to accurately track how much position value has changed within this cycle.
            # Starting from total_value (current exchange position values) is correct
            # because any sells within this cycle will decrement it, and buys will increment it.
            
            drawdown_str = f"{drawdown:.2f}%" if drawdown is not None else "N/A"
            logger.info(
                f"Cycle | Positions: {open_count}/{MAX_OPEN_POSITIONS} | "
                f"Position Value: ${total_value:.2f} | Cash/BP: ${buying_power:.2f} | "
                f"Total Equity: ${equity:.2f} | Drawdown: {drawdown_str} | "
                f"Position Cap: ${max_portfolio_value:.2f}"
            )

            buys_allowed            = True
            running_portfolio_value = total_value  # Position exposure only

            if total_value >= max_portfolio_value:
                logger.warning(f"Position exposure ${total_value:.2f} >= cap ${max_portfolio_value:.2f} (equity=${equity:.2f})")
                await sell_largest_position()
                buys_allowed = False

            if open_count >= MAX_OPEN_POSITIONS:
                logger.info(f"🛑 Max positions reached ({open_count}/{MAX_OPEN_POSITIONS}). Holding off on buys.")
                buys_allowed = False

            now = time.time()

            if now - last_universe_scan >= UNIVERSE_REFRESH_SECONDS:
                try:
                    active_universe = await scan_stable_assets(
                        limit_scope=UNIVERSE_SIZE, candidates=DYNAMIC_UNIVERSE_CANDIDATES
                    )
                    last_universe_scan = now
                    logger.info(
                        f"🔍 Universe refreshed ({len(active_universe)} of "
                        f"{len(DYNAMIC_UNIVERSE_CANDIDATES)} candidates by 24h volume): {active_universe}"
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Universe scan failed, keeping previous universe {active_universe}: {e}")

            # Any symbol we currently hold must stay under exit management even
            # if it has since rotated out of the active entry universe --
            # otherwise a position could go unmonitored (no stop-loss/trailing
            # stop checks) purely because its volume rank dropped.
            held_symbols = {
                denormalize_symbol(alpaca_sym)
                for alpaca_sym, p in current_positions.items()
                if p["market_value"] >= MIN_POSITION_USD
            }
            symbols_to_process = list(dict.fromkeys(active_universe + list(held_symbols)))

            # PARALLEL FETCH: Fetch all OHLCV dataframes concurrently
            fetch_tasks = [get_clean_ohlcv_dataframe(sym) for sym in symbols_to_process]
            dfs = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # ── BATCHED ML INFERENCE: Process all valid symbols in one forward pass ──
            valid_dataframes = {}
            for symbol, df in zip(symbols_to_process, dfs):
                if df is None or isinstance(df, Exception):
                    if isinstance(df, Exception):
                        logger.error(f"Failed to fetch data for {symbol}: {df}")
                    continue
                valid_dataframes[symbol] = df

            # Batch predictions for all valid symbols at once
            if valid_dataframes:
                try:
                    signals = predictor.predict_batch(valid_dataframes)
                    # Convert single signal to dict if needed
                    if isinstance(signals, (float, int)):
                        signals = {list(valid_dataframes.keys())[0]: signals}
                except Exception as ml_error:
                    logger.error(f"Batch ML inference failed: {repr(ml_error)}")
                    signals = {}
            else:
                signals = {}

            for symbol, df in valid_dataframes.items():

                alpaca_sym = normalize_symbol(symbol)

                regime, trend, atr_pct = compute_regime_and_trend(df)
                regime_params = get_regime_params(regime)
                position_size_multiplier = 1.0  # regime_flag removed (was Windows-only path)

                signal = signals.get(symbol, 0.5)
                latest_signals[symbol] = signal
                # FORCE LOG AT INFO LEVEL - this will appear 100%
                logger.info(
                    f"🔬 TEST | Asset: {symbol} | ML Signal: {signal:.4f} | "
                    f"Threshold: {regime_params['buy_signal']:.4f} (regime={regime}) | Trend: {trend}"
                )

                price                  = df["close"].iloc[-1]

                if price <= 0:
                    continue

                # ── Step 3: SHADOW challenger inference (never trades) ────────
                # If a GBT challenger artifact exists, score the SAME
                # decision-time feature row the champion just used and log
                # both probabilities for the step-4 bake-off. Failures here
                # are non-fatal and must never affect trading decisions.
                try:
                    _shadow = get_shadow_gbt()
                    if _shadow.available():
                        _feat_row = predictor.last_features.get(symbol)
                        if _feat_row:
                            _gbt_prob = _shadow.predict_row(_feat_row)
                        log_shadow_prediction(
                            symbol,
                            gbt_prob=_gbt_prob,
                            transformer_signal=signal,
                            regime=regime,
                            trend=trend,
                            atr_pct=atr_pct,
                            price=float(price),
                        )
                except Exception as sh_err:
                    logger.debug(f"Shadow inference skipped for {symbol}: {sh_err}")

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
                    held_hours = (now - entry_time.get(symbol, now)) / 3600

                    decision = evaluate_exit(
                        avg_entry=avg_entry,
                        price=price,
                        highest_seen=highest_prices.get(symbol, avg_entry),
                        held_hours=held_hours,
                        signal=signal,
                        regime=regime,
                        atr_pct=atr_pct,
                        profit_target_pct=regime_params["profit_target_pct"],
                        stop_loss_pct=regime_params["stop_loss_pct"],
                        sell_signal=regime_params["sell_signal"],
                        max_hold_hours=MAX_HOLD_HOURS,
                        min_hold_hours_before_signal=MIN_HOLD_HOURS_BEFORE_SIGNAL,
                        trailing_stop_atr_multiplier=TRAILING_STOP_ATR_MULTIPLIER,
                        min_trailing_stop_pct=MIN_TRAILING_STOP_PCT,
                        max_trailing_stop_pct=MAX_TRAILING_STOP_PCT,
                    )
                    pnl_pct              = decision.pnl_pct
                    highest_seen         = decision.highest_seen
                    highest_prices[symbol] = highest_seen
                    exit_reason          = decision.exit_reason

                    if exit_reason and has_pending_exit(symbol):
                        # A sell was already submitted for this exact position
                        # last cycle (or the one before) and Alpaca still
                        # reports it as fully held -- the limit order just
                        # hasn't filled yet. Without this check, the same
                        # exit condition would refire every cycle and submit
                        # a duplicate full-size SELL on top of the pending
                        # one, repeating until cancel_stale_orders_async's
                        # 3-minute cleanup catches up.
                        logger.info(f"⏳ {exit_reason} — {symbol} sell already pending, skipping duplicate submission")
                    elif exit_reason:
                        logger.info(f"{exit_reason} — SELL {symbol} @ {fmt_price(price)} | Regime: {regime}")
                        _oid = {}
                        success = await place_order(symbol, OrderSide.SELL, qty_held, price, avg_entry=avg_entry, order_id_out=_oid)
                        if success:
                            task = asyncio.create_task(send_discord_alert(
                                title=f"🔴 SELL {symbol}",
                                description=f"**Price:** ${price:.4f}\n**Reason:** {exit_reason}\n**Regime:** {regime}",
                                color=0xFF0000
                            ))
                            task.add_done_callback(lambda t: (
                                logger.error(f"Discord alert failed: {t.exception()}")
                                if not t.cancelled() and t.exception() else None
                            ))
                            cooldown_until[symbol] = now + COOLDOWN_SECONDS_SELL
                            mark_pending_exit(symbol)
                            # Step 0: record the realized outcome paired with
                            # this position's decision-time context. The
                            # matching entry row (with features) was logged at
                            # BUY time; training joins them by symbol/time.
                            log_exit_outcome(
                                symbol,
                                avg_entry=float(avg_entry),
                                price=float(price),
                                qty=float(qty_held),
                                exit_reason=str(exit_reason),
                                regime=regime,
                                held_hours=float(held_hours),
                                pnl_pct=float(pnl_pct),
                                order_id=_oid.get("order_id"),
                            )
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
                        # A fully one-sided book (either side empty) is itself
                        # an extreme-condition signal: treat it as a veto
                        # rather than defaulting to imbalance=1.0.
                        if total_bid_size <= 0 or total_ask_size <= 0:
                            logger.info(f"🚫 BUY skipped {symbol}: One-sided orderbook "
                                        f"(bids={total_bid_size}, asks={total_ask_size})")
                            _whale_veto = True
                            imbalance = 0.0
                        else:
                            imbalance = total_bid_size / total_ask_size
                        
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
                        sold_notional = await swap_weakest_position(symbol, signal, latest_signals)
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
                    qty           = money_qty(adjusted_risk / price)
                    trade_value   = min(mul(qty, price), MAX_SINGLE_TRADE_USD)
                    qty           = div(trade_value, price)

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
                    _oid = {}
                    success = await place_order(symbol, OrderSide.BUY, qty, price, order_id_out=_oid)
                    if success:
                        task = asyncio.create_task(send_discord_alert(
                            title=f"🟢 BUY {symbol}",
                            description=f"**Price:** ${price:.4f}\n**Signal:** {signal:.4f}\n**Size:** ${trade_value:.2f}\n**Regime:** {regime}",
                            color=0x00FF00
                        ))
                        task.add_done_callback(lambda t: (
                            logger.error(f"Discord alert failed: {t.exception()}")
                            if not t.cancelled() and t.exception() else None
                        ))
                        cooldown_until[symbol]   = now + COOLDOWN_SECONDS_BUY
                        entry_time[symbol]        = now
                        highest_prices[symbol]    = price
                        running_portfolio_value  += trade_value
                        open_count               += 1
                        # Step 0: snapshot the EXACT feature vector + context
                        # this buy decision was made on. predictor.last_features
                        # holds the raw last-row vector from this cycle's
                        # predict_batch() call for this symbol.
                        log_entry_experience(
                            symbol,
                            signal=signal,
                            regime=regime,
                            trend=trend,
                            atr_pct=atr_pct,
                            price=float(price),
                            qty=float(qty),
                            trade_value=float(trade_value),
                            features=predictor.last_features.get(symbol),
                            order_id=_oid.get("order_id"),
                        )
                        if open_count >= MAX_OPEN_POSITIONS:
                            logger.info("🔒 Max positions reached — no more buys this cycle.")
                            buys_allowed = False
            
            # Save state at the end of each cycle
            await asyncio.to_thread(save_bot_state, cooldown_until, entry_time, latest_signals, highest_prices)
            
            # Periodic cleanup of stale cooldown entries (older than 1 hour)
            # to prevent unbounded memory growth from historical cooldown data
            cutoff = now - 3600  # 1 hour ago
            for sym in list(cooldown_until.keys()):
                if cooldown_until[sym] < cutoff:
                    cooldown_until.pop(sym, None)

            # Prune per-symbol tracking state for symbols that are neither
            # held nor in the active entry universe, so latest_signals /
            # entry_time / highest_prices don't grow without bound as the
            # dynamic universe rotates (and save_bot_state doesn't keep
            # rewriting dead rows every cycle).
            keep_symbols = symbols_to_process
            for state in (latest_signals, entry_time, highest_prices):
                for sym in list(state.keys()):
                    if sym not in keep_symbols:
                        state.pop(sym, None)

            # ── --once test flag: exit after first full cycle ──────────────────
            if "--once" in sys.argv:
                logger.info("🏁 --once flag passed. Cycle complete. Exiting.")
                return

            # Sleep once per entire cycle, not once per asset
            await asyncio.sleep(SLEEP_PER_LOOP)

        except Exception as e:
            # logger.exception (not logger.error) so the traceback -- not
            # just the exception's string -- lands in the logs. This is the
            # blanket catch-all around the whole trading cycle: if a bug
            # ever causes a call to silently fail here (e.g. a future edit
            # accidentally mismatches an async/await pair), logger.error's
            # bare message alone wouldn't show WHERE it happened, and the
            # bot would just log this same generic line forever every 30s
            # while looking "alive" in the logs.
            logger.exception(f"Critical loop error: {e}")
            await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(run_trading_mode())
