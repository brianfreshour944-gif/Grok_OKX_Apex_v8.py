# data_feeds.py — Alpaca market data: asset scanner + OHLCV+VWAP fetcher.

import pandas as pd
from datetime import datetime, timedelta, timezone

from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import logger, data_client, SEQUENCE_LEN
import asyncio

async def get_orderbook_with_retry(symbol: str, retries: int = 3, backoff: float = 1.0):
    """Fetches the latest orderbook with exponential backoff on failure."""
    from alpaca.data.requests import CryptoLatestOrderbookRequest
    for attempt in range(retries):
        try:
            req = CryptoLatestOrderbookRequest(symbol_or_symbols=[symbol])
            return data_client.get_crypto_latest_orderbook(req)[symbol]
        except Exception as e:
            logger.warning(f"Orderbook fetch failed for {symbol} (try {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(backoff * (2 ** attempt))

async def scan_stable_assets(limit_scope: int = 35) -> list:
    """
    Dynamically scans Alpaca for the top crypto assets by 24-hour dollar volume.
    Returns a list of symbols (e.g., 'BTC/USD') limited to `limit_scope`.
    Falls back to a safe 3-symbol list if the scan fails entirely.
    """
    candidates = [
        "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "LTC/USD", "BCH/USD",
        "LINK/USD", "UNI/USD", "AVAX/USD", "DOT/USD", "AAVE/USD", "ADA/USD",
        "SHIB/USD", "ATOM/USD", "GRT/USD", "MKR/USD", "COMP/USD", "NEAR/USD",
        "XRP/USD", "BAT/USD", "CRV/USD", "SUSHI/USD", "XTZ/USD", "YFI/USD",
    ]
    try:
        volume_data = []
        start_time  = datetime.now() - timedelta(days=1)

        for symbol in candidates:
            try:
                req  = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start_time
                )
                bars = data_client.get_crypto_bars(req).data.get(symbol, [])
                if bars:
                    day_bar = bars[-1]
                    dollar_volume = float(day_bar.volume) * float(day_bar.close)
                    volume_data.append({"symbol": symbol, "volume": dollar_volume})
            except Exception:
                continue

        if not volume_data:
            return ["BTC/USD", "ETH/USD", "SOL/USD"]

        df_vol      = pd.DataFrame(volume_data).sort_values("volume", ascending=False)
        top_symbols = df_vol["symbol"].head(limit_scope).tolist()
        logger.info(f"🔍 Scanner selected {len(top_symbols)} high-volume assets.")
        return top_symbols

    except Exception as e:
        logger.error(f"Scanner exception: {e}")
        return ["BTC/USD", "ETH/USD", "SOL/USD"]


async def get_clean_ohlcv_dataframe(symbol):
    """
    Fetches native 15-MINUTE bars from Alpaca and returns the last SEQUENCE_LEN.

    CRITICAL: This MUST match train_transformer.py's BAR_TIMEFRAME
    (TimeFrame(15, TimeFrameUnit.Minute)). The transformer was trained on
    15-min bar statistics — feeding it resampled 5-min bars (the old 1-min→5-min
    resample) puts every feature out-of-distribution and collapses the model
    output toward ~0.5 for ALL assets (no trades ever fire). Native 15-min bars
    also have server-side VWAP already volume-weighted over the real window.

    Columns returned: open, high, low, close, volume, vwap, trade_count
    Returns None if data is insufficient.
    """
    try:
        req  = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            limit=600,
        )
        bars = data_client.get_crypto_bars(req).data.get(symbol, [])
        
        # Ensure we only trade on finalized data by filtering out the current incomplete minute
        current_minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        bars = [b for b in bars if b.timestamp < current_minute]
        
        if len(bars) < SEQUENCE_LEN:
            return None

        df = pd.DataFrame([{
            "timestamp":   b.timestamp,
            "open":        float(b.open        or 0),
            "high":        float(b.high        or 0),
            "low":         float(b.low         or 0),
            "close":       float(b.close       or 0),
            "volume":      float(b.volume      or 0),
            "vwap":        float(b.vwap        or 0),
            "trade_count": float(b.trade_count or 0),
        } for b in bars])
        df.set_index("timestamp", inplace=True)

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Where Alpaca returned vwap=0 (rare empty bars), fall back to close
        df["vwap"] = df["vwap"].where(df["vwap"] > 0, df["close"])

        df = df[df["close"] > 0]   # drop zero-price candles
        if len(df) < SEQUENCE_LEN:
            return None

        return df.tail(SEQUENCE_LEN)

    except Exception as e:
        logger.error(f"Data fetch error {symbol}: {e}")
        return None
