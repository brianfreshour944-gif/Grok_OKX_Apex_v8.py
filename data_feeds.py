# data_feeds.py — Alpaca market data: asset scanner + OHLCV fetcher.

import pandas as pd
from datetime import datetime, timedelta

from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import logger, data_client, SEQUENCE_LEN


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
    Fetches the last 600 1-minute bars from Alpaca, resamples to 5-minute,
    strips zero-price candles, and returns the last SEQUENCE_LEN rows.
    Returns None if data is insufficient.
    """
    try:
        req  = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            limit=600
        )
        bars = data_client.get_crypto_bars(req).data.get(symbol, [])
        if len(bars) < SEQUENCE_LEN:
            return None

        df = pd.DataFrame([{
            "timestamp": b.timestamp,
            "open":   float(b.open   or 0),
            "high":   float(b.high   or 0),
            "low":    float(b.low    or 0),
            "close":  float(b.close  or 0),
            "volume": float(b.volume or 0),
        } for b in bars])
        df.set_index("timestamp", inplace=True)

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.resample("5min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).fillna(0)

        df = df[df["close"] > 0]   # drop zero-price candles
        if len(df) < SEQUENCE_LEN:
            return None

        return df.tail(SEQUENCE_LEN)

    except Exception as e:
        logger.error(f"Data fetch error {symbol}: {e}")
        return None
